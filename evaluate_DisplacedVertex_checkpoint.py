#!/usr/bin/env python3
"""Evaluate a DisplacedVertex PyTorch checkpoint directly on H5 files.

This is the non-ONNX path for environments where onnxruntime is unavailable.
It evaluates every event matched by --data-glob and reports the same key
low-FPR operating point used during training.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dv_training_utils import (
    DisplacedVertexGNN,
    H5EventDataset,
    binary_auc_np,
    collate_one,
    load_feature_stats_json,
    tpr_at_fpr_np,
)


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ckpt_get(ckpt: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in ckpt:
        return ckpt[key]
    hp = ckpt.get("hyper_parameters", {})
    if isinstance(hp, dict) and key in hp:
        return hp[key]
    return default


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return state_dict


def resolve_input_paths(data_glob: str) -> list[Path]:
    paths = sorted(
        {Path(p).expanduser().resolve() for p in glob.glob(data_glob)}
    )
    paths = [p for p in paths if p.exists() and p.is_file() and p.suffix.lower() in (".h5", ".hdf5")]
    if not paths:
        raise RuntimeError(f"No H5 files matched --data-glob {data_glob!r}")
    return paths

def limit_input_paths(paths: list[Path], max_events: int) -> tuple[list[Path], int]:
    """Return the shortest file prefix containing at least max_events events."""
    if max_events <= 0:
        return paths, -1

    selected: list[Path] = []
    event_count = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            n_events = len(f["events"]) if "events" in f else 0
        selected.append(path)
        event_count += n_events
        if event_count >= max_events:
            break

    if event_count < max_events:
        return paths, event_count
    return selected, event_count


def path_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def make_run_manifest(
    args: argparse.Namespace, ckpt_path: Path, input_paths: list[Path]
) -> dict[str, Any]:
    stats_path = Path(args.feature_stats_json) if args.feature_stats_json else None
    return {
        "format_version": 1,
        "checkpoint": path_identity(ckpt_path),
        "feature_stats": path_identity(stats_path),
        "feature_norm_kind": args.feature_norm_kind,
        "max_events": int(args.max_events),
        "amp": bool(args.amp),
        "no_cuda": bool(args.no_cuda),
        "inputs": [path_identity(path) for path in input_paths],
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def load_prediction_chunk(path: Path, source_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        labels = np.asarray(payload["labels"], dtype=np.float32)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        seconds = float(np.asarray(payload["seconds"]).reshape(-1)[0])
        stored_source = str(np.asarray(payload["source_path"]).reshape(-1)[0])
    if stored_source != str(source_path) or labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Prediction chunk metadata is invalid.")
    if labels.shape[0] != scores.shape[0] or labels.shape[0] == 0:
        raise ValueError("Prediction chunk arrays are empty or have different lengths.")
    if not np.all(np.isin(labels, [0.0, 1.0])) or not np.all(np.isfinite(scores)):
        raise ValueError("Prediction chunk contains invalid labels or scores.")
    return labels, scores, seconds


def atomic_save_prediction_chunk(
    path: Path, source_path: Path, labels: np.ndarray, scores: np.ndarray, seconds: float
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f, labels=labels, scores=scores, seconds=np.asarray([seconds]),
            source_path=np.asarray([str(source_path)]),
        )
    os.replace(tmp_path, path)


def checkpoint_uses_buffer(state_dict: dict[str, Any], name: str) -> bool:
    return name in state_dict or f"module.{name}" in state_dict


def build_model(ckpt_path: Path, args: argparse.Namespace) -> tuple[DisplacedVertexGNN, dict[str, Any]]:
    ckpt = safe_torch_load(ckpt_path)
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain 'model_state'.")

    state = strip_module_prefix(ckpt["model_state"])
    xdim = int(ckpt_get(ckpt, "xdim", 7))
    edim = int(ckpt_get(ckpt, "edim", 5))
    norm_node = checkpoint_uses_buffer(state, "mu_center") or bool(ckpt_get(ckpt, "normalize_node_features", False))
    norm_edge = checkpoint_uses_buffer(state, "edge_center") or bool(ckpt_get(ckpt, "normalize_edge_features", False))
    norm_kind = str(args.feature_norm_kind or ckpt_get(ckpt, "feature_norm_kind", "standard"))
    norm_clip = float(ckpt_get(ckpt, "feature_norm_clip", -1.0))

    feature_stats = ckpt_get(ckpt, "feature_stats", None)
    if feature_stats is None and (norm_node or norm_edge):
        stats_json = args.feature_stats_json or ckpt_get(ckpt, "feature_stats_json", None)
        if stats_json:
            feature_stats = load_feature_stats_json(str(stats_json), norm_kind=norm_kind)

    model = DisplacedVertexGNN(
        xdim=xdim,
        edim=edim,
        hdim=int(ckpt_get(ckpt, "hidden_dim", 128)),
        n_layers=int(ckpt_get(ckpt, "layers", ckpt_get(ckpt, "num_layers", 4))),
        dropout=float(ckpt_get(ckpt, "dropout", 0.1)),
        layer_type=str(ckpt_get(ckpt, "layer_type", "mpnn")),
        gat_heads=int(ckpt_get(ckpt, "gat_heads", 4)),
        gat_edge_attn=bool(ckpt_get(ckpt, "gat_edge_attn", False)),
        sage_aggr=str(ckpt_get(ckpt, "sage_aggr", "mean")),
        edgeconv_aggr=str(ckpt_get(ckpt, "edgeconv_aggr", "mean")),
        pool=str(ckpt_get(ckpt, "pool", "meanmax")),
        use_fourier=bool(ckpt_get(ckpt, "fourier", False)),
        fourier_base=float(ckpt_get(ckpt, "fourier_base", 3.0)),
        fourier_min_exp=int(ckpt_get(ckpt, "fourier_min_exp", -6)),
        fourier_max_exp=int(ckpt_get(ckpt, "fourier_max_exp", 6)),
        normalize_node_features=norm_node,
        normalize_edge_features=norm_edge,
        feature_stats=feature_stats,
        feature_norm_clip=norm_clip,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state_dict mismatch. Missing={missing}; unexpected={unexpected}"
        )
    return model, ckpt


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--feature-stats-json", default=None)
    ap.add_argument("--feature-norm-kind", default=None, choices=["standard", "robust"])
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--target-fprs", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.02])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-cuda", action="store_true", default=False)
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument("--print-every", type=int, default=10000)
    ap.add_argument(
        "--resume", action="store_true", default=False,
        help="Reuse validated per-file prediction chunks from output-dir.",
    )
    args = ap.parse_args()

    ckpt_path = args.ckpt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_input_paths = resolve_input_paths(args.data_glob)
    input_paths, selected_event_count = limit_input_paths(all_input_paths, int(args.max_events))
    if args.max_events > 0:
        print(
            f"[data] selected_files={len(input_paths)}/{len(all_input_paths)} "
            f"selected_file_events={selected_event_count} max_events={args.max_events}",
            flush=True,
        )
    print(f"[data] files={len(input_paths)} data_glob={args.data_glob}", flush=True)

    cache_dir = output_dir / "prediction_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest = make_run_manifest(args, ckpt_path, input_paths)
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing_manifest = json.load(f)
        if existing_manifest != manifest:
            raise RuntimeError(
                f"Existing manifest {manifest_path} does not match this evaluation. "
                "Use the original arguments or choose a new --output-dir."
            )
        if not args.resume:
            raise RuntimeError(
                f"Resumable state already exists in {output_dir}; pass --resume to continue."
            )
        print(f"[resume] validated {manifest_path}", flush=True)
    else:
        orphan_chunks = list(cache_dir.glob("*.npz"))
        if orphan_chunks:
            raise RuntimeError(
                f"Found prediction chunks without a manifest in {cache_dir}; "
                "choose a new --output-dir."
            )
        atomic_write_json(manifest_path, manifest)
        print(f"[resume] wrote {manifest_path}", flush=True)

    model, ckpt = build_model(ckpt_path, args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device).eval()
    layer_type = ckpt_get(ckpt, "layer_type", "unknown")
    gat_edge_attn = ckpt_get(ckpt, "gat_edge_attn", False)
    print(
        "[model] "
        f"layer_type={layer_type} "
        f"gat_edge_attn={gat_edge_attn} "
        f"device={device}",
        flush=True,
    )

    label_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    processed = 0
    cached_files = 0
    total_seconds = 0.0
    remaining = int(args.max_events) if args.max_events > 0 else None
    t0 = time.time()

    for file_index, input_path in enumerate(input_paths):
        if remaining is not None and remaining <= 0:
            break
        chunk_path = cache_dir / f"{file_index:06d}.npz"

        if args.resume and chunk_path.exists():
            try:
                file_labels, file_scores, file_seconds = load_prediction_chunk(
                    chunk_path, input_path
                )
                if remaining is not None and len(file_labels) > remaining:
                    raise ValueError("Cached chunk exceeds the remaining event limit.")
            except Exception as exc:
                print(f"[resume] invalid {chunk_path}: {exc}; recomputing", flush=True)
            else:
                label_chunks.append(file_labels)
                score_chunks.append(file_scores)
                processed += len(file_labels)
                total_seconds += file_seconds
                cached_files += 1
                if remaining is not None:
                    remaining -= len(file_labels)
                print(
                    f"[resume] file={file_index + 1}/{len(input_paths)} "
                    f"events={len(file_labels)} total_events={processed}",
                    flush=True,
                )
                continue

        dataset = H5EventDataset([str(input_path)])
        n_file_events = len(dataset)
        if remaining is not None:
            n_file_events = min(n_file_events, remaining)
        loader = DataLoader(
            torch.utils.data.Subset(dataset, range(n_file_events)),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_one,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

        file_labels_list: list[float] = []
        file_scores_list: list[float] = []
        file_t0 = time.time()
        with torch.no_grad():
            for local_index, batch in enumerate(loader, start=1):
                x = batch["x"].to(device, non_blocking=True).float()
                edge_index = batch["edge_index"].to(device, non_blocking=True).long()
                edge_attr = batch["edge_attr"].to(device, non_blocking=True).float()
                n_muon_nodes = batch["n_muon_nodes"].to(device, non_blocking=True).long()
                y = float(batch["y"].view(-1)[0].item())
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    logit = model(
                        x, edge_index, edge_attr, n_muon_nodes=n_muon_nodes,
                        edge_dropout_p=0.0,
                    ).view(-1)[0]
                file_labels_list.append(y)
                file_scores_list.append(float(logit.detach().float().cpu().item()))
                global_index = processed + local_index
                if args.print_every > 0 and global_index % args.print_every == 0:
                    dt = max(time.time() - t0, 1e-9)
                    target = str(args.max_events) if args.max_events > 0 else "unknown"
                    print(
                        f"[eval] {global_index}/{target} events "
                        f"({global_index / dt:.2f} events/s this invocation)",
                        flush=True,
                    )

        file_labels = np.asarray(file_labels_list, dtype=np.float32)
        file_scores = np.asarray(file_scores_list, dtype=np.float64)
        file_seconds = time.time() - file_t0
        atomic_save_prediction_chunk(
            chunk_path, input_path, file_labels, file_scores, file_seconds
        )
        label_chunks.append(file_labels)
        score_chunks.append(file_scores)
        processed += len(file_labels)
        total_seconds += file_seconds
        if remaining is not None:
            remaining -= len(file_labels)
        print(
            f"[file] completed={file_index + 1}/{len(input_paths)} "
            f"events={len(file_labels)} total_events={processed} saved={chunk_path}",
            flush=True,
        )

    if not label_chunks:
        raise RuntimeError("No predictions were produced or loaded.")
    labels_np = np.concatenate(label_chunks).astype(np.float32, copy=False)
    scores_np = np.concatenate(score_chunks).astype(np.float64, copy=False)
    n_eval = len(labels_np)
    print(
        f"[data] evaluated_events={n_eval} cached_files={cached_files} "
        f"prediction_seconds={total_seconds:.3f}",
        flush=True,
    )
    probs_np = 1.0 / (1.0 + np.exp(-scores_np))
    pred_np = (probs_np >= float(args.threshold)).astype(np.int64)
    y_np = labels_np.astype(np.int64)

    tp = int(np.count_nonzero((pred_np == 1) & (y_np == 1)))
    fp = int(np.count_nonzero((pred_np == 1) & (y_np == 0)))
    tn = int(np.count_nonzero((pred_np == 0) & (y_np == 0)))
    fn = int(np.count_nonzero((pred_np == 0) & (y_np == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    auc = binary_auc_np(labels_np, scores_np)
    summary = {
        "ckpt": str(ckpt_path),
        "data_glob": args.data_glob,
        "n_files": len(input_paths),
        "n_events": int(n_eval),
        "n_signal": int(np.count_nonzero(y_np == 1)),
        "n_background": int(np.count_nonzero(y_np == 0)),
        "threshold": float(args.threshold),
        "acc": float((tp + tn) / max(len(y_np), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "balanced_acc": float(0.5 * (recall + specificity)),
        "auc": float(auc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "seconds": float(total_seconds),
    }
    write_csv(output_dir / "model_summary.csv", [summary])

    op_rows = []
    for target_fpr in args.target_fprs:
        op = tpr_at_fpr_np(labels_np, scores_np, target_fpr=target_fpr)
        op_rows.append({
            "ckpt": str(ckpt_path),
            "target_fpr": float(target_fpr),
            "tpr": float(op["tpr"]),
            "fpr": float(op["fpr"]),
            "threshold_logit": float(op["threshold"]),
            "threshold_prob": float(1.0 / (1.0 + math.exp(-float(op["threshold"])))) if np.isfinite(op["threshold"]) else float("nan"),
            "n_pos": int(op["n_pos"]),
            "n_neg": int(op["n_neg"]),
        })
    write_csv(output_dir / "tpr_at_target_fpr.csv", op_rows)

    print(f"[done] wrote {output_dir / 'model_summary.csv'}", flush=True)
    print(f"[done] wrote {output_dir / 'tpr_at_target_fpr.csv'}", flush=True)
    for row in op_rows:
        print(
            f"[metric] target_fpr={row['target_fpr']} tpr={row['tpr']:.6f} "
            f"fpr={row['fpr']:.6f} threshold_prob={row['threshold_prob']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
