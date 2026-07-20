#!/usr/bin/env python3
"""Validate one DV checkpoint and its ONNX export on identical events."""

import argparse
import csv
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dv_training_utils import H5EventDataset, collate_graphs
from DisplacedVertex_results_tune import (
    build_displaced_vertex_model_from_checkpoint,
    build_eval_indices,
    empty_hist_state,
    make_ort_feed,
    make_ort_session,
    operating_point_at_target_fpr_from_hist,
    roc_curve_from_hist,
    scores_from_onnx_output,
    update_hist_state,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--data-glob", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--split-key", default="val_idx")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--metric-bins", type=int, default=200000)
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--print-every", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--pytorch-only", action="store_true",
                    help="Run only full PyTorch metrics (use after a paired parity sample).")
    args = ap.parse_args()

    # These graphs are small; large OpenMP thread pools add substantial
    # per-event launch overhead during single-graph parity inference.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p).resolve() for p in sorted(glob.glob(args.data_glob))]
    if not paths:
        raise SystemExit(f"No files matched {args.data_glob!r}")

    model_args = SimpleNamespace(
        normalize_node_features=None, normalize_edge_features=None,
        feature_stats_json=None, feature_norm_kind=None, feature_norm_clip=None,
    )
    model, ckpt = build_displaced_vertex_model_from_checkpoint(
        Path(args.checkpoint).resolve(), internal_dtype=torch.float32, args=model_args
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = model.to(device).eval()
    pytorch_model = model.model
    session = None if args.pytorch_only else make_ort_session(
        Path(args.onnx).resolve(), no_cuda=args.no_cuda, intra_threads=1, inter_threads=1
    )
    output_names = [] if session is None else [x.name for x in session.get_outputs()]

    split = np.load(Path(args.split_file).resolve(), allow_pickle=True)
    dataset = H5EventDataset(
        [str(p) for p in paths], event_refs=split["event_refs"], labels=split["labels"],
        dataset_names=split["dataset_names"] if "dataset_names" in split.files else None,
        root_files=split["root_files"] if "root_files" in split.files else None,
        max_open_h5_files=16,
    )
    index_args = SimpleNamespace(
        split_key=args.split_key, split_file=args.split_file,
        strict_split_check=False,
        max_val_events=args.max_events if args.max_events > 0 else None,
    )
    indices, _ = build_eval_indices(index_args, paths, dataset)
    file_ids = np.fromiter((dataset.index[int(i)][0] for i in indices), dtype=np.int32, count=len(indices))
    indices = indices[np.argsort(file_ids, kind="stable")]

    pt_hist = empty_hist_state(args.metric_bins)
    onnx_hist = empty_hist_state(args.metric_bins)
    max_abs_logit = 0.0
    sum_abs_logit = 0.0
    max_abs_prob = 0.0
    sum_abs_prob = 0.0

    with torch.inference_mode():
        for start in range(0, len(indices), max(1, args.batch_size)):
            batch_indices = indices[start:start + max(1, args.batch_size)]
            items = [dataset[int(idx)] for idx in batch_indices]
            batch = collate_graphs(items)
            pt_logit = pytorch_model(
                batch["x"].to(device).float(),
                batch["edge_index"].to(device).long(),
                batch["edge_attr"].to(device).float(),
                n_muon_nodes=batch["n_muon_nodes"].to(device).long(),
                batch=batch["batch"].to(device).long(),
                node_is_muon=batch["node_is_muon"].to(device).bool(),
                edge_dropout_p=0.0,
            ).detach().cpu().numpy().reshape(-1)
            pt_prob = (1.0 / (1.0 + np.exp(-pt_logit.astype(np.float64)))).astype(np.float32)
            label = batch["y"].numpy().astype(np.int64).reshape(-1)
            update_hist_state(pt_hist, pt_prob, label)
            if session is not None:
                ort_logit = np.concatenate([
                    np.asarray(session.run(output_names, make_ort_feed(session, item))[0]).reshape(-1)
                    for item in items
                ])
                ort_prob = scores_from_onnx_output(ort_logit, single_output_mode="logit")
                update_hist_state(onnx_hist, ort_prob, label)
                dl = np.abs(pt_logit.astype(np.float64) - ort_logit.astype(np.float64))
                dp = np.abs(pt_prob.astype(np.float64) - ort_prob.astype(np.float64))
                max_abs_logit = max(max_abs_logit, float(dl.max()))
                sum_abs_logit += float(dl.sum())
                max_abs_prob = max(max_abs_prob, float(dp.max()))
                sum_abs_prob += float(dp.sum())
            done = min(start + len(batch_indices), len(indices))
            if args.print_every and (done % args.print_every == 0 or done == len(indices)):
                print(f"[parity] {done}/{len(indices)} events", flush=True)

    def metrics(hist):
        op = operating_point_at_target_fpr_from_hist(hist["hist_pos"], hist["hist_neg"], args.target_fpr)
        fpr, tpr, _ = roc_curve_from_hist(hist["hist_pos"], hist["hist_neg"])
        auc = float(np.trapezoid(tpr, fpr))
        return {"auc": auc, **op}

    pt = metrics(pt_hist)
    n = int(len(indices))
    if args.pytorch_only:
        result = {"num_events": n, "device_pytorch": str(device), "pytorch": pt}
        (out / "pytorch_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (out / "pytorch_validation.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "pytorch"])
            w.writeheader()
            for key in ("auc", "tpr", "fpr", "accuracy", "threshold"):
                w.writerow({"metric": key, "pytorch": pt[key]})
        print(json.dumps(result, indent=2), flush=True)
        return

    ox = metrics(onnx_hist)
    result = {
        "num_events": n,
        "device_pytorch": str(device),
        "providers_onnx": session.get_providers(),
        "pytorch": pt,
        "onnx": ox,
        "difference_onnx_minus_pytorch": {k: float(ox[k] - pt[k]) for k in ("auc", "tpr", "fpr", "accuracy", "threshold")},
        "max_abs_logit_difference": max_abs_logit,
        "mean_abs_logit_difference": sum_abs_logit / max(n, 1),
        "max_abs_probability_difference": max_abs_prob,
        "mean_abs_probability_difference": sum_abs_prob / max(n, 1),
    }
    (out / "pytorch_onnx_parity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out / "pytorch_onnx_parity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "pytorch", "onnx", "difference"])
        w.writeheader()
        for key in ("auc", "tpr", "fpr", "accuracy", "threshold"):
            w.writerow({"metric": key, "pytorch": pt[key], "onnx": ox[key], "difference": ox[key] - pt[key]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
