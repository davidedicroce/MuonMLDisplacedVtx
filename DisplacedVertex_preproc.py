#!/usr/bin/env python3
"""
DisplacedVertex_preproc.py

Compute trainer-compatible raw-feature normalization statistics from classifier H5 graph files,
and optionally write normalized H5 copies.

Expected event structure:
  /events/<id>/
      x            [N, 7]
      edge_index   [2, E]
      edge_attr    [E, 5]
      y or labels  [1]
  attrs:
      n_muon_nodes
      n_calo_nodes
      label

Raw node feature convention:
    x[:, :] = [r_pos, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

Raw edge feature convention:
    edge_attr[:, :] = [d_energy_like, d_phi, d_eta, cos_angle, same_sector]

Example: compute stats only
python DisplacedVertex_preproc.py \
    --input-glob "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/*.h5" \
    --stats-json "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json"

Example: compute stats and write normalized copies
python DisplacedVertex_preproc.py \
    --input-glob "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/*.h5" \
    --stats-json "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json" \
    --write-normalized-h5 \
    --output-dir "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs_normalized" \
    --node-norm-kind standard \
    --edge-norm-kind standard \
    --normalize-node-features \
    --normalize-edge-features
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


RAW_NODE_FEATURE_NAMES = [
    "r_pos",
    "theta_pos",
    "phi_pos",
    "theta_dir",
    "phi_dir",
    "energy_like",
    "nCells_or_DoF",
]

EDGE_FEATURE_NAMES = [
    "d_energy_like",
    "d_phi",
    "d_eta",
    "cos_angle",
    "same_sector",
]


def _decode_h5_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


def _safe_read_dataset(group, name: str, file_path: str, event_key: str):
    try:
        return group[name][...]
    except Exception as e:
        raise RuntimeError(
            f"Failed to read dataset {name!r} in file={file_path} event={event_key}: "
            f"{type(e).__name__}: {e}"
        ) from e


def _read_label(group, file_path: str, event_key: str) -> float:
    if "y" in group:
        y = group["y"][...]
    elif "labels" in group:
        y = group["labels"][...]
    elif "label" in group.attrs:
        y = np.asarray([group.attrs["label"]], dtype=np.float32)
    else:
        raise RuntimeError(f"Missing label y/labels/attr in {file_path} /events/{event_key}")
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size != 1 or y[0] not in (0.0, 1.0):
        raise RuntimeError(f"Expected scalar label 0/1 in {file_path} /events/{event_key}, got {y}")
    return float(y[0])


def _safe_get_attr(group, name: str, file_path: str, event_key: str):
    try:
        return group.attrs[name]
    except Exception as e:
        raise RuntimeError(
            f"Failed to read attribute {name!r} in file={file_path} event={event_key}: "
            f"{type(e).__name__}: {e}"
        ) from e


def _safe_concat(chunks: List[np.ndarray], shape_tail: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if not chunks:
        return np.zeros((0,) + shape_tail, dtype=dtype)
    return np.concatenate(chunks, axis=0).astype(dtype, copy=False)


def _compute_stats(arr: np.ndarray, names: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(f"Expected shape [N, {len(names)}], got {arr.shape}")

    standard = {}
    robust = {}
    for i, name in enumerate(names):
        x = arr[:, i]
        x = x[np.isfinite(x)]
        if x.size == 0:
            mean = 0.0
            std = 1.0
            median = 0.0
            iqr = 1.0
        else:
            mean = float(np.mean(x))
            std = float(np.std(x))
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0
            q25 = float(np.percentile(x, 25))
            q50 = float(np.percentile(x, 50))
            q75 = float(np.percentile(x, 75))
            iqr = q75 - q25
            if not np.isfinite(iqr) or iqr < 1e-12:
                iqr = 1.0
            median = q50
        standard[name] = {"mean": mean, "std": std}
        robust[name] = {"median": median, "iqr": iqr}
    return standard, robust


def _apply_standardize(arr: np.ndarray, stats: Dict[str, Dict[str, float]], names: List[str], clip: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    for i, name in enumerate(names):
        mean = float(stats[name]["mean"])
        std = max(float(stats[name]["std"]), 1e-12)
        out[:, i] = (out[:, i] - mean) / std
    if clip is not None and clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def _apply_robust(arr: np.ndarray, stats: Dict[str, Dict[str, float]], names: List[str], clip: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    for i, name in enumerate(names):
        median = float(stats[name]["median"])
        iqr = max(float(stats[name]["iqr"]), 1e-12)
        out[:, i] = (out[:, i] - median) / iqr
    if clip is not None and clip > 0:
        out = np.clip(out, -clip, clip)
    return out.astype(np.float32, copy=False)


def _normalize_array(
    arr: np.ndarray,
    *,
    names: List[str],
    standard_stats: Dict[str, Dict[str, float]],
    robust_stats: Dict[str, Dict[str, float]],
    kind: str,
    clip: Optional[float],
) -> np.ndarray:
    if kind == "standard":
        return _apply_standardize(arr, standard_stats, names, clip)
    if kind == "robust":
        return _apply_robust(arr, robust_stats, names, clip)
    raise ValueError(f"Unknown normalization kind: {kind}")


def _copy_attrs(src, dst):
    for k in src.attrs.keys():
        dst.attrs[k] = src.attrs[k]


def collect_stats_from_h5(h5_paths: List[str], max_events: int = -1):
    mu_chunks = []
    ca_chunks = []
    edge_chunks = []
    labels = []
    dataset_names = []
    skipped_files = []
    skipped_events = []
    total_events = 0

    for h5_path in h5_paths:
        print(f"[i] scanning {h5_path}", flush=True)
        try:
            with h5py.File(h5_path, "r") as f:
                if "events" not in f:
                    continue
                for ek in sorted(f["events"].keys()):
                    try:
                        g = f["events"][ek]
                        x = _safe_read_dataset(g, "x", h5_path, ek).astype(np.float32, copy=False)
                        edge_attr = _safe_read_dataset(g, "edge_attr", h5_path, ek).astype(np.float32, copy=False)
                        label = _read_label(g, h5_path, ek)

                        if "n_muon_nodes" not in g.attrs:
                            raise RuntimeError(f"Missing attribute 'n_muon_nodes' in {h5_path} /events/{ek}")
                        n_mu = int(_safe_get_attr(g, "n_muon_nodes", h5_path, ek))
                        n_tot = x.shape[0]
                        if n_mu < 0 or n_mu > n_tot:
                            raise RuntimeError(
                                f"Invalid n_muon_nodes={n_mu} for x.shape[0]={n_tot} in {h5_path} /events/{ek}"
                            )

                        mu_x = x[:n_mu]
                        ca_x = x[n_mu:]
                        if mu_x.shape[0] > 0:
                            mu_chunks.append(mu_x)
                        if ca_x.shape[0] > 0:
                            ca_chunks.append(ca_x)
                        if edge_attr.shape[0] > 0:
                            edge_chunks.append(edge_attr)
                        labels.append(label)
                        dataset_names.append(_decode_h5_attr(g.attrs.get("dataset_name", "unknown")))

                        total_events += 1
                        if max_events > 0 and total_events >= max_events:
                            break
                    except Exception as e:
                        msg = f"{h5_path} :: /events/{ek} :: {type(e).__name__}: {e}"
                        skipped_events.append(msg)
                        print(f"[warn] skipping unreadable event: {msg}", flush=True)
                        continue
        except Exception as e:
            msg = f"{h5_path} :: {type(e).__name__}: {e}"
            skipped_files.append(msg)
            print(f"[warn] skipping unreadable file: {msg}", flush=True)
            continue

        if max_events > 0 and total_events >= max_events:
            break

    mu_arr = _safe_concat(mu_chunks, (len(RAW_NODE_FEATURE_NAMES),), dtype=np.float32)
    ca_arr = _safe_concat(ca_chunks, (len(RAW_NODE_FEATURE_NAMES),), dtype=np.float32)
    edge_arr = _safe_concat(edge_chunks, (len(EDGE_FEATURE_NAMES),), dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)

    n_pos = int(np.count_nonzero(labels_arr == 1.0))
    n_neg = int(np.count_nonzero(labels_arr == 0.0))
    print(f"[i] scanned events: {total_events}", flush=True)
    print(f"[i] labels:      signal={n_pos}, background={n_neg}", flush=True)
    print(f"[i] mu nodes:    {mu_arr.shape}", flush=True)
    print(f"[i] calo nodes:  {ca_arr.shape}", flush=True)
    print(f"[i] edge attrs:  {edge_arr.shape}", flush=True)
    print(f"[i] skipped files:  {len(skipped_files)}", flush=True)
    print(f"[i] skipped events: {len(skipped_events)}", flush=True)

    mu_standard, mu_robust = _compute_stats(mu_arr, RAW_NODE_FEATURE_NAMES)
    ca_standard, ca_robust = _compute_stats(ca_arr, RAW_NODE_FEATURE_NAMES)
    edge_standard, edge_robust = _compute_stats(edge_arr, EDGE_FEATURE_NAMES)

    dataset_summary = {}
    for name, label in zip(dataset_names, labels):
        row = dataset_summary.setdefault(str(name), {"n_events": 0, "n_signal": 0, "n_background": 0})
        row["n_events"] += 1
        if int(label) == 1:
            row["n_signal"] += 1
        else:
            row["n_background"] += 1

    payload = {
        "mu_standard": mu_standard,
        "ca_standard": ca_standard,
        "edge_standard": edge_standard,
        "mu_robust": mu_robust,
        "ca_robust": ca_robust,
        "edge_robust": edge_robust,
        "label_summary": {
            "n_signal": n_pos,
            "n_background": n_neg,
            "positive_fraction": float(n_pos / max(len(labels_arr), 1)),
            "pos_weight_auto": float(n_neg / max(n_pos, 1)) if n_pos > 0 else None,
            "by_dataset": dataset_summary,
        },
        "meta": {
            "task": "graph_classification",
            "node_feature_names": RAW_NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "label_names": ["background", "signal"],
            "n_events_scanned": int(total_events),
            "n_mu_nodes": int(mu_arr.shape[0]),
            "n_ca_nodes": int(ca_arr.shape[0]),
            "n_edges": int(edge_arr.shape[0]),
            "source_h5_paths": [str(Path(p).resolve()) for p in h5_paths],
            "skipped_files": skipped_files,
            "skipped_events": skipped_events,
            "schema": "raw_classifier",
        },
    }
    return payload


def write_json(payload: Dict, out_json: str):
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] wrote stats JSON: {out_path}", flush=True)


def write_normalized_h5_copies(
    h5_paths: List[str],
    output_dir: str,
    stats_payload: Dict,
    *,
    normalize_node_features: bool,
    normalize_edge_features: bool,
    node_norm_kind: str,
    edge_norm_kind: str,
    clip: Optional[float],
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_standard = stats_payload["mu_standard"]
    mu_robust = stats_payload["mu_robust"]
    ca_standard = stats_payload["ca_standard"]
    ca_robust = stats_payload["ca_robust"]
    edge_standard = stats_payload["edge_standard"]
    edge_robust = stats_payload["edge_robust"]

    for src_path in h5_paths:
        src_name = os.path.basename(src_path)
        dst_path = out_dir / src_name
        print(f"[i] writing normalized copy {dst_path}", flush=True)

        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            _copy_attrs(src, dst)
            if "events" not in src:
                continue
            dst_events = dst.create_group("events")
            for ek in sorted(src["events"].keys()):
                sg = src["events"][ek]
                dg = dst_events.create_group(ek)
                _copy_attrs(sg, dg)

                x = sg["x"][...].astype(np.float32, copy=False)
                edge_attr = sg["edge_attr"][...].astype(np.float32, copy=False)
                n_mu = int(sg.attrs["n_muon_nodes"])
                mu_x = x[:n_mu]
                ca_x = x[n_mu:]

                if normalize_node_features:
                    if mu_x.shape[0] > 0:
                        mu_x = _normalize_array(
                            mu_x,
                            names=RAW_NODE_FEATURE_NAMES,
                            standard_stats=mu_standard,
                            robust_stats=mu_robust,
                            kind=node_norm_kind,
                            clip=clip,
                        )
                    if ca_x.shape[0] > 0:
                        ca_x = _normalize_array(
                            ca_x,
                            names=RAW_NODE_FEATURE_NAMES,
                            standard_stats=ca_standard,
                            robust_stats=ca_robust,
                            kind=node_norm_kind,
                            clip=clip,
                        )
                    x = np.concatenate([mu_x, ca_x], axis=0).astype(np.float32, copy=False)

                if normalize_edge_features and edge_attr.shape[0] > 0:
                    edge_attr = _normalize_array(
                        edge_attr,
                        names=EDGE_FEATURE_NAMES,
                        standard_stats=edge_standard,
                        robust_stats=edge_robust,
                        kind=edge_norm_kind,
                        clip=clip,
                    )

                for name in sg.keys():
                    if name == "x":
                        dg.create_dataset("x", data=x, compression="gzip", compression_opts=4)
                    elif name == "edge_attr":
                        dg.create_dataset("edge_attr", data=edge_attr, compression="gzip", compression_opts=4)
                    else:
                        dg.create_dataset(name, data=sg[name][...], compression="gzip", compression_opts=4)

            dst.attrs["preprocessed_from"] = str(Path(src_path).resolve())
            dst.attrs["node_normalized"] = bool(normalize_node_features)
            dst.attrs["edge_normalized"] = bool(normalize_edge_features)
            dst.attrs["node_norm_kind"] = str(node_norm_kind)
            dst.attrs["edge_norm_kind"] = str(edge_norm_kind)
            dst.attrs["feature_norm_clip"] = -1.0 if clip is None else float(clip)
    print(f"[done] wrote normalized H5 files to: {out_dir}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute raw-feature stats from DV classifier H5 files and optionally write normalized H5 copies."
    )
    ap.add_argument("--input-glob", required=True, help="Glob for input H5 files")
    ap.add_argument("--stats-json", required=True, help="Output JSON path for computed stats")
    ap.add_argument("--max-events", type=int, default=-1, help="Limit number of events to scan (-1 = all)")
    ap.add_argument("--write-normalized-h5", action="store_true", default=False,
                    help="Also write normalized H5 copies")
    ap.add_argument("--output-dir", default=None,
                    help="Directory for normalized H5 files; required if --write-normalized-h5")
    ap.add_argument("--normalize-node-features", action="store_true", default=False)
    ap.add_argument("--normalize-edge-features", action="store_true", default=False)
    ap.add_argument("--node-norm-kind", default="standard", choices=["standard", "robust"])
    ap.add_argument("--edge-norm-kind", default="standard", choices=["standard", "robust"])
    ap.add_argument("--clip", type=float, default=-1.0,
                    help="Optional absolute clip after normalization; <=0 disables clipping")
    return ap.parse_args()


def main():
    args = parse_args()
    h5_paths = sorted(glob.glob(args.input_glob))
    if not h5_paths:
        raise SystemExit(f"No H5 files matched: {args.input_glob}")
    print(f"[i] found {len(h5_paths)} H5 files", flush=True)
    stats_payload = collect_stats_from_h5(h5_paths, max_events=args.max_events)
    write_json(stats_payload, args.stats_json)
    if args.write_normalized_h5:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --write-normalized-h5")
        clip = None if args.clip <= 0 else float(args.clip)
        write_normalized_h5_copies(
            h5_paths,
            args.output_dir,
            stats_payload,
            normalize_node_features=args.normalize_node_features,
            normalize_edge_features=args.normalize_edge_features,
            node_norm_kind=args.node_norm_kind,
            edge_norm_kind=args.edge_norm_kind,
            clip=clip,
        )


if __name__ == "__main__":
    main()
