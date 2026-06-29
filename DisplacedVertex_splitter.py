#!/usr/bin/env python3
"""
Prepare and save a deterministic train/validation split for the displaced-vertex
binary graph-classification dataset.

The split is stratified by default so that train and validation preserve the
signal/background ratio.

Expected classifier H5 event schema:
  /events/<id>/x, edge_index, edge_attr, y or labels

Example:
python DisplacedVertex_splitter.py \
    --data-glob "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/*.h5" \
    --val-fraction 0.1 \
    --seed 12345 \
    --out "/eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz"
"""

import argparse
import glob
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np


def _decode_h5_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


def _read_label(g, file_path: str, event_key: str) -> float:
    if "y" in g:
        y = g["y"][...]
    elif "labels" in g:
        y = g["labels"][...]
    elif "label" in g.attrs:
        y = np.asarray([g.attrs["label"]], dtype=np.float32)
    else:
        raise RuntimeError(f"Missing classifier label in {file_path} /events/{event_key}")
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size != 1 or y[0] not in (0.0, 1.0):
        raise RuntimeError(f"Expected scalar label 0/1 in {file_path} /events/{event_key}, got {y}")
    return float(y[0])


def _read_event_metadata(paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    dataset_names = []
    root_files = []
    event_refs = []

    for file_idx, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            if "events" not in f:
                continue
            for k in sorted(f["events"].keys()):
                g = f["events"][k]
                labels.append(_read_label(g, p, k))
                dataset_names.append(_decode_h5_attr(g.attrs.get("dataset_name", "unknown")))
                root_files.append(_decode_h5_attr(g.attrs.get("root_file", Path(p).name)))
                event_refs.append((file_idx, k))

    return (
        np.asarray(labels, dtype=np.float32),
        np.asarray(dataset_names, dtype=object),
        np.asarray(root_files, dtype=object),
        np.asarray(event_refs, dtype=object),
    )


def _print_label_summary(tag: str, labels: np.ndarray):
    n = int(labels.size)
    n_pos = int(np.count_nonzero(labels == 1.0))
    n_neg = int(np.count_nonzero(labels == 0.0))
    frac = n_pos / max(n, 1)
    print(f"[i] {tag}: n={n}, signal={n_pos}, background={n_neg}, signal_fraction={frac:.6f}")


def _print_dataset_summary(tag: str, names: np.ndarray, labels: np.ndarray):
    print(f"[i] {tag} dataset composition:")
    uniq = np.unique(names)
    rows = []
    for u in uniq:
        m = names == u
        rows.append((int(np.count_nonzero(m)), str(u), int(np.count_nonzero(labels[m] == 1.0)), int(np.count_nonzero(labels[m] == 0.0))))
    for n, name, sig, bkg in sorted(rows, reverse=True):
        print(f"    {name}: n={n}, signal={sig}, background={bkg}")


def _stratified_split_indices(labels: np.ndarray, val_fraction: float, rng: np.random.RandomState):
    train_parts = []
    val_parts = []
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(labels == cls).astype(np.int64)
        rng.shuffle(idx)
        if idx.size == 0:
            continue
        n_val = int(round(val_fraction * idx.size))
        if idx.size >= 2:
            n_val = min(max(1, n_val), idx.size - 1)
        else:
            n_val = 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts) if train_parts else np.zeros(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.zeros(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def _random_split_indices(n: int, val_fraction: float, rng: np.random.RandomState):
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)
    return indices[n_val:].astype(np.int64), indices[:n_val].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True, help='Glob for H5 files, e.g. "./data/*.h5"')
    ap.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of events used for validation")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic split")
    ap.add_argument("--out", required=True, help="Output split file (.npz)")
    ap.add_argument("--max-train-events", type=int, default=-1, help="Optional cap for train size/debug")
    ap.add_argument("--no-stratify", dest="stratify", action="store_false", default=True,
                    help="Disable label-stratified splitting")
    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in the open interval (0, 1)")

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")
    print(f"[i] found {len(paths)} H5 files")

    labels, dataset_names, root_files, event_refs = _read_event_metadata(paths)
    n = int(labels.size)
    if n < 2:
        raise SystemExit("Not enough events to split")

    _print_label_summary("total", labels)
    rng = np.random.RandomState(args.seed)
    if args.stratify:
        train_idx, val_idx = _stratified_split_indices(labels, args.val_fraction, rng)
        if train_idx.size == 0 or val_idx.size == 0:
            raise SystemExit("Stratified split produced an empty train or validation set. Try --no-stratify or more data.")
    else:
        train_idx, val_idx = _random_split_indices(n, args.val_fraction, rng)

    if args.max_train_events > 0:
        train_idx = train_idx[: args.max_train_events]

    _print_label_summary("train", labels[train_idx])
    _print_label_summary("val", labels[val_idx])
    _print_dataset_summary("train", dataset_names[train_idx], labels[train_idx])
    _print_dataset_summary("val", dataset_names[val_idx], labels[val_idx])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    abs_paths = np.array([str(Path(p).resolve()) for p in paths], dtype=object)

    np.savez(
        out_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        seed=np.int64(args.seed),
        val_fraction=np.float64(args.val_fraction),
        stratified=np.bool_(args.stratify),
        data_glob=np.array(args.data_glob, dtype=object),
        h5_paths=abs_paths,
        labels=labels.astype(np.float32),
        dataset_names=dataset_names,
        root_files=root_files,
        event_refs=event_refs,
        task=np.array("graph_classification", dtype=object),
        class_names=np.array(["background", "signal"], dtype=object),
        train_n_signal=np.int64(np.count_nonzero(labels[train_idx] == 1.0)),
        train_n_background=np.int64(np.count_nonzero(labels[train_idx] == 0.0)),
        val_n_signal=np.int64(np.count_nonzero(labels[val_idx] == 1.0)),
        val_n_background=np.int64(np.count_nonzero(labels[val_idx] == 0.0)),
    )
    print(f"[done] wrote split: {out_path}")


if __name__ == "__main__":
    main()
