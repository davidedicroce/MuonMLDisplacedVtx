#!/usr/bin/env python3
"""
Prepare and save a deterministic train/val split for the displaced vertex
cylindrical-coordinate dataset.

This matches the output of DisplacedVertex_converter_cylindrical.py.

Expected node features:
        x[:, :] = [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

Expected target:
        y_vertex = [rho, phi, z]

Example
-------
python DisplacedVertex_splitter_cylindrical.py \
    --data-glob "./data_cylindrical/displaced_vertex_dataset_part*.h5" \
    --val-fraction 0.1 \
    --seed 12345 \
    --out ./data_cylindrical/split_displaced_vertex_cylindrical_seed12345.npz

Later in training:
    split = np.load("./data_cylindrical/split_displaced_vertex_cylindrical_seed12345.npz", allow_pickle=True)
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
"""

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

from dv_training_utils import H5EventDataset


def _decode_h5_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.bytes_):
        return v.tobytes().decode("utf-8", errors="replace")
    return str(v)


def _read_dataset_names(paths):
    names = []
    for p in paths:
        with h5py.File(p, "r") as f:
            if "events" not in f:
                continue
            for k in sorted(f["events"].keys()):
                g = f["events"][k]
                raw_name = g.attrs.get("dataset_name", "unknown")
                names.append(_decode_h5_attr(raw_name))
    return np.asarray(names, dtype=object)


def _print_split_dataset_summary(tag, names):
    uniq, counts = np.unique(names, return_counts=True)
    print(f"[i] {tag} dataset composition ({len(uniq)} datasets):")
    order = np.argsort(counts)[::-1]
    for i in order:
        print(f"    {uniq[i]}: {int(counts[i])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True, help='Glob for H5 files, e.g. "./data_cylindrical/displaced_vertex_dataset_part*.h5"')
    ap.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of events used for validation")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic split")
    ap.add_argument("--out", required=True, help="Output split file (.npz)")
    ap.add_argument("--max-train-events", type=int, default=-1, help="Optional cap for train size (debug)")
    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in the open interval (0, 1)")

    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")

    print(f"[i] found {len(paths)} H5 files")

    ds = H5EventDataset(paths)
    n = len(ds)
    if n < 2:
        raise SystemExit("Not enough events to split")

    dataset_names = _read_dataset_names(paths)
    if len(dataset_names) != n:
        raise SystemExit(
            f"Dataset-name metadata mismatch: read {len(dataset_names)} names for {n} events. "
            "Please regenerate H5 files with updated converter."
        )

    print(f"[i] total events: {n}")

    rng = np.random.RandomState(args.seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)

    n_val = max(1, int(args.val_fraction * n))
    n_val = min(n_val, n - 1)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    if args.max_train_events > 0:
        train_idx = train_idx[: args.max_train_events]

    train_dataset_names = dataset_names[train_idx]
    val_dataset_names = dataset_names[val_idx]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    abs_paths = np.array([str(Path(p).resolve()) for p in paths], dtype=object)

    np.savez(
        out_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        seed=np.int64(args.seed),
        val_fraction=np.float64(args.val_fraction),
        data_glob=np.array(args.data_glob, dtype=object),
        h5_paths=abs_paths,
        n_events=np.int64(n),
        coordinate_system=np.array("cylindrical", dtype=object),
        dataset_name_by_index=dataset_names,
        train_dataset_names=train_dataset_names,
        val_dataset_names=val_dataset_names,
        node_feature_order=np.array(
            [
                "r",
                "theta_pos",
                "phi_pos",
                "theta_dir",
                "phi_dir",
                "energy_like",
                "nCells_or_DoF",
            ],
            dtype=object,
        ),
        target_order=np.array(["rho", "phi", "z"], dtype=object),
    )

    print(f"[ok] wrote split file: {out_path}")
    print(f"[i] train events: {len(train_idx)}")
    print(f"[i] val events: {len(val_idx)}")
    _print_split_dataset_summary("train", train_dataset_names)
    _print_split_dataset_summary("val", val_dataset_names)


if __name__ == "__main__":
    main()
