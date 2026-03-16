#!/usr/bin/env python3
"""
Prepare and save a deterministic train/val split for the displaced vertex polar dataset.
"""

import argparse
import glob
from pathlib import Path

import numpy as np

from dv_training_utils import H5EventDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-glob", required=True, help='Glob for H5 files, e.g. "./data_polar/displaced_vertex_dataset_part*.h5"')
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
        coordinate_system=np.array("polar", dtype=object),
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
        target_order=np.array(["r", "theta", "phi"], dtype=object),
    )

    print(f"[ok] wrote split file: {out_path}")
    print(f"[i] train events: {len(train_idx)}")
    print(f"[i] val events: {len(val_idx)}")


if __name__ == "__main__":
    main()
