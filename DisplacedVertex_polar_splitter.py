#!/usr/bin/env python3

"""
Prepare and save a deterministic train/val split for the displaced vertex
polar-coordinate dataset.

This matches the output of DisplacedVertex_polar_converter.py.

Expected node features:
    x[:, :] = [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

Expected target:
    y_vertex = [r, theta, phi]

Example
-------
python DisplacedVertex_polar_splitter.py \
  --data-glob "./data_polar/displaced_vertex_dataset_part*.h5" \
  --val-fraction 0.1 \
  --seed 12345 \
  --out ./data_polar/split_displaced_vertex_polar_seed12345.npz

Later in training:
  split = np.load("./data_polar/split_displaced_vertex_polar_seed12345.npz", allow_pickle=True)
  train_idx = split["train_idx"]
  val_idx   = split["val_idx"]

  train_ds = torch.utils.data.Subset(dataset, train_idx)
  val_ds   = torch.utils.data.Subset(dataset, val_idx)
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------
# Dataset loader (matches DisplacedVertex_polar_converter output)
# ------------------------------------------------------------

class H5PolarEventDataset(Dataset):
    """
    Dataset for displaced vertex graph data in polar coordinates.

    Expected structure in H5:
        /events/<id>/
            x
            edge_index
            edge_attr
            y_vertex

    Expected semantics:
        x        : node features in polar form
        y_vertex : target vertex in polar form [r, theta, phi]
    """

    def __init__(self, h5_paths):
        self.h5_paths = list(h5_paths)
        if not self.h5_paths:
            raise ValueError("No H5 files provided.")

        self.index = []

        for fi, p in enumerate(self.h5_paths):
            with h5py.File(p, "r") as f:
                if "events" not in f:
                    continue

                keys = sorted(list(f["events"].keys()))
                for k in keys:
                    self.index.append((fi, k))

        if not self.index:
            raise ValueError("No events found in provided H5 files.")

        self._files = None
        self._pid = None

    def __len__(self):
        return len(self.index)

    def _ensure_open(self):
        pid = os.getpid()

        if self._files is not None and self._pid == pid:
            return

        if self._files is not None:
            for f in self._files:
                try:
                    f.close()
                except Exception:
                    pass

        self._pid = pid
        self._files = [h5py.File(p, "r") for p in self.h5_paths]

    def __getitem__(self, idx):
        self._ensure_open()

        fi, k = self.index[idx]
        f = self._files[fi]
        g = f["events"][k]

        x = torch.from_numpy(g["x"][...]).float()
        edge_index = torch.from_numpy(g["edge_index"][...]).long()
        edge_attr = torch.from_numpy(g["edge_attr"][...]).float()

        if "y_vertex" not in g:
            raise RuntimeError(
                f"Missing 'y_vertex' in {self.h5_paths[fi]} /events/{k}"
            )

        y_vertex = torch.from_numpy(g["y_vertex"][...]).float()

        # Optional consistency checks for the polar format
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected node feature matrix with shape [N, F], got shape {tuple(x.shape)} "
                f"in {self.h5_paths[fi]} /events/{k}"
            )

        if x.shape[1] != 7:
            raise RuntimeError(
                f"Expected 7 polar node features "
                f"[r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF], "
                f"got {x.shape[1]} in {self.h5_paths[fi]} /events/{k}"
            )

        if y_vertex.numel() != 3:
            raise RuntimeError(
                f"Expected polar target y_vertex = [r, theta, phi], "
                f"got shape {tuple(y_vertex.shape)} in {self.h5_paths[fi]} /events/{k}"
            )

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_vertex": y_vertex,
        }


# ------------------------------------------------------------
# Split creation
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-glob",
        required=True,
        help='Glob for H5 files, e.g. "./data_polar/displaced_vertex_dataset_part*.h5"',
    )

    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of events used for validation",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for deterministic split",
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Output split file (.npz)",
    )

    ap.add_argument(
        "--max-train-events",
        type=int,
        default=-1,
        help="Optional cap for train size (debug)",
    )

    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be in the open interval (0, 1)")

    paths = sorted(glob.glob(args.data_glob))

    if not paths:
        raise SystemExit(f"No H5 files matched: {args.data_glob}")

    print(f"[i] found {len(paths)} H5 files")

    ds = H5PolarEventDataset(paths)

    n = len(ds)

    if n < 2:
        raise SystemExit("Not enough events to split")

    print(f"[i] total events: {n}")

    rng = np.random.RandomState(args.seed)

    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)

    n_val = max(1, int(args.val_fraction * n))
    n_val = min(n_val, n - 1)  # always leave at least one event for training

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