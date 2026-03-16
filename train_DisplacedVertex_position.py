#!/usr/bin/env python3
"""
DDP multi-GPU training for displaced-vertex graph regression.

Example:
torchrun --standalone --nproc_per_node=8 train_DisplacedVertex_position.py \
    --data-glob "./data/displaced_vertex_dataset_part*.h5" \
    --split-file "./data/split_displaced_vertex_seed12345.npz" \
    --epochs 200 \
    --lr 2e-4 \
    --hidden-dim 128 \
    --layers 4 \
    --dropout 0.1 \
    --layer-type mpnn \
    --num-workers 4 \
    --pin-memory \
    --wandb \
    --wandb-project "DisplacedVertex" \
    --wandb-name "dv_regression_v1" \
    --early-stop \
    --fourier \
    --weight-decay 0.02 \
    --edge-dropout 0.1 \
    --feat-noise-std 0.01 \
    --ema \
    --ema-decay 0.999 \
    --save-dir "models" \
    --save "displaced_vertex_gnn.pt"
"""

import argparse

from dv_training_utils import add_training_args, run_training, ddp_cleanup


COORDINATE_SYSTEM = "cartesian"
TARGET_LABELS = ["x", "y", "z"]

EXAMPLE = """Example:
torchrun --standalone --nproc_per_node=8 train_DisplacedVertex_position.py \\
    --data-glob "./data/displaced_vertex_dataset_part*.h5" \\
    --split-file "./data/split_displaced_vertex_seed12345.npz" \\
    --epochs 200 \\
    --lr 2e-4 \\
    --hidden-dim 128 \\
    --layers 4 \\
    --dropout 0.1 \\
    --layer-type mpnn \\
    --num-workers 4 \\
    --pin-memory \\
    --wandb \\
    --wandb-project "DisplacedVertex" \\
    --wandb-name "dv_regression_v1" \\
    --early-stop \\
    --fourier \\
    --weight-decay 0.02 \\
    --edge-dropout 0.1 \\
    --feat-noise-std 0.01 \\
    --ema \\
    --ema-decay 0.999 \\
    --save-dir "models" \\
    --save "displaced_vertex_gnn.pt"
"""


def main():
    ap = argparse.ArgumentParser(
        description="DDP multi-GPU training for displaced-vertex graph regression.",
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_training_args(ap)
    ap.set_defaults(save="displaced_vertex_gnn.pt")
    args = ap.parse_args()
    run_training(args, coordinate_system=COORDINATE_SYSTEM, target_labels=TARGET_LABELS)


if __name__ == "__main__":
    try:
        main()
    finally:
        ddp_cleanup()
