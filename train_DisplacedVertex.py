#!/usr/bin/env python3
"""
DDP multi-GPU training for displaced-vertex graph-level binary classification.

Input HDF5 schema is produced by DisplacedVertex_converter.py:
  /events/<id>/x          [N, 7]
  /events/<id>/edge_index [2, E]
  /events/<id>/edge_attr  [E, 5]
  /events/<id>/y          [1], 0=background, 1=signal

When --normalize-node-features / --normalize-edge-features are enabled,
the trainer stores the statistics as model buffers and applies normalization
inside DisplacedVertexGNN.forward(). Inference should pass raw x/edge_attr.

Example:
torchrun --standalone --nproc_per_node=8 train_DisplacedVertex.py \
    --data-glob "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/*.h5" \
    --split-file "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz" \
    --feature-stats-json "/shared/wp2p5/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json" \
    --normalize-node-features \
    --normalize-edge-features \
    --epochs 60 \
    --lr 0.00034882957103492035 \
    --hidden-dim 128 \
    --layers 5 \
    --dropout 0.020720971979495448 \
    --layer-type gat_residual \
    --gat-heads 2 \
    --gatv2-edge-attn \
    --num-workers 4 \
    --pin-memory \
    --wandb \
    --wandb-project "DisplacedVertex" \
    --wandb-name "dv_binary_classifier" \
    --early-stop-monitor val_tpr_at_target_fpr \
    --pos-weight auto \
    --no-fourier \
    --weight-decay 2.8684229445023318e-06 \
    --edge-dropout 0.033758773173194756 \
    --feat-noise-std 0.0018684258382055787 \
    --ema \
    --ema-decay 0.999 \
    --save-dir "models" \
    --save "displaced_vertex_classifier_gnn.pt"
"""

import argparse

from dv_training_utils import add_training_args, run_training, ddp_cleanup


EXAMPLE = __doc__


def main():
    ap = argparse.ArgumentParser(
        description="DDP multi-GPU training for displaced-vertex graph-level binary classification.",
        epilog=EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_training_args(ap)
    ap.set_defaults(
        save="displaced_vertex_classifier_gnn.pt",
        early_stop_monitor="val_tpr_at_target_fpr",
        pos_weight="auto",
    )
    args = ap.parse_args()
    run_training(args, task_name="displaced_vertex_classification")


if __name__ == "__main__":
    try:
        main()
    finally:
        ddp_cleanup()
