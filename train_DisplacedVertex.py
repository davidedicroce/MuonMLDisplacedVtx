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

--use-tracks also uses the track block of the graphs (DisplacedVertex_converter.py with a track tree):
  /events/<id>/track_x          [T, 11]   raw track features (asinh transform + normalization in the model)
  /events/<id>/track_edge_index [2, Et]   muon <-> track edges, track t = N + t
  /events/<id>/track_edge_attr  [Et, 5]
The track nodes get their own encoder and a node-type embedding (muon / calo / track); the feature stats JSON
must come from the current DisplacedVertex_preproc.py run on the same graphs (tr_* / track_edge_* stats), e.g.
  python DisplacedVertex_preproc.py --input-glob "<dir>/*.h5" --stats-json <dir>/normalization_stats_raw.json \
      --max-events-per-file 200
The ONNX export (DisplacedVertex_results_tune.py) then has the inputs x, edge_index, edge_attr, n_muon_nodes,
track_x, track_edge_index, track_edge_attr (DVInferenceToolBase with TrackParticleKey/PrimaryVertexKey).

The segment blocks of the graphs (DisplacedVertex_converter.py default) are used by default (--no-use-segment-edges /
--no-use-tracklets switch them off): segment-segment edges (seg_edge_index [2, Es], seg_edge_attr [Es, 7]) and tracklets
(tracklet_x [K, 11] as a fourth node type, tracklet_edge_index [2, Ek], tracklet_edge_attr [Ek, 4]); the chi2 feature of
the muon nodes is used automatically when the graphs have it. The stats JSON must come from DisplacedVertex_preproc.py
on the same graphs (seg_edge_* / tl_* / tracklet_edge_* stats, muon_node_feature_names). The ONNX export appends
seg_edge_index, seg_edge_attr, tracklet_x, tracklet_edge_index, tracklet_edge_attr to the inputs and records the
converter settings in its metadata, from which DVInferenceToolBase takes its segment-block settings.

Data and sampler: --data-glob can also point at packed parts ("<dir>/*.packed", pack_DisplacedVertex_dataset.py;
same split file and stats JSON, identical tensors, ~20-30x faster event reads). Every H5 part holds one sample (one
class), so the training batches mix the events of several files: --sampler-mix-files (default: --max-open-h5-files
for H5, all files for packed parts; 1 = the former single-file batches).

Architecture options (all off by default = the legacy model; same ONNX inputs, no C++ change):
  --block prenorm         pre-LayerNorm residual blocks h += Drop(MP(LN(h))), h += Drop(FFN(LN(h))) (--ffn-mult),
                          final LayerNorm; with --grad-clip (global gradient-norm clipping, logged per epoch)
  --relation-mp           relation-typed message passing: one GATv2 attention with its own projections and softmax
                          per relation (muon <- calo, calo <- muon, muon <- track, track <- muon, muon <- muon,
                          muon <- tracklet, tracklet <- muon, tracklet <- tracklet), self term per node type, separate
                          muon / calo encoders; --relation-aggr attn|attn_max, --calo-context two_way|one_way
                          (one_way: towers stay at their encoder output, faster)
  --edge-update seg|seg_tracklet   (with --relation-mp) hidden states on the segment-segment / tracklet edges,
                          updated in every layer from both end nodes
  --readout typed|typed_attn       mean+max per node type (+ --readout-heads attention pooling over the muon
                          segments and the tracklets) instead of the global mean+max over all nodes
e.g. --block prenorm --grad-clip 1.0 --relation-mp --edge-update seg_tracklet --readout typed_attn

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
