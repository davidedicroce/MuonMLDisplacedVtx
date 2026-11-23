#!/usr/bin/env python3
"""
DisplacedVertex_converter.py

Convert ROOT files into per-file graph-classification data, one output per ROOT input file (existing outputs are
skipped, as in Bucket_converter.py).

--output-format packed (default): <stem>.packed/ parts (memory-mapped .npy arrays of the training inputs, format in
dv_converter_utils.py): the fast training input (train_DisplacedVertex.py --data-glob "<dir-out>/*.packed"); the
splitter, DisplacedVertex_preproc.py and the evaluation scripts read them as well. --output-format h5: the gzip
HDF5 files described below, which also hold the diagnostics (phi, eta, vertex_rho_mm, muon_xyz_m, track_eta, ...)
used by the event-viewer notebooks; --output-format both writes both in one pass. Packed parts and H5 files of the
same conversion have the same event keys and order (split files are interchangeable); H5 files converted earlier are
packed with pack_DisplacedVertex_dataset.py.

Graph label:
  y = 0  background
  y = 1  signal, only when the ROOT filename matches any configured signal pattern and at
         least one truth vertex in the event satisfies 800 mm < rho < 8000 mm.

Node features keep the cylindrical converter convention:
  [r, theta_pos, phi_pos, theta_dir, phi_dir, energy_like, nCells_or_DoF]

Per event group (/events/<id>) of the H5 output (the packed parts hold x, edge_index, edge_attr and the blocks):
  x [N, 7], edge_index [2, E], edge_attr [E, 5]   muon nodes first, then calo nodes (n_muon_nodes, n_calo_nodes)
  track block (inputs with a TrigFSTrackDump tree, i.e. MuonBucketDump --doTrigFSTrackDump; --track-tree-name ""
  switches it off). It leaves x/edge_index/edge_attr untouched, so muon+calo models read the files as before:
    track_x [T, 11]          TRACK_FEATURE_NAMES (dv_converter_utils.py): theta_dir, phi_dir, log_pt, q_over_pt,
                             d0, d0_sig, dz0_sintheta, dz0_sig, n_pixel_hits, n_strip_hits, chi2_ndof
    track_edge_index [2, Et] segment<->track edges (dR < --max-track-segment-dr), global indices: track t = N + t
    track_edge_attr [Et, 5]  same 5 edge features as edge_attr (energy_like of a track = pT in MeV)
    attr n_track_nodes; diagnostics track_eta, track_phi, track_min_dr
  The calo sector of the same_sector edge feature is the MS sector number, as computed online by
  DVInferenceToolBase (--calo-sector-convention cpp, default). The old converter used a phi bin that never matches
  the muon bucket sector (same_sector = 0 on every edge); --calo-sector-convention legacy --track-tree-name ""
  --no-segment-chi2-feature --no-segment-edges --no-tracklets reproduces the old output exactly.

  Segment blocks (on by default; DVInferenceToolBase takes the settings from the ONNX metadata of the model):
    chi2 feature             x[:, 5] of the muon nodes = segment chi2 / max(nDoF, 1) (0 otherwise); file attribute
                             muon_node_feature_names records the convention
    segment edges            seg_edge_index [2, Es] (muon-node indices), seg_edge_attr [Es, 7] SEGMENT_EDGE_FEATURE_NAMES:
                             all segment pairs with dR < --segment-edge-max-dr (0.4), both directions
    tracklets                tracklet_x [K, 11] TRACKLET_FEATURE_NAMES, tracklet_edge_index [2, Ek] (tracklet k = N + k,
                             like the tracks), tracklet_edge_attr [Ek, 4] TRACKLET_EDGE_FEATURE_NAMES: chains of segments
                             in different station layers (dR < --tracklet-max-dr 0.4, chord angle <
                             --tracklet-max-chord-angle 0.2, >= --tracklet-min-layers 2), linked to their segments and
                             fully connected among themselves; attr n_tracklet_nodes

Example (trigger-level inputs: HLT towers, NSW segments, trigger full-scan tracks):
python -u DisplacedVertex_converter.py \
    --dir-in /eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/MuonBucketDump_displacedVtx_mu200 \
    --dir-out /eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/displacedVtx_classifier_trk_mu200 \
    --min-tower-energy-mev 1000 \
    --max-tower-segment-dr 0.4 \
    --track-tree-name TrigFSTrackDump --min-track-pt-mev 2000 --max-track-segment-dr 0.3 \
    2>&1 | tee log_displacedvertex_converter.txt
"""

import argparse

from dv_converter_utils import (
    add_dv_classification_converter_args,
    run_dv_directory_conversion,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ROOT files to DisplacedVertex HDF5 graphs for graph-level classification."
    )
    add_dv_classification_converter_args(parser)
    args = parser.parse_args()
    run_dv_directory_conversion(args)


if __name__ == "__main__":
    main()
