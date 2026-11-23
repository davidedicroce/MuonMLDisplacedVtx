# MuonMLDisplacedVtx

Utilities and training scripts for displaced-vertex machine learning studies in the ATLAS muon spectrometer.

This repository provides an end-to-end workflow:
- convert ROOT dumps to graph datasets in HDF5,
- create deterministic train/validation splits,
- train distributed GNN regressors for displaced vertex position.

## What The Code Does

The project builds graph-based event representations for displaced-vertex regression.

- Event-level target:
	- Cartesian pipeline: predict y_vertex = [x, y, z] in meters.
	- Polar pipeline: predict y_vertex = [r, theta, phi] (r in meters, angles in radians).
- Nodes:
	- Muon segment nodes and calorimeter tower nodes.
- Edges:
	- Directed segment<->tower edges for geometrically compatible pairs.
- Training:
	- Multi-GPU DistributedDataParallel (DDP) training with optional W and B logging.

## Main Files

- DisplacedVertex_converter_cartesian.py
        - Converts ROOT files into Cartesian HDF5 graph datasets under ./data.
- DisplacedVertex_converter_cylindrical.py
        - Converts ROOT files into cylindrical-coordinate HDF5 graph datasets under ./data_cylindrical.
- DisplacedVertex_converter_polar.py
        - Converts ROOT files into polar-coordinate HDF5 graph datasets under ./data_polar.
- DisplacedVertex_splitter_cartesian.py
        - Creates deterministic train/val split files for Cartesian datasets.
- DisplacedVertex_splitter_cylindrical.py
        - Creates deterministic train/val split files for cylindrical datasets.
- DisplacedVertex_splitter_polar.py
        - Creates deterministic train/val split files for polar datasets.
- train_DisplacedVertex_position.py
        - Trains the Cartesian regression model with DDP.
- train_DisplacedVertex_position_cylindrical.py
        - Trains the cylindrical regression model with DDP.
- train_DisplacedVertex_position_polar.py
	- Trains the polar regression model with DDP.

## Prerequisites

- Python 3.10+ recommended
- One or more CUDA GPUs for training
- Python packages:
	- numpy
	- h5py
	- uproot
	- torch
	- wandb

Install dependencies (example):

```bash
python -m pip install numpy h5py uproot torch wandb
```

## Expected Input Layout

ROOT inputs should be available under a directory similar to:

```text
hdd_data/
	MuonBucketDump_H*/outputs/MuonBucketDump_group.det-muon.*root
```

You can adapt the pattern with --input-dir and --pattern.

## Run Sequence

Choose one pipeline (Cartesian or Polar). The order is always:

1. Convert ROOT -> HDF5 graph parts
2. Create deterministic split file
3. Train model with torchrun


## Displaced-vertex classifier (trigger GNN)

Event classifier of the DV trigger chain (step 2: muon segments + calorimeter towers + segment blocks; step 3:
`--use-tracks`, plus the trigger full-scan tracks), deployed through `DVInferenceToolBase` (ONNX).

| file | role |
|---|---|
| `DisplacedVertex_converter.py`, `dv_converter_utils.py` | ROOT dumps -> graphs (packed parts by default, H5 with `--output-format h5`) |
| `DisplacedVertex_splitter.py` | stratified train / validation split |
| `DisplacedVertex_preproc.py` | normalization statistics (stored in the model) |
| `train_DisplacedVertex.py`, `dv_training_utils.py` | DDP training of `DisplacedVertexGNN` (architecture options: `--help`) |
| `tune_DisplacedVertex_optuna.py` | Optuna hyperparameter / architecture search and refits |
| `DisplacedVertex_results_tune.py` | evaluation of checkpoints, ONNX export |
| `pack_DisplacedVertex_dataset.py` | packs H5 graphs converted earlier |

Dataset: `/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/displacedVtx_classifier_mu200` (`~/data_ml/...`), with
its own README (same commands and graph description).

### Commands, in order

Steps 1-3 run where the dumps are (EOS, e.g. lxplus); steps 4-9 on the GPU machine (8 GPUs assumed). Every command is
complete: only the variables of step 0 and `GPU_DATA` / `KF` of step 4 may need adapting. Optuna studies and refits
are resumable: re-running a command with the same `--out-dir` / `--study-name` continues it.

**Step 0. Settings (lxplus shell)**

```bash
cd /eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/KubeFlow/MuonMLDisplacedVtx
IN=/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/MuonBucketDump_displacedVtx_mu200
OUT=/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/displacedVtx_classifier_mu200
```

**Step 1. Convert the 625 ROOT dumps into packed graphs** (one converter per sample, 8 in parallel; ~3-6 h, ~440 GB;
existing parts are skipped, so the command can be re-run while the dumps are still being produced: the dumper copies
each finished file to EOS with rsync, which writes a hidden temporary file and renames it when complete, so only
complete dumps match the pattern; files that fail are retried by the next run; the logs are appended)

```bash
ls $IN | sed -E 's/_part[0-9]+\.root$//' | sort -u | \
  xargs -P 8 -I{} sh -c 'python -u DisplacedVertex_converter.py --dir-in '"$IN"' --dir-out '"$OUT"' --pattern "{}_part*.root" >> '"$OUT"'/log_convert_{}.txt 2>&1'
echo "packed parts: $(ls -d $OUT/*.packed | wc -l) of $(ls $IN/*.root | wc -l) ROOT files"   # equal when complete
grep -h "All files processed" $OUT/log_convert_*.txt | tail -19   # last run of each sample: failed_files=0
```

**Step 2. Train / validation split**

```bash
python -u DisplacedVertex_splitter.py --data-glob "$OUT/*.packed" --out $OUT/split_displaced_vertex_seed12345.npz \
    --val-fraction 0.1 --seed 12345 2>&1 | tee $OUT/log_splitter.txt
```

**Step 3. Normalization statistics**

```bash
python -u DisplacedVertex_preproc.py --input-glob "$OUT/*.packed" --stats-json $OUT/normalization_stats_raw.json \
    --max-events-per-file 200 2>&1 | tee $OUT/log_preproc.txt
```

**Step 4. Copy the dataset to the GPU machine and set the variables (GPU-machine shell)**

```bash
GPU_DATA=/shared/wp2p5/data/displacedVtx_classifier_mu200
KF=/path/to/KubeFlow/MuonMLDisplacedVtx                      # checkout with this code version
xrdcp -r --parallel 8 root://eosatlas.cern.ch//eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/displacedVtx_classifier_mu200 $(dirname $GPU_DATA)/
ls -d $GPU_DATA/*.packed | wc -l                              # expect 625
cd $KF
DATA_FLAGS=(--train-script ./train_DisplacedVertex.py --data-glob "$GPU_DATA/*.packed"
            --split-file $GPU_DATA/split_displaced_vertex_seed12345.npz
            --feature-stats-json $GPU_DATA/normalization_stats_raw.json
            --compare-metric val_tpr_at_target_fpr --target-fpr 0.01 --num-workers 4 --pin-memory --wandb-mode disabled)
```

**Step 5. Optuna study A: the current architecture re-tuned on the new graphs** (the reference; trial 0 = trainer
defaults; fast trials on 100k training events, 2 GPUs each, 4 in parallel)

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" \
    --out-dir tuning_dv_mu200_A_legacy --study-name dv_mu200_A_legacy \
    --n-trials 80 --n-startup-trials 30 --fast-epochs 40 --fast-max-train-events 100000 \
    --fast-gpus-per-trial 2 --n-jobs 4 --refit-top-k 0 \
    2>&1 | tee -a log_tune_dv_mu200_A_legacy.txt
```

**Step 6. Optuna study B: the new architecture options searched together with the hyperparameters**
(pre-norm blocks, relation-typed message passing, edge states, typed readout; deeper stacks allowed; gradient
clipping in all trials)

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" \
    --out-dir tuning_dv_mu200_B_arch --study-name dv_mu200_B_arch \
    --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any \
    --layers-max 8 --grad-clip 2.0 \
    --n-trials 160 --n-startup-trials 50 --fast-epochs 40 --fast-max-train-events 100000 \
    --fast-gpus-per-trial 2 --n-jobs 4 --refit-top-k 0 \
    2>&1 | tee -a log_tune_dv_mu200_B_arch.txt
```

To tune a single architecture instead, replace the four `any` by fixed values, e.g. `--fixed-block prenorm
--fixed-relation-mp true --fixed-edge-update seg_tracklet --fixed-readout typed_attn` (and another
`--out-dir` / `--study-name`). `--calo-context one_way` (with `--fixed-relation-mp true`) gives the fastest model.

**Step 7. Refit the best 5 trials of each study on the full training split** (8 GPUs per refit)

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" \
    --out-dir tuning_dv_mu200_A_legacy --study-name dv_mu200_A_legacy \
    --refit-only --refit-top-k 5 --refit-epochs 200 --refit-gpus-per-trial 8 \
    2>&1 | tee -a log_tune_dv_mu200_A_legacy_refit.txt

python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" \
    --out-dir tuning_dv_mu200_B_arch --study-name dv_mu200_B_arch \
    --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any \
    --layers-max 8 --grad-clip 2.0 \
    --refit-only --refit-top-k 5 --refit-epochs 200 --refit-gpus-per-trial 8 \
    2>&1 | tee -a log_tune_dv_mu200_B_arch_refit.txt
```

(The refit uses the same search-space flags as the study.)

**Step 8. Step-3 model with the trigger full-scan tracks: study + refit** (same as step 6 with `--use-tracks`)

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" --use-tracks \
    --out-dir tuning_dv_mu200_C_trk --study-name dv_mu200_C_trk \
    --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any \
    --layers-max 8 --grad-clip 2.0 \
    --n-trials 120 --n-startup-trials 40 --fast-epochs 40 --fast-max-train-events 100000 \
    --fast-gpus-per-trial 2 --n-jobs 4 --refit-top-k 0 \
    2>&1 | tee -a log_tune_dv_mu200_C_trk.txt

python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" --use-tracks \
    --out-dir tuning_dv_mu200_C_trk --study-name dv_mu200_C_trk \
    --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any \
    --layers-max 8 --grad-clip 2.0 \
    --refit-only --refit-top-k 5 --refit-epochs 200 --refit-gpus-per-trial 8 \
    2>&1 | tee -a log_tune_dv_mu200_C_trk_refit.txt
```

**Step 9. Evaluation of the refit checkpoints + ONNX export** (per-sample efficiencies at fixed background rates,
ROC, ONNX vs PyTorch; the ONNX files are the trigger models)

```bash
python -u DisplacedVertex_results_tune.py \
    --ckpt-glob "tuning_dv_mu200_A_legacy/checkpoints/*_refit_*.pt" "tuning_dv_mu200_B_arch/checkpoints/*_refit_*.pt" \
                "tuning_dv_mu200_C_trk/checkpoints/*_refit_*.pt" \
    --data-glob "$GPU_DATA/*.packed" --no-explicit-dv-samples \
    --split-file $GPU_DATA/split_displaced_vertex_seed12345.npz \
    --feature-stats-json $GPU_DATA/normalization_stats_raw.json \
    --target-fprs 0.001 0.005 0.01 0.02 --output-dir onnx_eval_dv_mu200_v2 \
    2>&1 | tee log_dv_results_mu200_v2.txt
```

Notes:
- Compare study B against study A on the per-sample signal efficiency at 1% background of the refits (step 9),
  not on single fast trials (their epoch-to-epoch spread is ~0.01-0.02).
- The epoch logs of trials with `--grad-clip` print `grad_norm max/mean` and the clipped fraction: if most steps are
  clipped, raise the value.
- A single training without Optuna (trainer defaults, one GPU node):
  `torchrun --standalone --nproc_per_node=8 train_DisplacedVertex.py --data-glob "$GPU_DATA/*.packed" --split-file $GPU_DATA/split_displaced_vertex_seed12345.npz --feature-stats-json $GPU_DATA/normalization_stats_raw.json --epochs 60 --save-dir models --save dv_default.pt`
  (append any of `--block prenorm --grad-clip 2.0`, `--relation-mp`, `--calo-context one_way`,
  `--edge-update seg_tracklet`, `--readout typed_attn`, `--use-tracks`).
- `--sampler-mix-files 1` (trainer, or `--extra-train-args "--sampler-mix-files 1"` in the tuner) reproduces the
  former single-file batches.
- Event-viewer notebooks need H5 files: `python DisplacedVertex_converter.py --dir-in $IN --dir-out <other dir>
  --pattern "<file>.root" --output-format h5`.
- New dumps later: re-run step 1 (only the new files are converted), then steps 2-3 (split and stats depend on the
  full file list), and start new studies.

### Graphs

One graph per event, label `y = 1` (signal) or `0` (background). Events need >= 1 muon segment; in signal files only
the events with a truth vertex at 800 mm < rho < 8000 mm are kept (the others are skipped, not labelled 0).
Angles are in rad, distances in m unless stated. dR = sqrt(d_eta^2 + d_phi^2).

#### Nodes

`x [N, 7]` holds the muon-segment nodes first (`n_muon_nodes`), then the calorimeter towers; tracks and tracklets
are separate blocks appended after them (track t and tracklet k both have the global index N + t / N + k in their
edge lists; the model places the tracklets after the tracks).

**Muon segments** (MS segments of the ML bucket selection, as used online), `x[:n_muon_nodes]`:

| column | feature | definition |
|---|---|---|
| 0 | r_pos | distance of the segment position from the origin [m] |
| 1 | theta_pos | polar angle of the segment position |
| 2 | phi_pos | azimuth of the segment position |
| 3 | theta_dir | polar angle of the segment direction |
| 4 | phi_dir | azimuth of the segment direction |
| 5 | chi2_ndof | segment chi2 / max(nDoF, 1) (`muon_node_feature_names`; `energy_like` = 0 with `--no-segment-chi2-feature`) |
| 6 | nCells_or_DoF | segment nDoF |

**Calorimeter towers** (`HLT_HICaloTowerFS`), `x[n_muon_nodes:]`: towers with E > 1 GeV and dR < 0.4 to at least
one segment of the event; their position is the intersection of the tower (eta, phi) line with the calorimeter
envelope (r = 4.25 m, |z| = 6.5 m).

| column | feature | definition |
|---|---|---|
| 0-2 | r_pos, theta_pos, phi_pos | envelope intersection point |
| 3-4 | theta_dir, phi_dir | tower direction |
| 5 | energy_like | tower energy [MeV] |
| 6 | nCells_or_DoF | number of cells |

**Tracks** (`--use-tracks`; trigger full-scan FTF tracks, `TrigFSTrackDump`), `track_x [T, 11]`: tracks with
pT > 2 GeV and dR(perigee direction, segment) < 0.3 for at least one segment.

| feature | definition |
|---|---|
| theta_dir, phi_dir | perigee direction |
| log_pt | ln(pT / GeV) |
| q_over_pt | charge / pT [1/GeV] |
| d0, d0_sig | transverse impact parameter [mm], d0 / sigma(d0) |
| dz0_sintheta, dz0_sig | (z0 + vz - z_PV) sin(theta) [mm], (z0 + vz - z_PV) / sigma(z0), PV = hard-scatter vertex |
| n_pixel_hits, n_strip_hits | hits |
| chi2_ndof | track chi2 / max(nDoF, 1) |

d0, d0_sig, dz0_sintheta, dz0_sig and chi2_ndof enter the model as asinh(x / scale) (transform inside the model).

**MS tracklets** (`tracklet_x [K, 11]`): chains of segments of one candidate muon. Two segments are linked when they
are in different, known MS station layers (Inner / Middle / Outer / Extended / BarrelExtended), dR < 0.4 and the
chord angle < 0.2 rad (largest r-z angle between the straight line joining their positions and either segment
direction). A tracklet is a connected component of these links spanning >= 2 station layers. Inner / outer member:
smallest / largest distance from the origin.

| feature | definition |
|---|---|
| eta, phi | mean eta, circular mean phi of the member positions |
| theta_chord | polar angle of the inner -> outer member chord |
| n_segments, n_layers | members, distinct station layers |
| d_inner_m, d_outer_m | distance of the inner / outer member from the origin [m] |
| max_chord_angle | largest chord angle of the links of the tracklet |
| chord_rz_ip_m | r-z distance between the origin and the inner -> outer chord line [m] (pointing) |
| sum_ndof, mean_chi2_ndof | sum of the member nDoF, mean member chi2 / nDoF |

#### Edges

All edge blocks are directed and contain both directions. Features are dst - src.

**Segment <-> tower** (`edge_index [2, E]`, `edge_attr [E, 5]`): every segment-tower pair with dR < 0.4.
**Segment <-> track** (`track_edge_index`, `track_edge_attr [Et, 5]`): every segment-track pair with dR < 0.3. Same
5 features (energy_like of a track = pT [MeV], of a segment = 0):

| feature | definition |
|---|---|
| d_energy_like | energy_like(dst) - energy_like(src) |
| d_phi | phi(dst) - phi(src), wrapped |
| d_eta | eta(dst) - eta(src) |
| cos_angle | cosine between the two directions |
| same_sector | 1 if both are in the same MS sector (segment: bucket sector; tower / track: MS sector of phi, as in DVInferenceToolBase) |

**Segment <-> segment** (`seg_edge_index [2, Es]`, `seg_edge_attr [Es, 7]`): every segment pair with dR < 0.4.

| feature | definition |
|---|---|
| d_eta, d_phi | eta / phi of the segment positions, dst - src |
| cos_angle | cosine between the segment directions |
| chord_angle | largest r-z angle between the chord joining the two positions and either segment direction [rad] (0 for chords < 1 mm) |
| chord_length_m | r-z distance between the two positions [m] |
| same_layer | same MS station layer |
| same_sector | same MS sector |

**Tracklet edges** (`tracklet_edge_index [2, Ek]`, `tracklet_edge_attr [Ek, 4]`): each tracklet <-> each of its member
segments, and every ordered pair of tracklets (fully connected).

| feature | definition |
|---|---|
| d_eta, d_phi | dst - src (tracklet: mean eta / phi) |
| cos_angle | segment direction . tracklet chord (membership edges); chord . chord (tracklet pairs) |
| is_tracklet_pair | 0 segment <-> tracklet, 1 tracklet <-> tracklet |

#### Packed part layout

`<stem>.packed/` = uncompressed `.npy` files, memory-mapped by the training (`PackedEventDataset`):
`<block>.npy` with the block concatenated over the events (edge indices `[2, E]`, int32) and `<block>_ptr.npy`
(event e = rows `ptr[e]:ptr[e+1]`) for the blocks `x, edge_index, edge_attr, track_*, seg_edge_*, tracklet_*`;
per event `y`, `n_muon_nodes`, `event_key`, `dataset_name`, `root_file`; `meta.json` with the converter settings
(`converter_config`), `muon_node_feature_names` and the file attributes. Event keys and order are those of the H5
output of the same conversion, so split files are interchangeable between the two formats.
