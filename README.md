# Check Status
python3 - tuning_dv_mu200_relmp_1M/optuna.journal <<'EOF'
import json, sys, collections
S = {0: "RUNNING", 1: "COMPLETE", 2: "PRUNED", 3: "FAIL", 4: "WAITING"}
state, value, n = {}, {}, 0
for line in open(sys.argv[1]):
    d = json.loads(line)
    if d["op_code"] == 4:                      # CREATE_TRIAL
        state[n] = d.get("state", 0); n += 1
    elif d["op_code"] == 6:                    # SET_TRIAL_STATE_VALUES
        state[d["trial_id"]] = d["state"]
        if d.get("values"): value[d["trial_id"]] = d["values"][0]
by = collections.defaultdict(list)
for t, s in state.items(): by[S[s]].append(t)
for k in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING"):
    print(f"{k:9s} {len(by[k]):3d}  {by[k]}")
top = sorted(value.items(), key=lambda x: -x[1])[:5]
print("best:", ", ".join(f"#{t}={v:.5f}" for t, v in top))
EOF

# To Launch 
EOS_DATA=/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/displacedVtx_classifier_mu200
GPU_DATA=/tmp/$USER/wp2p5/data/displacedVtx_classifier_mu200
LIST=/tmp/$USER/wp2p5/dv_copy_list.txt
du -sh --exclude='*.bak_*' $EOS_DATA
df -h /tmp


cd /eos/project-i01/f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/KubeFlow/MuonMLDisplacedVtx
DATA_FLAGS=(--train-script ./train_DisplacedVertex.py --data-glob "$GPU_DATA/*.packed"
            --split-file $GPU_DATA/split_displaced_vertex_seed12345.npz
            --feature-stats-json $GPU_DATA/normalization_stats_raw.json
            --compare-metric val_tpr_at_target_fpr --target-fpr 0.001 --num-workers 4 --pin-memory --wandb-mode disabled)
SPACE=(--fixed-layer-type gat_residual --gatv2-edge-attn
       --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any --layers-max 8)
FAST=(--fast-epochs 40 --fast-max-train-events 100000 --fast-gpus-per-trial 1 --n-jobs 8
      --batch-size auto --objective-window 3)
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" "${SPACE[@]}" "${FAST[@]}" \
    --out-dir tuning_dv_mu200 --study-name dv_mu200 \
    --n-trials 160 --n-startup-trials 50 --prune --refit-top-k 0 \
    2>&1 | tee -a log_tune_dv_mu200.txt


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

Steps 1-3 run where the dumps are (EOS, e.g. lxplus); steps 4-8 on the GPU machine (8 GPUs assumed). Every command is
complete: only the variables of step 0 and `GPU_DATA` / `KF` of step 4 may need adapting. Optuna studies and confirm
phases are resumable: re-running a command with the same `--out-dir` / `--study-name` continues it (the trials that
were running are lost and replaced); a re-run refit command starts its refits again.

**Step 0. Settings (lxplus shell)**

```bash
cd /eos/project-f/fcc-ml/ddicroce/ATLAS_MuonSpectrometer/KubeFlow/MuonMLDisplacedVtx
IN=/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/MuonBucketDump_displacedVtx_mu200
OUT=/eos/atlas/atlascerngroupdisk/det-muon/muonSW/data_ml/displacedVtx_classifier_mu200
```

**Step 1. Convert the ROOT dumps into packed graphs** (one converter per sample, 8 in parallel; ~3-6 h, ~440 GB;
existing parts are skipped, so the command can be re-run while the dumps are still being produced: the dumper copies
each finished file to EOS with rsync, which writes a hidden temporary file and renames it when complete, so only
complete dumps match the pattern; files that fail are retried by the next run; the logs are appended)

Run it inside `screen` so it survives a disconnection (start: `screen -S dv_convert`, then the step-0 settings, then
the command; detach: Ctrl-a d; back: `screen -r dv_convert`). EOS access needs a valid Kerberos ticket for the whole
run (`klist`; `kinit` before starting if it expires within ~6 h). Progress from any shell:
`ls -d $OUT/*.packed | wc -l`.

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
            --compare-metric val_tpr_at_target_fpr --target-fpr 0.001 --num-workers 4 --pin-memory --wandb-mode disabled)
# search space: GATv2 + the four architecture options (24 combinations), up to 8 layers
SPACE=(--fixed-layer-type gat_residual --gatv2-edge-attn
       --fixed-block any --fixed-relation-mp any --fixed-edge-update any --fixed-readout any --layers-max 8)
# protocol of the fast trials: stored in the study, every later command of the study repeats it
FAST=(--fast-epochs 40 --fast-max-train-events 100000 --fast-gpus-per-trial 1 --n-jobs 8
      --batch-size auto --objective-window 3)
```

The protocol is the same in every phase of a study. It is stored in the study at its first run, and a later command
with other values stops and lists the differences.
- Score: validation signal efficiency (TPR) at 0.1% background (`--target-fpr 0.001`), averaged over the best 3
  consecutive epochs (`--objective-window 3`). The validation split holds 44,600 background events, so the 0.1%
  working point rests on ~45 of them and the score of a single epoch is noisy. Its best single epoch is biased
  upwards by that noise.
- Batch, per architecture (`--batch-size auto`, as in the segment tuner): every searched fast trial lets the
  trainer probe 16, 32, 64 and 128 graphs on its GPU for its own model. It takes the fastest batch whose worst case
  (the B largest of 512 random events) stays under 70% of the GPU memory, the smallest on near-ties.
- Learning rate: the searched lr is defined at a global batch of 64 and scaled by sqrt(batch/64) in each run; the
  EMA decay is rescaled to keep its horizon in events.
- Confirm runs and refits keep the trial's global batch, split over their GPUs. For a trial at 64: 1 GPU x 64 in
  the confirm runs, 8 GPUs x 8 in the refit. The refit then trains at the batch and lr the trial was ranked with.
  The old study refitted with 4x the batch of its fast trials at the same lr, and its fast ranking did not predict
  the refits (rank correlation ~0).
- Trial 0 runs at 64, the trainer default. The leaderboard shows each trial's batch.
- A trial probed at 16 refits with 2 graphs per GPU on 8 GPUs, which uses them poorly; `--refit-gpus-per-trial 2`
  gives it 8 per GPU.
- For one fixed batch in all trials instead, use `--global-batch-size 64` in place of `--batch-size auto`. There is
  then no probe, so the largest models may run out of GPU memory.

**Step 5. One Optuna study over the architectures and the hyperparameters** (fast trials on 100k training events,
1 GPU each, 8 in parallel, losing trials stopped early)

All trials use the GATv2 model (`SPACE`: the other layer types are not searched). `any` searches only the four new
architecture options, so the space contains the current GATv2 architecture (`--block legacy`, no relation-typed
message passing, no edge states, global readout) and its 23 combinations with pre-norm blocks, relation-typed
message passing, edge states and the typed readout. Trial 0 is the current best model: the trainer defaults
(architecture and hyperparameters). It lies inside the search space, so Optuna learns from it, which needs the
study without `--grad-clip` (the trainer default has no clipping; with `--grad-clip` trial 0 would still run, but
outside the space).

`--prune`: from epoch 10, a trial whose running score is below the median of the complete trials at the same
epoch is stopped (state PRUNED). Pruning starts after 8 complete trials, and trial 0 is never pruned. On the old study
this would have saved 36% of the epochs without stopping any of the final top 10.

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" "${SPACE[@]}" "${FAST[@]}" \
    --out-dir tuning_dv_mu200 --study-name dv_mu200 \
    --n-trials 160 --n-startup-trials 50 --prune --refit-top-k 0 \
    2>&1 | tee -a log_tune_dv_mu200.txt
less tuning_dv_mu200/leaderboard.md                  # rewritten after every trial (all trials: leaderboard.csv)
```

`leaderboard.md` holds:
- the best configuration;
- the rank of trial 0;
- per architecture: best and median score, number of complete and pruned trials, model size;
- the top 30 trials: score, best single epoch, AUC, epochs, size, main hyperparameters;
- the parameter importances.

`--report-only` (with the same flags) rewrites it without running anything.

To tune a single architecture instead, replace the four `any` by fixed values, e.g. `--fixed-block prenorm
--fixed-relation-mp true --fixed-edge-update seg_tracklet --fixed-readout typed_attn` (with another `--out-dir` /
`--study-name`). `--calo-context one_way` (with `--fixed-relation-mp true`) gives the fastest model.

**Step 6. Confirm the best 8 trials on 4x the training events, then refit the best 5 on the full training split**

One fast trial is one draw of the training noise, on 7% of the training split. The best fast trials often differ
by less than that noise, and a short training favours configurations that learn fast. The confirm phase re-runs
the 8 best fast trials with 2 seeds each on 400k training events (`--confirm-max-train-events 400000`, 1 GPU each,
8 in parallel). It ranks them by the mean of the 2 runs; the fast score does not count, since its protocol differs.
The ranking goes to `<out-dir>/confirm_summary.json` and the leaderboard, with the error of the mean and the
#1-vs-#2 significance. The refit (8 GPUs each) takes the 5 best of this ranking. Trial 0 (the current best model)
is confirmed and refitted when it is among them. Finished confirm runs are kept in `<out-dir>/confirm.jsonl`, so a
restarted command only runs what is missing.

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" "${SPACE[@]}" "${FAST[@]}" \
    --out-dir tuning_dv_mu200 --study-name dv_mu200 \
    --refit-only --confirm-top-k 8 --confirm-seeds 2 --confirm-max-train-events 400000 \
    --refit-top-k 5 --refit-epochs 200 --refit-gpus-per-trial 8 \
    2>&1 | tee -a log_tune_dv_mu200_refit.txt
less tuning_dv_mu200/leaderboard.md                  # "Best configuration": by refit score once the refits exist
```

The refit uses the same `SPACE` as the study (without it, it falls back to the current architecture) and the same
`FAST` (without it, the command stops at the protocol check).

**Step 7. Step-3 model with the trigger full-scan tracks: study + confirm + refit** (step 5 and 6 with
`--use-tracks`; trial 0 = the current best hyperparameters with the tracks)

```bash
python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" "${SPACE[@]}" "${FAST[@]}" --use-tracks \
    --out-dir tuning_dv_mu200_trk --study-name dv_mu200_trk \
    --n-trials 120 --n-startup-trials 40 --prune --refit-top-k 0 \
    2>&1 | tee -a log_tune_dv_mu200_trk.txt

python -u tune_DisplacedVertex_optuna.py "${DATA_FLAGS[@]}" "${SPACE[@]}" "${FAST[@]}" --use-tracks \
    --out-dir tuning_dv_mu200_trk --study-name dv_mu200_trk \
    --refit-only --confirm-top-k 8 --confirm-seeds 2 --confirm-max-train-events 400000 \
    --refit-top-k 5 --refit-epochs 200 --refit-gpus-per-trial 8 \
    2>&1 | tee -a log_tune_dv_mu200_trk_refit.txt
```

**Step 8. Evaluation of the refits + ONNX export** (per-sample efficiencies at fixed background rates, ROC, ONNX vs
PyTorch; the ONNX files are the trigger models)

```bash
python -u DisplacedVertex_results_tune.py \
    --ckpt-glob "tuning_dv_mu200/checkpoints/*_refit_*.pt" "tuning_dv_mu200_trk/checkpoints/*_refit_*.pt" \
    --data-glob "$GPU_DATA/*.packed" --no-explicit-dv-samples \
    --split-file $GPU_DATA/split_displaced_vertex_seed12345.npz \
    --feature-stats-json $GPU_DATA/normalization_stats_raw.json \
    --target-fprs 0.001 0.005 0.01 0.02 --output-dir onnx_eval_dv_mu200_v2 \
    2>&1 | tee log_dv_results_mu200_v2.txt
```

Notes:
- Compare the refits on the per-sample signal efficiency at 0.1% background (step 8), not on single fast trials
  (at 1% background their epoch-to-epoch spread was ~0.01-0.02; at 0.1% it is larger). The refits are single runs:
  in the old study, refits of the same configuration spread by ~0.03. Trial 0 (the current best model) is the
  fast-trial reference of the study; it is refitted only if it ranks in the top 5.
- Studies from before 2026-11-24 (TPR at 1%, single best epoch, per-GPU batch) continue with their own protocol:
  `--target-fpr 0.01 --objective-window 1` and no `--global-batch-size`. Their scores are not comparable with these
  studies.
- Gradient clipping (`--grad-clip`, a fixed value for all trials of a study) is off in these studies to keep trial 0
  in the search space. For a single-architecture study with pre-norm blocks it can be added; the epoch logs then
  print `grad_norm max/mean` and the clipped fraction (if most steps are clipped, raise the value).
- A single training without Optuna (trainer defaults, one GPU node; the checkpoint keeps the epoch with the best TPR
  at 0.1% background):
  `torchrun --standalone --nproc_per_node=8 train_DisplacedVertex.py --data-glob "$GPU_DATA/*.packed" --split-file $GPU_DATA/split_displaced_vertex_seed12345.npz --feature-stats-json $GPU_DATA/normalization_stats_raw.json --target-fpr 0.001 --epochs 60 --save-dir models --save dv_default.pt`
  (append any of `--block prenorm --grad-clip 2.0`, `--relation-mp`, `--calo-context one_way`,
  `--edge-update seg_tracklet`, `--readout typed_attn`, `--use-tracks`).
- `--sampler-mix-files 1` (trainer, or `--extra-train-args "--sampler-mix-files 1"` in the tuner) reproduces the
  former single-file batches. The leaderboard header shows the data format, the sampler and the dataset-indexing time
  of the trials (packed parts: seconds; H5 parts took a median 41 min per trial in the old study).
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
