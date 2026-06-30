#!/usr/bin/env bash
set -euo pipefail

# Condor launcher for GAT-residual Optuna tuning.
# Override any of these from the submit file with environment variables if needed.

REPO_DIR="${REPO_DIR:-$PWD}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train_DisplacedVertex.py}"

DATA_GLOB="${DATA_GLOB:-/shared/wp2p5/data/data_displacedVtx_mu200_graphs/*.h5}"
SPLIT_FILE="${SPLIT_FILE:-/shared/wp2p5/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz}"
FEATURE_STATS_JSON="${FEATURE_STATS_JSON:-/shared/wp2p5/data/data_displacedVtx_mu200_graphs/normalization_stats_raw.json}"

OUT_DIR="${OUT_DIR:-/shared/wp2p5/models/tuning_dv_classifier_gat}"
STUDY_NAME="${STUDY_NAME:-dv_classifier_gat}"

N_TRIALS="${N_TRIALS:-30}"
FAST_EPOCHS="${FAST_EPOCHS:-10}"
FAST_GPUS_PER_TRIAL="${FAST_GPUS_PER_TRIAL:-2}"
N_JOBS="${N_JOBS:-2}"
FAST_MAX_TRAIN_EVENTS="${FAST_MAX_TRAIN_EVENTS:-10000}"
REFIT_TOP_K="${REFIT_TOP_K:-2}"
REFIT_EPOCHS="${REFIT_EPOCHS:-30}"
REFIT_GPUS_PER_TRIAL="${REFIT_GPUS_PER_TRIAL:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "${REPO_DIR}"

echo "[tune.sh] host=$(hostname)"
echo "[tune.sh] cwd=$(pwd)"
echo "[tune.sh] started_at=$(date -Is)"
echo "[tune.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python -u "${REPO_DIR}/tune_DisplacedVertex_optuna.py" \
  --train-script "${TRAIN_SCRIPT}" \
  --data-glob "${DATA_GLOB}" \
  --split-file "${SPLIT_FILE}" \
  --feature-stats-json "${FEATURE_STATS_JSON}" \
  --normalize-node-features \
  --normalize-edge-features \
  --out-dir "${OUT_DIR}" \
  --study-name "${STUDY_NAME}" \
  --storage-backend journal \
  --fixed-layer-type gat_residual \
  --compare-metric val_tpr_at_target_fpr \
  --target-fpr 0.01 \
  --loss-types bce bce_smooth focal asymmetric_focal \
  --n-trials "${N_TRIALS}" \
  --fast-epochs "${FAST_EPOCHS}" \
  --fast-gpus-per-trial "${FAST_GPUS_PER_TRIAL}" \
  --n-jobs "${N_JOBS}" \
  --fast-max-train-events "${FAST_MAX_TRAIN_EVENTS}" \
  --refit-max-train-events -1 \
  --refit-top-k "${REFIT_TOP_K}" \
  --refit-epochs "${REFIT_EPOCHS}" \
  --refit-gpus-per-trial "${REFIT_GPUS_PER_TRIAL}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory \
  --wandb-mode disabled

echo "[tune.sh] finished_at=$(date -Is)"
