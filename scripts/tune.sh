#!/usr/bin/env bash
set -euo pipefail

# Condor launcher for GAT-residual Optuna tuning.
# Override any of these from the submit file with environment variables if needed.

REPO_DIR="${REPO_DIR:-$PWD}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train_DisplacedVertex.py}"

DATA_DIR="${DATA_DIR:-/eos/user/y/yshresth/mudb}"
DATA_GLOB="${DATA_GLOB:-${DATA_DIR}/*.h5}"
SPLIT_FILE="${SPLIT_FILE:-${DATA_DIR}/split_displaced_vertex_seed12345.npz}"
FEATURE_STATS_JSON="${FEATURE_STATS_JSON:-${DATA_DIR}/normalization_stats_raw.json}"

OUT_DIR="${OUT_DIR:-/eos/user/y/yshresth/mounresult/tuning_dv_classifier_gat}"
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
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-300}"

CONDOR_LOG_DIR="${CONDOR_LOG_DIR:-${REPO_DIR}/scripts/condor_logs}"

cd "${REPO_DIR}"
mkdir -p "${CONDOR_LOG_DIR}"

echo "[tune.sh] host=$(hostname)"
echo "[tune.sh] cwd=$(pwd)"
echo "[tune.sh] started_at=$(date -Is)"
echo "[tune.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

HEARTBEAT_PID=""
(
  while true; do
    sleep "${HEARTBEAT_INTERVAL}"
    completed_trials=0
    if [ -f "${OUT_DIR}/trials.jsonl" ]; then
      completed_trials=$(wc -l < "${OUT_DIR}/trials.jsonl" || echo 0)
    fi
    latest_log="$(find "${OUT_DIR}/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)"
    echo "[heartbeat] $(date -Is) completed_trial_records=${completed_trials} latest_trial_log=${latest_log:-none}"
  done
) &
HEARTBEAT_PID="$!"
trap 'if [ -n "${HEARTBEAT_PID}" ]; then kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true; fi' EXIT

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
