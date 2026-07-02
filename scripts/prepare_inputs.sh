#!/usr/bin/env bash
set -euo pipefail

# Prepare the split file and raw-feature normalization statistics required by
# tune.sh / train_DisplacedVertex.py.

REPO_DIR="${REPO_DIR:-$PWD}"
DATA_DIR="${DATA_DIR:-/eos/user/y/yshresth/mudb}"
DATA_GLOB="${DATA_GLOB:-${DATA_DIR}/*.h5}"
SPLIT_FILE="${SPLIT_FILE:-${DATA_DIR}/split_displaced_vertex_seed12345.npz}"
FEATURE_STATS_JSON="${FEATURE_STATS_JSON:-${DATA_DIR}/normalization_stats_raw.json}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-12345}"
MAX_STATS_EVENTS="${MAX_STATS_EVENTS:--1}"

if [ ! -f "${REPO_DIR}/DisplacedVertex_splitter.py" ] && [ -f "${PWD}/DisplacedVertex_splitter.py" ]; then
  REPO_DIR="${PWD}"
fi

cd "${REPO_DIR}"

echo "[prepare_inputs] repo=${REPO_DIR}"
echo "[prepare_inputs] data_glob=${DATA_GLOB}"
echo "[prepare_inputs] split_file=${SPLIT_FILE}"
echo "[prepare_inputs] feature_stats_json=${FEATURE_STATS_JSON}"

python -u "${REPO_DIR}/DisplacedVertex_splitter.py" \
  --data-glob "${DATA_GLOB}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}" \
  --out "${SPLIT_FILE}"

python -u "${REPO_DIR}/DisplacedVertex_preproc.py" \
  --input-glob "${DATA_GLOB}" \
  --stats-json "${FEATURE_STATS_JSON}" \
  --max-events "${MAX_STATS_EVENTS}"

echo "[prepare_inputs] done"
