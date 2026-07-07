#!/usr/bin/env bash
set -euo pipefail

# Create a small mixed signal/background H5 dataset by symlinking randomly
# selected files from a larger dataset directory.
#
# Defaults are safe to override:
#   SOURCE_DATA_DIR=/eos/user/y/yshresth/mudb
#   SPLIT_FILE=/eos/user/y/yshresth/mudb/split_displaced_vertex_seed12345.npz
#   SMALL_DATA_DIR=/eos/user/y/yshresth/mudb_small_mixed
#   N_SIGNAL_FILES=10
#   N_BACKGROUND_FILES=10
#   SEED=12345
#   PREPARE_SPLIT_STATS=1

REPO_DIR="${REPO_DIR:-$PWD}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/eos/user/y/yshresth/mudb}"
SPLIT_FILE="${SPLIT_FILE:-${SOURCE_DATA_DIR}/split_displaced_vertex_seed12345.npz}"
SMALL_DATA_DIR="${SMALL_DATA_DIR:-/eos/user/y/yshresth/mudb_small_mixed}"
N_SIGNAL_FILES="${N_SIGNAL_FILES:-10}"
N_BACKGROUND_FILES="${N_BACKGROUND_FILES:-10}"
SEED="${SEED:-12345}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
PREPARE_SPLIT_STATS="${PREPARE_SPLIT_STATS:-1}"
MAX_STATS_EVENTS="${MAX_STATS_EVENTS:--1}"

mkdir -p "${SMALL_DATA_DIR}"
find "${SMALL_DATA_DIR}" -maxdepth 1 -type l -name '*.h5' -delete

TMP_LIST="$(mktemp)"
trap 'rm -f "${TMP_LIST}"' EXIT

python - "${SOURCE_DATA_DIR}" "${SPLIT_FILE}" "${N_SIGNAL_FILES}" "${N_BACKGROUND_FILES}" "${SEED}" > "${TMP_LIST}" <<'PY'
import random
import sys
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
split_file = Path(sys.argv[2])
n_sig = int(sys.argv[3])
n_bkg = int(sys.argv[4])
seed = int(sys.argv[5])

rng = random.Random(seed)
signal = []
background = []
mixed = []

if not split_file.exists():
    print(f"ERROR: split file does not exist: {split_file}", file=sys.stderr)
    sys.exit(2)

split = np.load(split_file, allow_pickle=True)
required = {"h5_paths", "event_refs", "labels"}
missing = sorted(required.difference(split.files))
if missing:
    print(f"ERROR: split file is missing required keys: {missing}", file=sys.stderr)
    sys.exit(2)

saved_paths = [Path(str(p)) for p in split["h5_paths"].tolist()]
labels = np.asarray(split["labels"], dtype=np.float32).reshape(-1)
event_refs = np.asarray(split["event_refs"], dtype=object)

counts = {i: [0, 0] for i in range(len(saved_paths))}
for ref, y in zip(event_refs, labels):
    fi = int(ref[0])
    if y == 1.0:
        counts[fi][1] += 1
    elif y == 0.0:
        counts[fi][0] += 1

for fi, (n_neg, n_pos) in counts.items():
    saved = saved_paths[fi]
    path = source / saved.name
    if not path.exists():
        path = saved
    if not path.exists():
        continue
    if n_pos > 0 and n_neg == 0:
        signal.append(path)
    elif n_neg > 0 and n_pos == 0:
        background.append(path)
    elif n_pos > 0 and n_neg > 0:
        mixed.append(path)

rng.shuffle(signal)
rng.shuffle(background)
rng.shuffle(mixed)

chosen_sig = signal[:n_sig]
chosen_bkg = background[:n_bkg]

if len(chosen_sig) < n_sig:
    need = n_sig - len(chosen_sig)
    chosen_sig.extend(mixed[:need])
if len(chosen_bkg) < n_bkg:
    need = n_bkg - len(chosen_bkg)
    chosen_bkg.extend(mixed[len(chosen_sig):len(chosen_sig) + need])

if len(chosen_sig) < n_sig or len(chosen_bkg) < n_bkg:
    print(
        f"ERROR: not enough files. signal={len(signal)}, background={len(background)}, mixed={len(mixed)}",
        file=sys.stderr,
    )
    sys.exit(2)

print(f"# found signal_files={len(signal)} background_files={len(background)} mixed_files={len(mixed)}", file=sys.stderr)
print(f"# selected signal_files={len(chosen_sig)} background_files={len(chosen_bkg)}", file=sys.stderr)

for p in chosen_sig + chosen_bkg:
    print(p)
PY

echo "[create_mixed_small_dataset] source=${SOURCE_DATA_DIR}"
echo "[create_mixed_small_dataset] target=${SMALL_DATA_DIR}"
echo "[create_mixed_small_dataset] linking files..."

while IFS= read -r src; do
  [ -n "${src}" ] || continue
  ln -s "${src}" "${SMALL_DATA_DIR}/$(basename "${src}")"
done < "${TMP_LIST}"

echo "[create_mixed_small_dataset] linked $(find "${SMALL_DATA_DIR}" -maxdepth 1 -type l -name '*.h5' | wc -l) files"

if [ "${PREPARE_SPLIT_STATS}" = "1" ]; then
  cd "${REPO_DIR}"
  DATA_GLOB="${SMALL_DATA_DIR}/*.h5"
  SPLIT_FILE="${SMALL_DATA_DIR}/split_displaced_vertex_seed${SEED}.npz"
  FEATURE_STATS_JSON="${SMALL_DATA_DIR}/normalization_stats_raw.json"

  python -u "${REPO_DIR}/DisplacedVertex_splitter.py" \
    --data-glob "${DATA_GLOB}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --out "${SPLIT_FILE}"

  python -u "${REPO_DIR}/DisplacedVertex_preproc.py" \
    --input-glob "${DATA_GLOB}" \
    --stats-json "${FEATURE_STATS_JSON}" \
    --max-events "${MAX_STATS_EVENTS}"

  echo "[create_mixed_small_dataset] split=${SPLIT_FILE}"
  echo "[create_mixed_small_dataset] stats=${FEATURE_STATS_JSON}"
fi
