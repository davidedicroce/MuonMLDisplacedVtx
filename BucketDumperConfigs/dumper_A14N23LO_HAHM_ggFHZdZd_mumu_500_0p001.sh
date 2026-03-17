#!/bin/bash

NUM_EVENTS=-1  # Set to -1 to process all events, or specify a positive integer for testing

INPUT_DIR="${PWD}/datasets/group.det-muon.545613.MGPy8EG_A14N23LO_HAHM_ggFHZdZd_mumu_500_0p001.RDO_MU200.R4-250226.v1_EXT0"

# Setup ATLAS environment
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup Athena,main,latest,here

if [ -d "athena" ]; then
  echo "Building Athena..."
  mkdir -p build && cd build && rm -rf * && cmake ../athena/Projects/WorkDir/ && make -j 8 && cd -
  source build/x86_64-el9-gcc14-opt/setup.sh
else
  echo "Running default Athena setup since x86_64-el9-gcc14-opt/setup.sh not found"
fi

# Settings for Phase II RUN4  
set -euo pipefail

# ----------------------------
# GPU Detection
# ----------------------------
USE_GPU=0

if command -v nvidia-smi &> /dev/null; then
  echo "Checking for available GPUs..."
  if nvidia-smi -L &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
      echo "Found $GPU_COUNT GPU(s) available"
      nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
      USE_GPU=1
      export CUDA_VISIBLE_DEVICES=0
      echo "GPU inference ENABLED (MuonBucketDump will use ONNX CUDA backend)"
    else
      echo "No GPUs detected. Using CPU for inference."
    fi
  else
    echo "nvidia-smi found but GPUs not accessible. Using CPU for inference."
  fi
else
  echo "nvidia-smi not found. Using CPU for inference."
fi
echo

# ----------------------------
# User configuration
# ----------------------------
# Where to run + store outputs/logs
WORKDIR="${TestArea:-$PWD}/MuonBucketDump_HAHM_ggFHZdZd_mumu_500_0p001"
OUTDIR="${WORKDIR}/outputs"
LOGDIR="${WORKDIR}/logs"

# Optional: threads option for muonBucketDump (only enable if your module supports it)
THREADS=8

# Storage controls
CLEANUP_ON_SUCCESS=1

# Dump feature toggles.
# Enable both muon truth and calorimeter dumps for complete information.
DO_CALO_DUMP=1
DO_TRUTH_MUON_VERTEX_DUMP=1
DO_ML_BUCKET_SCORE=1
DO_ML_BUCKET_FILTER=0

# Dynamic configuration from ATLAS Python modules (Phase II RUN4)
export ATLAS_GEO_TAG=$(python -c "from AthenaConfiguration.TestDefaults import defaultGeometryTags; print(defaultGeometryTags.RUN4)")

# ----------------------------
# Setup
# ----------------------------
killall -9 gdb >/dev/null 2>&1 || true

mkdir -p "${WORKDIR}" "${OUTDIR}" "${LOGDIR}"
cd "${WORKDIR}"

echo "WORKDIR: ${WORKDIR}"
echo "INPUT_DIR: ${INPUT_DIR}"
echo "OUTDIR: ${OUTDIR}"
echo "LOGDIR: ${LOGDIR}"
echo "GPU Status: $([ "${USE_GPU}" -eq 1 ] && echo "ENABLED" || echo "DISABLED (CPU only)")"
echo
echo "Strategy: Run muonBucketDump directly on RDO files (no digitization)"
echo

echo "ATLAS_GEO_TAG: ${ATLAS_GEO_TAG}"
echo

export | tee "${WORKDIR}/environ.log" >/dev/null

# ----------------------------
# Discover input files
# ----------------------------
if [ ! -d "${INPUT_DIR}" ]; then
  echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}"
  exit 1
fi

mapfile -t INPUT_FILES < <(find "${INPUT_DIR}" -type f \( -name "*.root" -o -name "*.pool.root" \) | sort)

if [ "${#INPUT_FILES[@]}" -eq 0 ]; then
  echo "ERROR: No .root files found under: ${INPUT_DIR}"
  exit 1
fi

echo "Found ${#INPUT_FILES[@]} input ROOT files."
printf "%s\n" "${INPUT_FILES[@]}" | head -n 20
if [ "${#INPUT_FILES[@]}" -gt 20 ]; then
  echo "... (showing first 20)"
fi
echo

# ----------------------------
# Run MuonBucketDump
# ----------------------------
i=1
for inFile in "${INPUT_FILES[@]}"; do
  base="$(basename "${inFile}")"
  baseNoExt="${base%.pool.root}"
  baseNoExt="${baseNoExt%.root}"

  logFile="${LOGDIR}/MuonBucketDump_${i}_${baseNoExt}.log"
  outFile="${OUTDIR}/MuonBucketDump_${baseNoExt}.root"

  echo "[$i/${#INPUT_FILES[@]}] Processing: ${inFile}"
  echo "  output: ${outFile}"
  echo "  log: ${logFile}"

  bucketFlags=(
    --threads "${THREADS}"
    --nEvents "${NUM_EVENTS}"
    --outRootFile "${outFile}"
    --defaultGeoFile RUN4
    --inputFile "${inFile}"
  )

  [ "${DO_CALO_DUMP}" -eq 1 ] && bucketFlags+=(--doCaloDump)
  [ "${DO_TRUTH_MUON_VERTEX_DUMP}" -eq 1 ] && bucketFlags+=(--doTruthMuonVertexDump)
  [ "${DO_ML_BUCKET_SCORE}" -eq 1 ] && bucketFlags+=(--doMLBucketScore)
  [ "${DO_ML_BUCKET_FILTER}" -eq 1 ] && bucketFlags+=(--doMLBucketFilter)

  if [ "${USE_GPU}" -eq 0 ]; then
    bucketFlags+=(--use-cpu)
  fi

  python -m MuonBucketDump.muonBucketDump \
    "${bucketFlags[@]}" \
    >> "${logFile}" 2>&1

  rc=$?
  if [ $rc -ne 0 ]; then
    echo "ERROR: MuonBucketDump failed"
    echo "Last 50 lines of log:"
    tail -n 50 "${logFile}"
    exit $rc
  fi

  if [ "${CLEANUP_ON_SUCCESS}" -eq 1 ]; then
    rm -f \
      "${WORKDIR}/PoolFileCatalog.xml" \
      "${WORKDIR}"/prmon.* 2>/dev/null || true
  fi

  i=$((i + 1))
done

echo
echo "All done."
echo "Logs: ${LOGDIR}"
echo "Outputs: ${OUTDIR}"