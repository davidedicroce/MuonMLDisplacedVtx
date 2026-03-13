#!/bin/bash

NUM_EVENTS=-1  # Set to -1 to process all events, or specify a positive integer for testing
EVENTS_PER_FILE=5000  # Split output into multiple files with this many events each

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
# use pwd for input dir
INPUT_DIR="${PWD}/datasets/group.det-muon.514992.MGPy8EG_A14N23LO_HNL2_ctau1000_mumumu.HITS.R4-250226.v1_EXT0"

# Where to run + store outputs/logs
WORKDIR="${TestArea:-$PWD}/MuonBucketDump_HNL2_ctau1000"
OUTDIR="${WORKDIR}/outputs"
LOGDIR="${WORKDIR}/logs"

# Optional: threads option for muonBucketDump (only enable if your module supports it)
THREADS=8

ENABLE_DIGIT_PARAMS_DB_FOR_CALO=1

# Digitization uses Run 4 / Phase II
RUN_PERIOD="RUN4"

# Storage controls
CLEANUP_ON_SUCCESS=1

# If your muonBucketDump accepts an output ROOT argument, set it here.
WRITE_OUTROOT=1

# Dump feature toggles.
# Enable both muon truth and calorimeter dumps for complete information.
DO_CALO_DUMP=1
DO_TRUTH_MUON_VERTEX_DUMP=1
DO_ML_BUCKET_SCORE=1
DO_ML_BUCKET_FILTER=0

# Dynamic configuration from ATLAS Python modules (Phase II RUN4)
export GEOMODEL_DB_FILE=$(python -c "from MuonGeoModelTestR4.testGeoModel import MuonPhaseIITestDefaults; print(MuonPhaseIITestDefaults.GEODB_R4);")
export ATLAS_CONDDB_TAG=$(python -c "from AthenaConfiguration.TestDefaults import defaultConditionsTags; print(defaultConditionsTags.RUN4_MC)")
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
echo "Strategy: Digitize calorimeter only → RDO with calo + truth"
echo "          muonBucketDump reads calo from RDO, muon truth from truth containers"
echo
echo "NOTE: Expect warnings about missing muon RDO containers (MDT_SDO, RPC_SDO, etc.)"
echo "      These are normal - HITS has no muon simulation, only truth muons"
echo "      doCaloDump will work correctly with calorimeter data"
echo

echo "ATLAS_CONDDB_TAG: ${ATLAS_CONDDB_TAG}"
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
  digiLogFile="${LOGDIR}/Digitization_${i}_${baseNoExt}.log"

  echo "[$i/${#INPUT_FILES[@]}] Processing: ${inFile}"
  echo "  digi log: ${digiLogFile}"
  echo "  out log: ${logFile}"

  # Process in chunks
  chunkIdx=0
  skipEvents=0
  
  while true; do
    chunkIdx=$((chunkIdx + 1))
    rdoFile=""
    
    if [ "${ENABLE_DIGIT_PARAMS_DB_FOR_CALO}" -eq 1 ] && [[ "${inFile}" == *".HITS.pool.root" || "${inFile}" == *".HITS.root" ]]; then
      rdoFile="${WORKDIR}/RDO_${baseNoExt}_chunk${chunkIdx}.pool.root"
      rm -f "${rdoFile}"

      echo "  chunk $chunkIdx: digitizing (skipEvents=${skipEvents}) → RDO" | tee -a "${digiLogFile}"

      # Digitize in chunks
      Digi_tf.py \
        --CA \
        --multithreaded True \
        --digiSeedOffset1 170 \
        --digiSeedOffset2 170 \
        --geometrySQLite True \
        --geometrySQLiteFullPath ${GEOMODEL_DB_FILE} \
        --geometryVersion "default:${ATLAS_GEO_TAG}" \
        --conditionsTag "default:${ATLAS_CONDDB_TAG}" \
        --preExec "all:from Campaigns.PhaseII import PhaseIIPileUpMC21a;from Campaigns.MC23 import MC23d;from AthenaConfiguration.Enums import LHCPeriod;(MC23d(flags) if flags.GeoModel.Run == LHCPeriod.Run3 else PhaseIIPileUpMC21a(flags));flags.Detector.EnablePLR=False;flags.Detector.EnableBCMPrime=False" \
        --postExec "default:flags.dump(evaluate=True)" \
        --postInclude 'all:PyJobTransforms.UseFrontier' \
        --outputRDOFile "${rdoFile}" \
        --skipEvents ${skipEvents} \
        --maxEvents ${EVENTS_PER_FILE} \
        --inputHitsFile "${inFile}" \
        --imf False \
        >> "${digiLogFile}" 2>&1

      rc=$?
      if [ $rc -ne 0 ]; then
        # Check if we've run out of events (expected end condition)
        if tail -n 50 "${digiLogFile}" | grep -q "0 events"; then
          echo "  Finished digitizing all chunks" | tee -a "${digiLogFile}"
          rm -f "${rdoFile}"
          break
        else
          echo "ERROR: Failed digitizing chunk $chunkIdx for: ${inFile}"
          echo "Last 80 lines of digitization log:"
          tail -n 80 "${digiLogFile}"
          exit $rc
        fi
      fi
      
      runInputFile="${rdoFile}"
    else
      runInputFile="${inFile}"
    fi

    if [ "${WRITE_OUTROOT}" -eq 1 ]; then
      outFile="${OUTDIR}/MuonBucketDump_${baseNoExt}_chunk${chunkIdx}.root"
      echo "  chunk $chunkIdx: processing → ${outFile}" | tee -a "${logFile}"

      bucketFlags=(
        --threads "${THREADS}"
        --nEvents "${EVENTS_PER_FILE}"
        --outRootFile "${outFile}"
        --defaultGeoFile RUN4
      )
      
      bucketFlags+=(--inputFile "${runInputFile}")

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
        # Check if we've run out of events (expected end condition)
        if tail -n 100 "${logFile}" | grep -q "0 events passing"; then
          echo "  Finished processing all chunks" | tee -a "${logFile}"
          break
        else
          echo "ERROR: MuonBucketDump chunk $chunkIdx failed"
          echo "Last 50 lines of log:"
          tail -n 50 "${logFile}"
          exit $rc
        fi
      fi
    else
      bucketFlags=(
        --threads "${THREADS}"
        --nEvents "${EVENTS_PER_FILE}"
        --defaultGeoFile RUN4
      )
      
      bucketFlags+=(--inputFile "${runInputFile}")

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
        # Check if we've run out of events (expected end condition)
        if tail -n 100 "${logFile}" | grep -q "0 events passing"; then
          echo "  Finished processing all chunks" | tee -a "${logFile}"
          break
        else
          echo "ERROR: MuonBucketDump chunk $chunkIdx failed"
          echo "Last 50 lines of log:"
          tail -n 50 "${logFile}"
          exit $rc
        fi
      fi
    fi
    
    # Cleanup chunk RDO file after processing
    [ -n "${rdoFile}" ] && [ -f "${rdoFile}" ] && rm -f "${rdoFile}"
    
    skipEvents=$((skipEvents + EVENTS_PER_FILE))
  done

  if [ "${CLEANUP_ON_SUCCESS}" -eq 1 ]; then
    rm -f \
      "${WORKDIR}/log.HITtoRDO" \
      "${WORKDIR}/ValidNtuple.digi.root" \
      "${WORKDIR}/PoolFileCatalog.xml" \
      "${WORKDIR}/sqlite200" \
      "${WORKDIR}"/prmon.* 2>/dev/null || true
  fi

  i=$((i + 1))
done

echo
echo "All done."
echo "Logs: ${LOGDIR}"
echo "Outputs: ${OUTDIR}"