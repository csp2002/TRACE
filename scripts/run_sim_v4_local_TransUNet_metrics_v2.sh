#!/bin/bash
# Re-run simulation_v4.py with denser threshold sweeps for 4 metrics.
# Target: local dataset, TransUNet, option 3.
#
# Updated thresholds (sDSC unchanged):
#   dice    : 0.50 to 0.90 step 0.05  ->  9 thresholds
#   fn_rate : 0.05 to 0.50 step 0.05  -> 10 thresholds
#   fp_rate : 0.05 to 0.30 step 0.05  ->  6 thresholds
#   hd95    : 2    to 20   step 2     -> 10 thresholds
#
# GPU assignment (GPU 0/4/7 idle at launch; GPU 0 has light external load):
#   GPU 0  : dice          (9 thresholds)
#   GPU 4  : fn_rate       (10 thresholds)
#   GPU 7  : fp_rate -> hd95 (16 thresholds, sequential; balances longest stream)
#
# Each (threshold, mode) ≈ 5–6 min on TransUNet/local; expected wall time ≈ 3 h.
# After all streams finish, regenerate the per-metric comparison plot.

set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 3DSAM
cd "$(dirname "$0")/.."

LOG_DIR="./simulation_logs"
mkdir -p "${LOG_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
MODEL="TransUNet"
DATASET="local"
OPTION=3
EXCLUDE=""
SAVE="${SAVE:-./simulation_output}"

DICE_THS=$(seq 0.50 0.05 0.90 | tr '\n' ' ')
FN_THS=$(seq 0.05 0.05 0.50 | tr '\n' ' ')
FP_THS=$(seq 0.05 0.05 0.30 | tr '\n' ' ')
HD95_THS=$(seq 2 2 20 | tr '\n' ' ')

MASTER_LOG="${LOG_DIR}/master_simv4_rerun_${TS}.log"
echo "===== simulation_v4 rerun launched @ ${TS} =====" | tee "${MASTER_LOG}"

run_metric() {
    local gpu="$1" metric="$2" tag="$3"
    shift 3
    local thresholds="$*"
    local log="${LOG_DIR}/simv4_rerun_${tag}_${TS}.log"
    echo "[start ] ${tag}: gpu=${gpu} thresholds=${thresholds}" | tee -a "${MASTER_LOG}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m tumor_seg.simulation_other_metrics \
        --model_name "${MODEL}" \
        --dataset "${DATASET}" \
        --option "${OPTION}" \
        --metric "${metric}" \
        --save_path "${SAVE}" \
        --thresholds ${thresholds} \
        --exclude-patients "${EXCLUDE}" \
        > "${log}" 2>&1
    local status=$?
    echo "[finish] ${tag}: status=${status} log=${log}" | tee -a "${MASTER_LOG}"
    return ${status}
}

# Stream 1: GPU 0 — dice
( run_metric 0 dice    "dice_GPU0"    ${DICE_THS}; echo "[stream1 done]" >> "${MASTER_LOG}" ) &
PID_S1=$!

# Stream 2: GPU 4 — fn_rate
( run_metric 4 fn_rate "fn_rate_GPU4" ${FN_THS};   echo "[stream2 done]" >> "${MASTER_LOG}" ) &
PID_S2=$!

# Stream 3: GPU 7 — fp_rate -> hd95 (sequential)
(
    run_metric 7 fp_rate "fp_rate_GPU7" ${FP_THS}
    run_metric 7 hd95    "hd95_GPU7"    ${HD95_THS}
    echo "[stream3 done]" >> "${MASTER_LOG}"
) &
PID_S3=$!

echo "Launched: stream1 pid=${PID_S1}, stream2 pid=${PID_S2}, stream3 pid=${PID_S3}" | tee -a "${MASTER_LOG}"
wait ${PID_S1} ${PID_S2} ${PID_S3}
echo "All streams finished — regenerating plots" | tee -a "${MASTER_LOG}"

# plot script not bundled in release
# python <plot_script_path> 2>&1 | tee -a "${MASTER_LOG}"

echo "===== rerun complete @ $(date +%Y%m%d_%H%M%S) =====" | tee -a "${MASTER_LOG}"
