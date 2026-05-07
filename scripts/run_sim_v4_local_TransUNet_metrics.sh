#!/bin/bash
# Parallel simulation_v4.py runs on local dataset, TransUNet, option 3.
# One run per metric (dice, sdsc, fn_rate, fp_rate, hd95) with metric-appropriate
# thresholds (picked from the empirical per-slice distribution on local/middle).
# Thresholds cover the full "accept vs reject" spectrum for both baseline and ours.
#
# GPU memory per job ~3 GB (user-reported). Pack 5 jobs across GPU 1 and GPU 5.

set -euo pipefail

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

run() {
    local gpu="$1"
    local metric="$2"
    shift 2
    local thresholds="$*"
    local log="${LOG_DIR}/simv4_${MODEL}_${DATASET}_option${OPTION}_${metric}_${TS}.log"
    echo "[launch] metric=${metric} gpu=${gpu} thresholds=${thresholds} log=${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m tumor_seg.simulation_other_metrics \
        --model_name "${MODEL}" \
        --dataset "${DATASET}" \
        --option "${OPTION}" \
        --metric "${metric}" \
        --thresholds ${thresholds} \
        --exclude-patients "${EXCLUDE}" \
        > "${log}" 2>&1 &
    echo "  pid=$!"
}

# Pack 3 jobs on GPU 1 and 2 on GPU 5 (both near-empty at launch time).
run 1 dice    0.50 0.60 0.70 0.80 0.85 0.90 0.95
run 1 sdsc    0.40 0.55 0.70 0.85 0.95 0.99
run 1 hd95    2 3 5 10 20 50
run 5 fn_rate 0.05 0.10 0.20 0.30 0.50 0.80
run 5 fp_rate 0.05 0.10 0.15 0.25 0.50

wait
echo "[done] all metric simulations finished"
