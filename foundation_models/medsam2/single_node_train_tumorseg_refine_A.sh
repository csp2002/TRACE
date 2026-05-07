#!/bin/bash
# Ablation A: video-ours + MiddleAnchoredSampler (conditioning frame = middle).
# Usage: bash single_node_train_tumorseg_refine_A.sh <dataset> [gpu_id] [num_gpus]
set -euo pipefail
export PATH=/usr/local/cuda/bin:${PATH:-}
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medsam2
cd "$(dirname "$(readlink -f "$0")")"

DATASET="${1:-colon}"
GPUS="${2:-0}"
NUM_GPUS="${3:-1}"

CONFIG="configs/sam2.1_hiera_tiny512_tumorseg_refine_A_midanchor.yaml"
DATA_ROOT="${DATA_ROOT:-./data/tumorseg_npz_v2}"   # min_slices=2, pad-friendly
DATASET_PATH="${DATA_ROOT}/${DATASET}/train"
OUTPUT_PATH="./exp_log/tumorseg_refineA_${DATASET}"

[ -d "${DATASET_PATH}" ] || { echo "[ERR] ${DATASET_PATH} not found" >&2; exit 1; }
echo "[A/midanchor] ${DATASET} GPU=${GPUS} data=${DATASET_PATH}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m training.train \
    -c "${CONFIG}" \
    --dataset-path "${DATASET_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --use-cluster 0 --num-gpus "${NUM_GPUS}" --num-nodes 1

echo "[A/midanchor] ${DATASET} training done"
