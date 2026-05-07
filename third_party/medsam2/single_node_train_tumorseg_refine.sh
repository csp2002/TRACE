#!/bin/bash
# Train MedSAM2 video-mode + plug-in refinement module (TRACE) jointly
# on one of the 5 tumor datasets.
#
# Usage:
#   bash single_node_train_tumorseg_refine.sh <dataset> [gpu_id] [num_gpus]
# Example:
#   bash single_node_train_tumorseg_refine.sh colon 0 1
set -euo pipefail
export PATH=/usr/local/cuda/bin:${PATH:-}
cd "$(dirname "$(readlink -f "$0")")"

DATASET="${1:-colon}"
GPUS="${2:-0}"
NUM_GPUS="${3:-1}"

CONFIG="configs/sam2.1_hiera_tiny512_tumorseg_refine.yaml"
DATA_ROOT="${DATA_ROOT:-./data/tumorseg_npz}"
DATASET_PATH="${DATA_ROOT}/${DATASET}/train"
OUTPUT_PATH="./exp_log/tumorseg_refine_${DATASET}"

if [ ! -d "${DATASET_PATH}" ]; then
  echo "[ERR] ${DATASET_PATH} not found. Run scripts/convert_png_to_npz.py first." >&2
  exit 1
fi

echo "[INFO] dataset=${DATASET} GPUs=${GPUS} (${NUM_GPUS} proc)  data=${DATASET_PATH}"
echo "[INFO] config=${CONFIG}  output=${OUTPUT_PATH}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m training.train \
    -c "${CONFIG}" \
    --dataset-path "${DATASET_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    --num-nodes 1

echo "training done"
