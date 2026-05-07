#!/bin/bash
# Train MedSAM2 in video mode on this project's 5 tumor datasets.
# Usage:
#   bash single_node_train_tumorseg.sh <dataset> [gpu_id] [num_gpus]
# Examples:
#   bash single_node_train_tumorseg.sh colon 1 1
#   bash single_node_train_tumorseg.sh kits 0,1 2
#
# Assumes data has already been converted via:
#   python scripts/convert_png_to_npz.py --dataset <ds> --split train
#   python scripts/convert_png_to_npz.py --dataset <ds> --split test
set -euo pipefail
export PATH=/usr/local/cuda/bin:${PATH:-}
# Always run from the MedSAM2 repo root so `training.*` imports resolve.
cd "$(dirname "$(readlink -f "$0")")"

DATASET="${1:-colon}"
GPUS="${2:-1}"
NUM_GPUS="${3:-1}"

CONFIG="configs/sam2.1_hiera_tiny512_tumorseg.yaml"
DATA_ROOT="${DATA_ROOT:-./data/tumorseg_npz}"
DATASET_PATH="${DATA_ROOT}/${DATASET}/train"
OUTPUT_PATH="./exp_log/tumorseg_${DATASET}"

if [ ! -d "${DATASET_PATH}" ]; then
  echo "[ERR] ${DATASET_PATH} not found. Run scripts/convert_png_to_npz.py first." >&2
  exit 1
fi

echo "[INFO] dataset=${DATASET} GPUs=${GPUS} (${NUM_GPUS} proc)  data=${DATASET_PATH}"
echo "[INFO] config=${CONFIG}  output=${OUTPUT_PATH}"

# Use `python -m training.train` (not `python training/train.py`) so CWD is on
# sys.path and `from training.*` imports resolve correctly.
CUDA_VISIBLE_DEVICES="${GPUS}" python -m training.train \
    -c "${CONFIG}" \
    --dataset-path "${DATASET_PATH}" \
    --output-path "${OUTPUT_PATH}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    --num-nodes 1

echo "training done"
