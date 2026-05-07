#!/bin/bash
# Evaluate the fine-tuned MedSAM2 video-mode + refinement checkpoint on a
# dataset's test split. Output → results/results_<dataset>_video_refine.json
# (does not touch baseline or tumorseg jsons).
#
# Usage: bash scripts/test_tumorseg_refine.sh <dataset> [gpu_id] [ckpt_path]
set -euo pipefail
# Activate the medsam2 conda env unconditionally (we don't assume the parent
# shell has it active — e.g. nohup children may lose PATH activation).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medsam2
cd "$(dirname "$(readlink -f "$0")")/.."

DATASET="${1:-colon}"
GPU="${2:-0}"
CKPT="${3:-exp_log/tumorseg_refine_${DATASET}/checkpoints/checkpoint.pt}"

if [ ! -f "${CKPT}" ]; then
  echo "[ERR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

echo "[INFO] dataset=${DATASET} gpu=${GPU} ckpt=${CKPT}"
CUDA_VISIBLE_DEVICES="${GPU}" python test_medsam2_2d_video_refine.py \
  --data "${DATASET}" \
  --checkpoint "${CKPT}" \
  --device cuda:0
