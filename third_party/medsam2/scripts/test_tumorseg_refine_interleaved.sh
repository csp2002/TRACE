#!/bin/bash
# Test SAM2TrainWithRefineInterleaved checkpoints with the matched
# interleaved-inference path. Output → results/results_<dataset>_video_refineE_interleaved.json
#
# Usage: bash scripts/test_tumorseg_refine_interleaved.sh <dataset> [gpu_id] [ckpt_path]
set -euo pipefail
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medsam2
cd "$(dirname "$(readlink -f "$0")")/.."

DATASET="${1:-colon}"
GPU="${2:-0}"
CKPT="${3:-exp_log/tumorseg_refineE_${DATASET}/checkpoints/checkpoint.pt}"

[ -f "${CKPT}" ] || { echo "[ERR] checkpoint not found: ${CKPT}" >&2; exit 1; }
echo "[INTERLEAVED] dataset=${DATASET} gpu=${GPU} ckpt=${CKPT}"

CUDA_VISIBLE_DEVICES="${GPU}" python test_medsam2_2d_video_refine_interleaved.py \
  --data "${DATASET}" \
  --checkpoint "${CKPT}" \
  --device cuda:0
