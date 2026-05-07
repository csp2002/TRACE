#!/bin/bash
# Evaluate a fine-tuned MedSAM2 checkpoint in video mode on a dataset's test split.
# Output is written to results/results_<dataset>_video_tumorseg.json so the
# baseline file results_<dataset>_video.json (MedSAM2_latest baseline) stays intact.
#
# Usage:  bash scripts/test_tumorseg.sh <dataset> [gpu_id] [ckpt_path]
# Defaults:
#   gpu_id     = 1
#   ckpt_path  = exp_log/tumorseg_<dataset>/checkpoints/checkpoint.pt
set -euo pipefail
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medsam2
cd "$(dirname "$(readlink -f "$0")")/.."

DATASET="${1:-colon}"
GPU="${2:-1}"
CKPT="${3:-exp_log/tumorseg_${DATASET}/checkpoints/checkpoint.pt}"

if [ ! -f "${CKPT}" ]; then
  echo "[ERR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

echo "[INFO] dataset=${DATASET} gpu=${GPU} ckpt=${CKPT}"

# test_medsam2_2d_video.py hardcodes output to results/results_<ds>_video.json.
# Stash any existing baseline file first so we don't clobber it.
SRC="results/results_${DATASET}_video.json"
DST="results/results_${DATASET}_video_tumorseg.json"
STASH="/tmp/results_${DATASET}_video_BASELINE_STASH_$$.json"
[ -f "${SRC}" ] && cp "${SRC}" "${STASH}" && echo "[INFO] stashed baseline → ${STASH}"

CUDA_VISIBLE_DEVICES="${GPU}" python test_medsam2_2d_video.py \
  --data "${DATASET}" \
  --checkpoint "${CKPT}" \
  --device cuda:0

# Move fresh output to the tumorseg-suffixed name, then restore the baseline.
if [ -f "${SRC}" ]; then
  mv "${SRC}" "${DST}"
  echo "[OK] results → ${DST}"
fi
[ -f "${STASH}" ] && mv "${STASH}" "${SRC}" && echo "[OK] baseline restored → ${SRC}"
