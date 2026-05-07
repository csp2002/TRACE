#!/bin/bash
# Edit workload comparison script for MedSAM2
# Compares edit workload between Mode A and Mode B

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medsam2
cd "$(dirname "$0")/.."

# Create logs directory
LOG_DIR="./simulation_logs"
mkdir -p ${LOG_DIR}

MODEL_NAME="MedSAM2"
timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${LOG_DIR}/edit_workload_${MODEL_NAME}_${timestamp}.log"

echo "=========================================="
echo "Running Edit Workload Comparison for ${MODEL_NAME}"
echo "Log file: ${log_file}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=6 python -m tumor_seg.edit_workload \
    --model_name ${MODEL_NAME} \
    --exclude-patients "" \
    --medsam2_cfg sam2.1_hiera_t512.yaml \
    2>&1 | tee ${log_file}

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Error: Edit workload comparison for ${MODEL_NAME} failed!"
    echo "Check log file: ${log_file}"
    exit 1
fi

echo "=========================================="
echo "Edit workload comparison for ${MODEL_NAME} completed!"
echo "=========================================="
