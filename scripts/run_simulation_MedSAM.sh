#!/bin/bash
# Simulation script for MedSAM
# Runs option1 and option3 sequentially for all datasets

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 3DSAM
cd "$(dirname "$0")/.."

# Create logs directory
LOG_DIR="./simulation_logs"
mkdir -p ${LOG_DIR}

MODEL_NAME="MedSAM"
datasets=("lits" "colon")
options=(3)

for option in "${options[@]}"; do
    for dataset in "${datasets[@]}"; do
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="${LOG_DIR}/${MODEL_NAME}_option${option}_${dataset}_${timestamp}.log"
        
        echo "=========================================="
        echo "Running ${MODEL_NAME} - Option ${option} - Dataset: ${dataset}"
        echo "Log file: ${log_file}"
        echo "=========================================="
        
        CUDA_VISIBLE_DEVICES=7 python -m tumor_seg.simulation \
            --model_name ${MODEL_NAME} \
            --dataset ${dataset} \
            --option ${option} \
            --thresholds 0.70 0.75 0.80 0.85 0.90 0.95 \
            --exclude-patients "" \
            2>&1 | tee ${log_file}
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "Error: ${MODEL_NAME} - Option ${option} - Dataset ${dataset} failed!"
            echo "Check log file: ${log_file}"
            exit 1
        fi
    done
done

echo "=========================================="
echo "All ${MODEL_NAME} simulations completed!"
echo "=========================================="
