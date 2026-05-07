#!/bin/bash
# Master launcher for edit_workload_comparison_v2 experiments.
# Runs all 9 methods × 5 datasets using tmux for parallelism.
#
# GPU survey at launch time:
#   GPU 1 / 2 / 3 / 5  → ~48 GB free, idle (selected)
#   GPU 0 / 4 / 6 / 7  → busy (skipped)
#
# Per-job memory is ~3 GB for traditional 224×224 models and ~5–8 GB for
# MedSAM (1024×1024) / MedSAM2 (512×512). Co-locating 2–3 jobs per GPU keeps
# the footprint well under 48 GB while overlapping compute.

set -u

REPO_ROOT="$(dirname "$0")/.."
TRANSUNET_DIR="${REPO_ROOT}/project_TransUNet/TransUNet"
LOG_DIR="${REPO_ROOT}/Simulation/logs"
LAUNCH_DIR="${REPO_ROOT}/scripts/.ew_launch"
mkdir -p "${LOG_DIR}" "${LAUNCH_DIR}"

TS=$(date +%Y%m%d_%H%M%S)

# Columns: session|model|gpu|conda_env|extra_args
JOBS=(
    "ew_transunet|TransUNet|1|3DSAM|"
    "ew_attnunet|AttentionUNet|1|3DSAM|"
    "ew_medformer|MedFormer|2|3DSAM|"
    "ew_unetpp|UNetPlusPlus|2|3DSAM|"
    "ew_swinunet|SwinUnet|3|3DSAM|"
    "ew_fatnet|FAT_Net|3|3DSAM|"
    "ew_h2former|H2Former|3|3DSAM|"
    "ew_medsam|MedSAM|5|3DSAM|"
    "ew_medsam2|MedSAM2|5|medsam2|--medsam2_cfg sam2.1_hiera_t512.yaml"
)

echo "=========================================="
echo "Launching ${#JOBS[@]} edit_workload tmux sessions @ ${TS}"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s MiB free, %s%% util\n", $1, $2, $3}'
echo ""

for job in "${JOBS[@]}"; do
    IFS='|' read -r session model gpu env extra <<< "$job"
    logfile="${LOG_DIR}/edit_workload_${model}_${TS}.log"
    runner="${LAUNCH_DIR}/${session}_${TS}.sh"

    cat > "${runner}" <<EOF
#!/bin/bash
# Auto-generated runner for ${session}
set -e
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${env}
cd ${TRANSUNET_DIR}
CUDA_VISIBLE_DEVICES=${gpu} python -m tumor_seg.edit_workload \\
    --model_name ${model} \\
    --exclude-patients "" \\
    ${extra} 2>&1 | tee ${logfile}
status=\${PIPESTATUS[0]}
echo ""
echo "[${session}] python exit status: \${status}"
exit \${status}
EOF
    chmod +x "${runner}"

    # Kill any prior session with the same name before relaunch
    tmux kill-session -t "${session}" 2>/dev/null || true

    tmux new-session -d -s "${session}" "bash ${runner}"
    printf "  launched  %-14s gpu=%s  env=%-7s  model=%-14s  log=%s\n" \
        "${session}" "${gpu}" "${env}" "${model}" "${logfile}"
done

echo ""
echo "All ${#JOBS[@]} sessions launched."
echo ""
echo "Monitor:"
echo "  tmux ls | grep '^ew_'"
echo "  tmux attach -t ew_transunet      # (Ctrl-b d to detach)"
echo "  tail -f ${LOG_DIR}/edit_workload_<Model>_${TS}.log"
echo ""
echo "Check completion:"
echo "  grep -H 'exit status' ${LOG_DIR}/edit_workload_*_${TS}.log"
