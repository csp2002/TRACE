#!/bin/bash
# Launch 3 ablation settings × 5 datasets = 15 trainings across 8 GPUs
# via a single tmux session with 8 chained windows.
#
# Chain layout (each chain runs sequentially inside one tmux window, on one GPU):
#   GPU 0 :  A_colon    → B_colon    → C_colon
#   GPU 1 :  A_lits     → B_lits     → C_lits
#   GPU 2 :  A_pancreas → B_pancreas
#   GPU 3 :  A_kits     → C_pancreas
#   GPU 4 :  A_local    → B_local
#   GPU 5 :  B_kits     → C_local
#   GPU 6 :  C_kits
#   GPU 7 :  (empty, reserved for test runs once trainings finish)
#
# Logs go to /tmp/tumorseg_ablation_<setting><ds>.log
#
# Usage:  bash scripts/launch_ablations_ABC.sh
#         tmux attach -t tumorseg_ablations   # to watch
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

SESSION=tumorseg_ablations
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" -n gpu0

# Create windows explicitly by index with -n giving the name; tmux requires
# -t session (not session:name) when the target window doesn't exist yet.
for name in gpu1 gpu2 gpu3 gpu4 gpu5 gpu6 gpu7; do
  tmux new-window -t "${SESSION}" -n "${name}"
done

# Helper to build a single dataset training command with logging.
# $1=letter(A|B|C), $2=dataset, $3=gpu
run_cmd() {
  local L="$1" ds="$2" gpu="$3"
  echo "bash single_node_train_tumorseg_refine_${L}.sh ${ds} ${gpu} 1 2>&1 | tee /tmp/tumorseg_ablation_${L}_${ds}.log"
}

# Chain runs use '&&' so a failed training halts the chain (we want to notice).
chain0="$(run_cmd A colon 0) && $(run_cmd B colon 0) && $(run_cmd C colon 0)"
chain1="$(run_cmd A lits  1) && $(run_cmd B lits  1) && $(run_cmd C lits  1)"
chain2="$(run_cmd A pancreas 2) && $(run_cmd B pancreas 2)"
chain3="$(run_cmd A kits 3) && $(run_cmd C pancreas 3)"
chain4="$(run_cmd A local 4) && $(run_cmd B local 4)"
chain5="$(run_cmd B kits 5) && $(run_cmd C local 5)"
chain6="$(run_cmd C kits 6)"

tmux send-keys -t "${SESSION}:gpu0" "${chain0}" Enter
tmux send-keys -t "${SESSION}:gpu1" "${chain1}" Enter
tmux send-keys -t "${SESSION}:gpu2" "${chain2}" Enter
tmux send-keys -t "${SESSION}:gpu3" "${chain3}" Enter
tmux send-keys -t "${SESSION}:gpu4" "${chain4}" Enter
tmux send-keys -t "${SESSION}:gpu5" "${chain5}" Enter
tmux send-keys -t "${SESSION}:gpu6" "${chain6}" Enter

echo "[OK] tmux session '${SESSION}' launched with 7 active GPU chains."
echo "     tmux attach -t ${SESSION}"
echo "     Logs: /tmp/tumorseg_ablation_{A,B,C}_{dataset}.log"
