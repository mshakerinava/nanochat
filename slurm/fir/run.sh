#!/bin/bash
#SBATCH --job-name=ssm_d20
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:4
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --signal=TERM@120
#SBATCH --account=rrg-bengioy-ad

set -u

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
REPORT="$NANOCHAT_BASE_DIR/report/header.md"
SCRIPT_PATH="$HOME/scratch/nanochat/slurm/fir/run.sh"
CHECKPOINTS_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/ssm_d20"

echo "[$(date)] Job $SLURM_JOB_ID starting"
echo "Checkpoint directory: $CHECKPOINTS_DIR"

#############################################################
# 1. Stop if training is already finished
#############################################################
if [ -f "$REPORT" ] && grep -q "## Base model training" "$REPORT"; then
    echo "[$(date)] Training complete — NOT resubmitting."
    exit 0
fi

#############################################################
# 2. Auto-resubmit on timeout
#############################################################
on_term() {
    echo "[$(date)] Caught SIGTERM near time limit."

    if [ -f "$REPORT" ] && grep -q "## Base model training" "$REPORT"; then
        echo "[$(date)] Training finished — NOT resubmitting."
        exit 0
    fi

    echo "[$(date)] Resubmitting via sbatch $SCRIPT_PATH"
    sbatch "$SCRIPT_PATH"
    exit 0
}
trap on_term TERM

#############################################################
# 3. Environment
#############################################################
source .venv/bin/activate

#############################################################
# 4. Detect first run → reset report
#############################################################
if ! ls "$CHECKPOINTS_DIR"/meta_*.json >/dev/null 2>&1; then
    echo "[$(date)] No checkpoints found — FIRST RUN"
    echo "[$(date)] Resetting report with: python -m nanochat.report reset"
    python -m nanochat.report reset
    RESUME_ARG=""
else
    echo "[$(date)] Checkpoints found — RESUMING TRAINING"
    RESUME_ARG="--resume-from-step 0"
fi

#############################################################
# 5. Start training
#############################################################
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=20 \
    --run=dummy \
    --model_type=ssm \
    --save_every=100 \
    --device_batch_size=16 \
    $RESUME_ARG
