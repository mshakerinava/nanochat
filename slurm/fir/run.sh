#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:4
#SBATCH --time=3:00:00
#SBATCH --mem=100G
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --requeue
#SBATCH --account=rrg-bengioy-ad

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate

# Location to store resume state flag
STATE_FILE="resume.flag"

# On SIGUSR1: requeue the job
_handler() {
    echo "Caught SIGUSR1: requeueing..."
    touch "$STATE_FILE"
    scontrol requeue $SLURM_JOB_ID
}
trap _handler SIGUSR1

# Determine resume mode
RESUME_ARG=""
if [ -f "$STATE_FILE" ]; then
    echo "Resuming training..."
    RESUME_ARG="--resume-from-step latest"
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Base model (pretraining)

torchrun --standalone --nproc_per_node=4 \
    -m scripts.base_train -- \
    --depth=20 \
    --run=dummy \
    --model_type=ssm \
    --save_every=100 \
    --device_batch_size=16 \
    $RESUME_ARG
