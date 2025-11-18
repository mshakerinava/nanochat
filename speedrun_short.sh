#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 4) Example launch with SSM model:
# MODEL_TYPE=ssm bash speedrun_short.sh
# 5) Example launch with SSM model and wandb:
# MODEL_TYPE=ssm WANDB_RUN=speedrun bash speedrun_short.sh

# -----------------------------------------------------------------------------
# symlink the cache directory to the scratch directory
if [ -L "$HOME/.cache" ]; then
    echo "~/.cache is already a symlink"
else
    # Create scratch cache dir parent if needed
    echo "Copying ~/.cache to ~/scratch/.cache"
    cp -rf $HOME/.cache $HOME/scratch/.cache
    echo "Removing original ~/.cache ..."
    rm -rf $HOME/.cache
    echo "Creating symlink ~/.cache -> ~/scratch/.cache"
    ln -s $HOME/scratch/.cache $HOME/.cache
    echo "Symlink created."
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Model type setup
# Set MODEL_TYPE=ssm to use SSM model instead of GPT, e.g.:
# MODEL_TYPE=ssm bash speedrun_short.sh
if [ -z "$MODEL_TYPE" ]; then
    # by default use GPT model
    MODEL_TYPE="gpt"
fi
echo "Using model type: $MODEL_TYPE"

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Number of processes/GPUs to use
NPROC_PER_NODE=1

# Model depth
MODEL_DEPTH=2

# Model tag
MODEL_TAG="${MODEL_TYPE}_d${MODEL_DEPTH}"

# pretrain the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=$MODEL_DEPTH --run=$WANDB_RUN --model_type=$MODEL_TYPE --num_iterations=1
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss -- --model-tag=$MODEL_TAG
# evaluate the model on CORE tasks
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --model-tag=$MODEL_TAG --max-per-task=10

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN --num_iterations=1 --model-tag=$MODEL_TAG
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid --model-tag=$MODEL_TAG -x 1

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN --num_iterations=2 --model-tag=$MODEL_TAG
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft --model-tag=$MODEL_TAG -x 1

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli --model-tag=$MODEL_TAG -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web --model-tag=$MODEL_TAG

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN --model-tag=$MODEL_TAG
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K --model-tag=$MODEL_TAG

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate