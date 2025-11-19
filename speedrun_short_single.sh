#!/bin/bash
set -e

# symlink the cache directory to the scratch directory
if [ -L "$HOME/.cache" ]; then
    echo "~/.cache is already a symlink"
else
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
cp -rf tokenizer $NANOCHAT_BASE_DIR/

source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Model type setup
if [ -z "$MODEL_TYPE" ]; then
    MODEL_TYPE="gpt"
fi
echo "Using model type: $MODEL_TYPE"

# -----------------------------------------------------------------------------
# Write report header
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Base model (pretraining)
MODEL_DEPTH=2
MODEL_TAG="${MODEL_TYPE}_d${MODEL_DEPTH}"

python -m scripts.base_train --depth=$MODEL_DEPTH --run=$WANDB_RUN --model_type=$MODEL_TYPE --num_iterations=10 --save-every=5
python -m scripts.base_loss --model-tag=$MODEL_TAG
# python -m scripts.base_eval --model-tag=$MODEL_TAG --max-per-task=10

# -----------------------------------------------------------------------------
# Midtraining
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.mid_train --run=$WANDB_RUN --num_iterations=1 --model-tag=$MODEL_TAG
python -m scripts.chat_eval -i mid --model-tag=$MODEL_TAG -x 1

# -----------------------------------------------------------------------------
# Supervised Finetuning
python -m scripts.chat_sft --run=$WANDB_RUN --num_iterations=2 --model-tag=$MODEL_TAG
python -m scripts.chat_eval -i sft --model-tag=$MODEL_TAG -x 1

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report generate
