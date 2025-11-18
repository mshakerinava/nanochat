#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Symlink the cache directory to the scratch directory

if [ -L "$HOME/.cache" ]; then
    echo "~/.cache is already a symlink"
else
    echo "Copying ~/.cache to ~/scratch/.cache"
    cp -rf "$HOME/.cache" "$HOME/scratch/.cache"
    echo "Removing original ~/.cache ..."
    rm -rf "$HOME/.cache"
    echo "Creating symlink ~/.cache -> ~/scratch/.cache"
    ln -s "$HOME/scratch/.cache" "$HOME/.cache"
    echo "Symlink created."
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup

deactivate || true
module load StdEnv/2023
module load python-build-bundle/2025a
module load python/3.11
module load arrow/21.0.0
module load go/1.21.3
module load rust/1.91.0

[ -d .venv ] || python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade maturin
python -m pip install pdex

while read -r pkg; do
  [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
  pip install --no-index "$pkg" || pip install "$pkg"
done < requirements.txt

# -----------------------------------------------------------------------------
# Make nanochat importable from anywhere

rm -rf .venv/lib/*/site-packages/nanochat
cp -r nanochat .venv/lib/*/site-packages/

# -----------------------------------------------------------------------------
# Tokenizer

# Build the rustbpe Tokenizer wheel using the venv's maturin
PYTHONPATH= python -m maturin build --release --manifest-path rustbpe/Cargo.toml

# Install the rustbpe Tokenizer wheel
PYTHONPATH= python -m pip install --force-reinstall --no-deps rustbpe/target/wheels/nanochat-0.1.0-cp311-cp311-linux_x86_64.whl

# Download the first ~2B characters of pretraining dataset
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 500 &
DATASET_DOWNLOAD_PID=$!

# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000

# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"
echo "Dataset download completed."
