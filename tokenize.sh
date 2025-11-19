#!/bin/bash

# -----------------------------------------------------------------------------
# Python venv setup

deactivate || true
module load StdEnv/2023
module load python-build-bundle/2025a
module load python/3.11
module load arrow/21.0.0
module load go/1.21.3
module load rust/1.91.0
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Tokenizer

# Build the rustbpe Tokenizer wheel using the venv's maturin
PYTHONPATH= python -m maturin build --release --manifest-path rustbpe/Cargo.toml

# Install the rustbpe Tokenizer wheel
PYTHONPATH= python -m pip install --force-reinstall --no-deps rustbpe/target/wheels/nanochat-0.1.0-cp311-cp311-linux_x86_64.whl

# Download the first ~2B characters of pretraining dataset
python -m nanochat.dataset -n 8

# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000

# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval
