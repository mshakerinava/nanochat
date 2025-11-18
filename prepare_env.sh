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
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# # create a .venv local virtual environment (if it doesn't exist)
# [ -d ".venv" ] || uv venv
# # install the repo dependencies
# uv sync --extra gpu
# # activate venv so that `python` uses the project's venv instead of system python
# source .venv/bin/activate

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run --no-sync maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 500 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval


echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID
echo "Dataset download completed."
