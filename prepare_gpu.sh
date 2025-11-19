#!/bin/bash
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
set -e
module load cudacore/.12.6.3 
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
