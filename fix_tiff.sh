#!/bin/bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
source ~/.bashrc
micromamba create -p ~/scratch/libtiff-env libtiff -y
export LD_LIBRARY_PATH=$HOME/scratch/libtiff-env/lib:${LD_LIBRARY_PATH}
