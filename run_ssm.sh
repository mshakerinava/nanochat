#!/bin/bash
#SBATCH --gres=gpu:h100:4
#SBATCH -c 6
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --partition=short-unkillable
#SBATCH --time=03:00:00

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

MODEL_TYPE=ssm bash speedrun_ssm.sh
