#!/bin/bash
#SBATCH --job-name=muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

#SBATCH --output=/data/scratch/medhaven/logs/MED_better_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/MED_better_muon_%j.err

# Disable TorchDynamo / Inductor completely
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Enable expandable memory segments (critical for H100)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL robustness
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# Good default (async execution)
export CUDA_LAUNCH_BLOCKING=0

torchrun --standalone --nproc_per_node=4 train_gpt_medium_original.py
