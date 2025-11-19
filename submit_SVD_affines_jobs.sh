#!/bin/bash
# Submit five independent sbatch jobs to run train_gpt_medium_SVD_affines.py
# Each job sets MAP_KIND=affine and INTERCEPT to one of: 0.00,0.25,0.50,0.75,1.00
#SBATCH --job-name=affines_MEDIUM_GPT
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=2
#SBATCH --array=0-4
#SBATCH --output=/data/scratch/medhaven/logs/affines_MEDIUM_GPT_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/affines_MEDIUM_GPT_%A_%a.err

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



intercepts=(0.00 0.25 0.50 0.75 1.00)

# Use SLURM array task id to select intercept
idx=${SLURM_ARRAY_TASK_ID:-0}
v=${intercepts[$idx]}
RUN_ID=$idx

echo "Starting array task ${SLURM_ARRAY_TASK_ID}: INTERCEPT=${v} RUN_ID=${RUN_ID}"

# export mapping env vars for the Python script
export MAP_KIND=affine
export INTERCEPT=${v}
export RUN_ID=${RUN_ID}

torchrun --standalone --nproc_per_node=2 train_gpt_medium_SVD_affines.py
