#!/bin/bash
#SBATCH --job-name=compare_all_optimizers_MEDIUM_GPT
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --array=0-2
#SBATCH --output=/data/scratch/medhaven/logs/MED_compare_optimizers_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/MED_compare_optimizers_%A_%a.err

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

opts=(muon adam sgd)
HIDDEN_OPTIM=${opts[$SLURM_ARRAY_TASK_ID]}
RUN_ID=$SLURM_ARRAY_TASK_ID

echo "Starting HIDDEN_OPTIM=${HIDDEN_OPTIM} RUN_ID=${RUN_ID} (array task ${SLURM_ARRAY_TASK_ID})"

# torchrun will spawn N processes (one per GPU on the node)
torchrun --standalone --nproc_per_node=1 train_gpt_medium_compare_optimizers.py