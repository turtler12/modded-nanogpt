#!/bin/bash
#SBATCH --job-name=diff_mappings_muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=1              
#SBATCH --output=/data/scratch/medhaven/logs/identity_diff_mappings_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/identity_diff_mappings_muon_%j.err

# baseline
#SPECTRAL_RECIPE=NS
# R1: huberized ns
#SPECTRAL_RECIPE=R1 R1_TAU0=0.05 R1_SLOPE=0.9 CHEB_K=5
# R2: power compression
#SPECTRAL_RECIPE=R2 R2_ALPHA=0.7 CHEB_K=5
# R3: soft thresholding + cap
#SPECTRAL_RECIPE=R3 R3_LAMBDA=0.05 CHEB_K=5
# identity mapping
SPECTRAL_RECIPE=ID


# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# export TORCH_NCCL_BLOCKING_WAIT=0
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=WARN
# export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --standalone --nproc_per_node=1 train_gpt_diff_mappings.py
#torchrun --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE:-8} train_gpt_diff_mappings.py
