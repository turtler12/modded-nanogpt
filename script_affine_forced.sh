#!/bin/bash
#SBATCH --job-name=svd_forced_affine
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --array=1-7
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/affine_forced_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/affine_forced_%A_%a.err

# List of intercepts, one per array task (1..7)
INTERCEPTS=(0 0.3 0.68 0.75 0.8 0.9 1)

# Map SLURM_ARRAY_TASK_ID (1–7) -> corresponding intercept
idx=$((SLURM_ARRAY_TASK_ID - 1))
export INTERCEPT="${INTERCEPTS[$idx]}"

# Keep spectral norm <= 1 by normalizing singular values to [0,1] before mapping.
export MAP_NORMALIZE=1
export MAP_KIND="affine"     # choose our affine mapping

echo "[$(date)] Running affine mapping with INTERCEPT=${INTERCEPT} (task ${SLURM_ARRAY_TASK_ID})"

torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py
