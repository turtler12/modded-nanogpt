#!/bin/bash
#SBATCH --job-name=svd_mappings
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --array=1-10
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/svd_y_clips_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/svd_y_clips_%A_%a.err

# Map array index -> clip: 1..9 -> 0.1..0.9, 10 -> 1.0
if [ "$SLURM_ARRAY_TASK_ID" -lt 10 ]; then
  export YCLIP="0.${SLURM_ARRAY_TASK_ID}"
else
  export YCLIP="1.0"
fi
export YCLIP_NORMALIZE=1

torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py
