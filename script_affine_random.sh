#!/bin/bash
#SBATCH --job-name=svd_affine_rand
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=06:00:00
#SBATCH --array=1-8             # 8 random intercepts
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/affine_rand_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/affine_rand_%A_%a.err

# Generate a random affine intercept between 0 and 1
export MAP_KIND="affine"
export MAP_NORMALIZE=1
# export INTERCEPT=$(awk -v seed=$RANDOM 'BEGIN {srand(seed); print rand()}') # for 8 random values


# Job index logic:
if [ "$SLURM_ARRAY_TASK_ID" -lt 8 ]; then
  # For jobs 1–7: random intercept between 0 and 1
  export INTERCEPT=$(awk -v seed=$RANDOM 'BEGIN {srand(seed); print rand()}')
else
  # For job 8: fixed intercept = 1.0
  export INTERCEPT=1.0
fi

echo "[$(date)] Running affine mapping with INTERCEPT=${INTERCEPT}"

torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py
