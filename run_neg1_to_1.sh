#!/bin/bash
#SBATCH --job-name=svd_step_neg1to1
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --array=1-7
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/neg1_to_1.log
#SBATCH --error=/data/scratch/medhaven/logs/neg1_to_1.err
#SBATCH --open-mode=append

# epsilons mapped to array tasks 1..7
EPSILONS=(0 0.005 0.01 0.015 0.02 0.04 0.07)
idx=$((SLURM_ARRAY_TASK_ID - 1))
export MAP_EPS="${EPSILONS[$idx]}"

# mapping config
export MAP_KIND="step"
export MAP_RANGE="m1_1"  # -1 → 1
export MAP_NORMALIZE=1

echo "[$(date)] RANGE=${MAP_RANGE} EPS=${MAP_EPS} (task ${SLURM_ARRAY_TASK_ID}) on ${HOSTNAME}"

# run (assumes cwd has train_gpt_mappings_with_svd.py)
torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py
