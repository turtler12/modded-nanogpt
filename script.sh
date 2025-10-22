#!/bin/bash
#SBATCH --job-name=muon_ns_clip_sweep
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --array=0-8
#SBATCH --output=/data/scratch/medhaven/logs/job_output_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/job_output_%A_%a.err

set -euo pipefail

module load cuda/12.2 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

export NS_EQN_IDX=${SLURM_ARRAY_TASK_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

torchrun --standalone --nproc_per_node=1 train_gpt_with_xclips.py
