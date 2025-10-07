#!/bin/bash
#
#SBATCH --job-name=nanogpt-h100
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:08:00               # Adjust time as needed
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


set -euxo pipefail

# Always run from repo root (where this file lives)
cd "$(dirname "$0")"

# Make sure log/record dirs exist
mkdir -p logs records

# If you use modules on Lingo, uncomment and set versions:
# module load cuda/12.2
# module load gcc/12.3

# previous version
torchrun --standalone --nproc_per_node=1 train_gpt.py

# ---- Choose ONE of these runs ----
# 1) Default training
#python -u train_gpt.py 2>&1 | tee "logs/train_%j.log"

# 2) Or the medium config
# python -u train_gpt_medium.py 2>&1 | tee "logs/train_medium_%j.log"
