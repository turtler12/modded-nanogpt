#!/bin/bash
#SBATCH --job-name=svd_mappings
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/svd_mappings_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/svd_mappings_%j.err


torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py