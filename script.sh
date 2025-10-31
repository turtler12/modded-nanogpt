#!/bin/bash
#SBATCH --job-name=muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --gpus=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

#SBATCH --output=/data/scratch/medhaven/logs/original_muon_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/original_muon_%j.err


torchrun --standalone --nproc_per_node=8 train_gpt.py
