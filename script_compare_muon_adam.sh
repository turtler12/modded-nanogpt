#!/bin/bash
#SBATCH --job-name=compare_all_optimizers
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=1                  
#SBATCH --output=/data/scratch/medhaven/logs/compare_optimizers_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/compare_optimizers_%j.err

torchrun --standalone --nproc_per_node=1 train_gpt_compare_muon_and_adam.py
