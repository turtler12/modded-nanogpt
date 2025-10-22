#!/bin/bash
#SBATCH --job-name=muon_vs_adam
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=1                  
#SBATCH --output=/data/scratch/medhaven/logs/job_output_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/job_output_%j.err

torchrun --standalone --nproc_per_node=1 train_gpt_compare_muon_and_adam.py
