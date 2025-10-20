#!/bin/bash
#
#SBATCH --job-name=muon
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:25:00 # (hh:mm:ss)
#SBATCH --output=/data/scratch/medhaven/logs/job_output_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/job_output_%j.err
#SBATCH --gpus=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G

torchrun --standalone --nproc_per_node=1 train_gpt.py
