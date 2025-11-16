#!/bin/bash
#SBATCH --job-name=step_function_svd
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/svd_step_function_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/svd_step_function_%j.err

export MAP_KIND="my_function"   # uses svd_map_identity in your code

# comparing overshooting

echo "[cfg] MAP_KIND=$MAP_KIND MAP_NORMALIZE=$MAP_NORMALIZE"
torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py
