#!/bin/bash
#SBATCH --job-name=compare_all_optimizers_MEDIUM_GPT
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --output=/data/scratch/medhaven/logs/MED_compare_optimizers_%j.log
#SBATCH --error=/data/scratch/medhaven/logs/MED_compare_optimizers_%j.err

# Run Muon, then Adam, then SGD on the medium GPT.
# Uses the OPTIM_KIND env var handled inside train_gpt_medium_compare_optimizers.py.
#export TORCHDYNAMO_DISABLE=1
i=0
for kind in muon adam sgd; do
  export OPTIM_KIND=$kind
  export RUN_ID=$i

  echo "=== Starting OPTIM_KIND=${OPTIM_KIND} RUN_ID=${RUN_ID} ==="

  torchrun --standalone --nproc_per_node=1 train_gpt_medium_compare_optimizers.py

  ((i++))
done
