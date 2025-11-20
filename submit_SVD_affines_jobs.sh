#!/bin/bash
#SBATCH --job-name=affines_MEDIUM_GPT
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gpus=2
#SBATCH --array=0-5%2
#SBATCH --output=/data/scratch/medhaven/logs/MEDv2_affines_%A_%a.out
#SBATCH --error=/data/scratch/medhaven/logs/MEDv2_affines_%A_%a.err

# --- Runtime env hygiene ---
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

# 6 runs to schedule (array index 0..5)
NORMALIZE=(1 1 1 0 0 0)
INTERCEPT=(0.00 0.50 1.00 0.00 0.50 1.00)

# Pick config for this array task
IDX=${SLURM_ARRAY_TASK_ID}
export MAP_KIND="affine"
export MAP_NORMALIZE=${NORMALIZE[$IDX]}
export INTERCEPT=${INTERCEPT[$IDX]}
export APPLY_INIT_MAP=0

echo "[$(date)] idx=$IDX  MAP_KIND=$MAP_KIND  MAP_NORMALIZE=$MAP_NORMALIZE  INTERCEPT=$INTERCEPT  APPLY_INIT_MAP=$APPLY_INIT_MAP"
echo "GPUs requested: $SLURM_GPUS_PER_TASK  on $SLURM_JOB_NODELIST"

# Activate env so torchrun is on PATH
source /data/scratch/medhaven/miniconda3/etc/profile.d/conda.sh
conda activate base
python -V
python -c "import torch,shutil; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('torchrun:', shutil.which('torchrun'))"

# Launch (2 GPUs per task)
torchrun --standalone --nproc_per_node=2 train_gpt_medium_SVD_affines.py
