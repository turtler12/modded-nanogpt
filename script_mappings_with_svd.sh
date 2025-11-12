#!/bin/bash
#SBATCH --job-name=svd_mappings
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --array=1-10
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/data/scratch/medhaven/logs/svd_maps_%A_%a.log
#SBATCH --error=/data/scratch/medhaven/logs/svd_maps_%A_%a.err

# Usage examples:
#   SWEEP=yclip  sbatch this_script.sh          # sweeps YCLIP in {0.1,...,1.0}
#   SWEEP=affine sbatch this_script.sh          # sweeps INTERCEPT in {0.0,0.1,...,0.9}
#   SWEEP=polar  sbatch --array=1-1 this_script # runs Polar (no sweep needed)
#
# Notes:
# - All modes set MAP_NORMALIZE=1 (per-matrix normalization) to match your training code.
# - The run_id printed by the Python script already includes a tag (affXX / yclipXX / polar).

set -euo pipefail

SWEEP="${SWEEP:-yclip}"         # yclip | affine | polar
export MAP_NORMALIZE=1          # keep op-norm <= 1 (matches your code path)

case "$SWEEP" in
  yclip)
    # Array: 1..10 -> YCLIP 0.1..1.0
    if [ "$SLURM_ARRAY_TASK_ID" -lt 10 ]; then
      export YCLIP="0.${SLURM_ARRAY_TASK_ID}"
    else
      export YCLIP="1.0"
    fi
    export MAP_KIND="yclip"
    echo "[cfg] SWEEP=yclip  YCLIP=${YCLIP}  MAP_NORMALIZE=${MAP_NORMALIZE}"
    ;;

  affine)
    # Array: 1..10 -> INTERCEPT 0.0, 0.1, ..., 0.9
    # (If you want 10 tiny intercepts instead, change the table below.)
    INTERCEPTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    idx=$((SLURM_ARRAY_TASK_ID - 1))
    if [ $idx -lt 0 ] || [ $idx -ge ${#INTERCEPTS[@]} ]; then
      echo "Invalid array index ${SLURM_ARRAY_TASK_ID} for affine sweep"; exit 2
    fi
    export INTERCEPT="${INTERCEPTS[$idx]}"
    export MAP_KIND="affine"
    echo "[cfg] SWEEP=affine  INTERCEPT=${INTERCEPT}  MAP_NORMALIZE=${MAP_NORMALIZE}"
    ;;

  polar)
    # Polar: single run; ignore array index (set array to 1-1 when submitting).
    export MAP_KIND="polar"
    echo "[cfg] SWEEP=polar (Polar-Express-equivalent SVD map)  MAP_NORMALIZE=${MAP_NORMALIZE}"
    ;;

  *)
    echo "Unknown SWEEP='$SWEEP' (use: yclip | affine | polar)"; exit 1
    ;;
esac

torchrun --standalone --nproc_per_node=1 train_gpt_mappings_with_svd.py
