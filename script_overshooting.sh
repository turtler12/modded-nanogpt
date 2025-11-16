#!/bin/bash
#SBATCH --job-name=overshooting_comparison
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=05:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

# Disable SLURM output completely:
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

export MAP_KIND="my_function"
export MAP_NORMALIZE=1

LOGDIR="/data/scratch/medhaven/logs"

for OVERSHOOT in 0 1; do
    export OVERSHOOT

    if [ "$OVERSHOOT" -eq 1 ]; then
        OUT_STD="$LOGDIR/with_overshooting.log"
        OUT_ERR="$LOGDIR/with_overshooting.err"
    else
        OUT_STD="$LOGDIR/without_overshooting.log"
        OUT_ERR="$LOGDIR/without_overshooting.err"
    fi

    echo "[cfg] MAP_KIND=$MAP_KIND OVERSHOOT=$OVERSHOOT MAP_NORMALIZE=$MAP_NORMALIZE" \
        > "$OUT_STD"

    # Save stdout and stderr separately
    torchrun --standalone --nproc_per_node=1 test_overshooting.py \
        >> "$OUT_STD" \
        2>> "$OUT_ERR"
done
