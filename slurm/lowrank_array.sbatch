#!/bin/bash
# Rank-deficient benchmark (hidden 2D Rosenbrock in 8D) with the (k, m) screen:
# one (seed, N_total) cell per array task.
#
#   python scripts/lowrank_scaling.py --list-tasks     # default 3 seeds x 6 budgets = 18
#   sbatch slurm/lowrank_array.sbatch
#   python scripts/plot_lowrank.py --results results/lowrank
#
# Variants: EXTRA_ARGS="--reuse" or "--discovery fixed --discovery-value 500" or
# "--selection fixed --k 2 --m 2" (use a different OUT for each).

#SBATCH --job-name=usbi-lowrank
#SBATCH --array=0-17
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1                 # delete this line to run on CPU nodes
##SBATCH --partition=your-partition
##SBATCH --account=your-account
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs

# --- environment (edit) ------------------------------------------------------
# module load python/3.11 cuda/12.1
# source ~/venvs/unified-sbi/bin/activate
# -----------------------------------------------------------------------------

OUT=${OUT:-results/lowrank}
EXTRA_ARGS=${EXTRA_ARGS:-}

python scripts/lowrank_scaling.py \
    --out "$OUT" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --threads "$SLURM_CPUS_PER_TASK" \
    $EXTRA_ARGS
