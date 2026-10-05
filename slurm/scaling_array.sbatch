#!/bin/bash
# One (seed, N_total) cell of the scaling study per array task.
#
#   python scripts/scaling_study.py --list-tasks      # prints the grid; default 3 seeds x 6 budgets = 18
#   sbatch slurm/scaling_array.sbatch                 # submit
#   python scripts/plot_scaling.py --results results/scaling   # after all tasks finish
#
# Finished cells are skipped, so resubmitting after a failure only reruns the missing ones.
# Edit the partition / account / environment lines for your cluster.

#SBATCH --job-name=usbi-scaling
#SBATCH --array=0-17
#SBATCH --time=06:00:00
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
# source ~/venvs/unified-sbi/bin/activate      # or: conda activate unified-sbi
# -----------------------------------------------------------------------------

OUT=${OUT:-results/scaling}
EXTRA_ARGS=${EXTRA_ARGS:-}                     # e.g. EXTRA_ARGS="--learn-scales" sbatch ...

python scripts/scaling_study.py \
    --out "$OUT" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --threads "$SLURM_CPUS_PER_TASK" \
    $EXTRA_ARGS
