#!/bin/bash
#SBATCH --job-name=calibrated-three-sweep-pair
#SBATCH --partition=gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${EXPERIMENT_SOURCE:?EXPERIMENT_SOURCE is required}"
: "${EXPERIMENT_DEPLOYMENT:?EXPERIMENT_DEPLOYMENT is required}"
: "${CONTROLLER_SOURCE:?CONTROLLER_SOURCE is required}"
: "${CONTROLLER_COMMIT:?CONTROLLER_COMMIT is required}"
: "${PAIR_MANIFEST:?PAIR_MANIFEST is required}"
exec /home1/sizz1997/miniconda3/bin/python \
    "$CONTROLLER_SOURCE/scripts/experiments/ubai/run_calibrated_three_sweep_pair.py" \
    --run-host
