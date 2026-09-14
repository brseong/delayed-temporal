#!/bin/bash
#SBATCH --job-name=calibrated-three-sweep
#SBATCH --partition=gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${EXPERIMENT_DEPLOYMENT:?EXPERIMENT_DEPLOYMENT is required}"
: "${EXPERIMENT_SOURCE:?EXPERIMENT_SOURCE is required}"
# The source checkout and deployment are independently checked before extraction.
exec /home1/sizz1997/miniconda3/bin/python \
    "$EXPERIMENT_SOURCE/scripts/experiments/ubai/prepare_calibrated_three_sweeps_ubai.py" \
    --run-task --deployment "$EXPERIMENT_DEPLOYMENT"
