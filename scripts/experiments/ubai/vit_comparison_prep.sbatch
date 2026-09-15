#!/bin/bash
#SBATCH --job-name=vit-comparison-prep
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${EXPERIMENT_SOURCE:?EXPERIMENT_SOURCE is required}"
: "${EXPERIMENT_DEPLOYMENT:?EXPERIMENT_DEPLOYMENT is required}"
exec /home1/sizz1997/miniconda3/bin/python \
    "$EXPERIMENT_SOURCE/scripts/experiments/ubai/run_vit_comparison_ubai.py" \
    --prep --deployment "$EXPERIMENT_DEPLOYMENT"
