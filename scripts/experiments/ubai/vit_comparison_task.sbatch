#!/bin/bash
#SBATCH --job-name=vit-comparison-pair
#SBATCH --partition=gpu4,gpu5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?A Slurm allocation is required}"
: "${EXPERIMENT_SOURCE:?EXPERIMENT_SOURCE is required}"
: "${EXPERIMENT_DEPLOYMENT:?EXPERIMENT_DEPLOYMENT is required}"
exec /home1/sizz1997/miniconda3/bin/python \
    "$EXPERIMENT_SOURCE/scripts/experiments/ubai/run_vit_comparison_ubai.py" \
    --deployment "$EXPERIMENT_DEPLOYMENT"
