#!/bin/bash
#SBATCH --job-name=calibrated-noise-prep
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Validate the calibrated deployment on a CPU node without allocating a GPU.
set -euo pipefail
: "${CALIBRATED_DEPLOYMENT:?CALIBRATED_DEPLOYMENT is required}"
export CALIBRATED_CHECK_ONLY=1
bash "$(dirname "$CALIBRATED_DEPLOYMENT")/code/calibrated_noise_task.sbatch"
