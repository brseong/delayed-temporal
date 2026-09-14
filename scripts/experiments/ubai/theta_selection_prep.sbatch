#!/bin/bash
#SBATCH --job-name=theta-prep
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

: "${THETA_REMOTE_ASSETS:?THETA_REMOTE_ASSETS is required}"
: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_CONTAINER_IMAGE:?THETA_CONTAINER_IMAGE is required}"

cd "$THETA_REMOTE_ASSETS"
sha256sum --check SHA256SUMS

test -f "$THETA_ENV_ARCHIVE"

mkdir -p "$(dirname "$THETA_CONTAINER_IMAGE")"
if [[ ! -f "$THETA_CONTAINER_IMAGE" ]]; then
    srun \
        --container-image=ubuntu:24.04 \
        --container-save="$THETA_CONTAINER_IMAGE" \
        /bin/true
fi
test -f "$THETA_CONTAINER_IMAGE"

echo "Theta selection assets prepared"
