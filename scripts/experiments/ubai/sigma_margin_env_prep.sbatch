#!/bin/bash
#SBATCH --job-name=sigma-margin-env
#SBATCH --partition=cpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

: "${THETA_ENV_ARCHIVE:?THETA_ENV_ARCHIVE is required}"
: "${THETA_ENV_ROOT:?THETA_ENV_ROOT is required}"

environment_parent="$(dirname -- "$THETA_ENV_ROOT")"
lock_path="$environment_parent/.sigma-margin-env.lock"
marker="$THETA_ENV_ROOT/.archive.sha256"
mkdir -p "$environment_parent"
exec 9> "$lock_path"
flock 9

if [[ -x "$THETA_ENV_ROOT/bin/python" && -s "$marker" ]]; then
    echo "Shared environment is ready: $THETA_ENV_ROOT"
    exit 0
fi
if [[ -e "$THETA_ENV_ROOT" ]]; then
    echo "Shared environment exists but is incomplete: $THETA_ENV_ROOT" >&2
    exit 2
fi

staging="$environment_parent/.dt-shared.${SLURM_JOB_ID:?SLURM_JOB_ID is required}.partial"
cleanup() {
    rm -rf -- "$staging"
}
trap cleanup EXIT
mkdir -p "$staging"
archive_sha256="$(sha256sum "$THETA_ENV_ARCHIVE" | awk '{print $1}')"
zstd --decompress --stdout "$THETA_ENV_ARCHIVE" | tar -xf - -C "$staging"
test -x "$staging/dt/bin/python"
printf '%s  %s\n' "$archive_sha256" "$(basename -- "$THETA_ENV_ARCHIVE")" \
    > "$staging/dt/.archive.sha256"
mv "$staging/dt" "$THETA_ENV_ROOT"
rmdir "$staging"
trap - EXIT
echo "Prepared shared environment: $THETA_ENV_ROOT"
