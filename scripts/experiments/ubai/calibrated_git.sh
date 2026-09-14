#!/bin/sh
# Run the bundled Git binary with private libraries inside the evaluation container.
set -eu
git_runtime_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LD_LIBRARY_PATH="$git_runtime_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    exec "$git_runtime_dir/git.bin" "$@"
