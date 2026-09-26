# Dynamic GPU-pool scheduler (source this file; requires bash >= 4.3 for `wait -n`).
#
# Keeps at most one job per GPU and assigns each new job to whichever GPU frees up first, so a
# queue of single-GPU experiments longer than the GPU list is drained without ever overloading a
# GPU. Usage:
#
#   source "${repo_root}/scripts/lib/gpu_pool.sh"
#   gpu_pool_init "${cuda_devices[@]}"
#   for index in "${!expr_names[@]}"; do
#       gpu_pool_acquire                 # blocks until a GPU is free; sets $GPU_POOL_ACQUIRED
#       gpu=$GPU_POOL_ACQUIRED
#       CUDA_VISIBLE_DEVICES=${gpu} python3 ... &
#       gpu_pool_register $! "$gpu"      # record which GPU this job occupies
#   done
#   wait
#
# NOTE: gpu_pool_acquire returns the id via the global $GPU_POOL_ACQUIRED rather than stdout on
# purpose — a `$(gpu_pool_acquire)` command substitution would run in a subshell whose `wait -n`
# cannot see the parent shell's background jobs, breaking the blocking.

declare -A _GPU_POOL_PID2GPU
_GPU_POOL_FREE=()
GPU_POOL_ACQUIRED=""

gpu_pool_init() {
    _GPU_POOL_FREE=("$@")
    _GPU_POOL_PID2GPU=()
}

_gpu_pool_reclaim() {   # return every finished job's GPU to the free pool
    local pid
    for pid in "${!_GPU_POOL_PID2GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            _GPU_POOL_FREE+=("${_GPU_POOL_PID2GPU[$pid]}")
            unset '_GPU_POOL_PID2GPU[$pid]'
        fi
    done
}

gpu_pool_acquire() {    # block until a GPU is free, then set $GPU_POOL_ACQUIRED
    while [ ${#_GPU_POOL_FREE[@]} -eq 0 ]; do
        wait -n
        _gpu_pool_reclaim
    done
    GPU_POOL_ACQUIRED=${_GPU_POOL_FREE[0]}
    _GPU_POOL_FREE=("${_GPU_POOL_FREE[@]:1}")
}

gpu_pool_register() {   # $1 = background pid, $2 = GPU it occupies
    _GPU_POOL_PID2GPU[$1]=$2
}
