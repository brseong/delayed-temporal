#!/bin/bash

# Run the manuscript appendix theta-sensitivity scan through the maintained,
# resumable ViT-B Gaussian protocol. Physical GPUs are restricted by the shared
# runner to 4-7, with one evaluator process per selected GPU.

set -euo pipefail
trap 'kill -- -$$' SIGINT SIGTERM

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
read -r -a theta_values <<< "${THETA_VALUES:-40 400 2000}"
scan_root_tag="${SCAN_ROOT_TAG:-vit_base_theta_noise_5k_float64_v2}"
figure_prefix="${THETA_FIGURE_PREFIX:-$repo_root/artifacts/figures/noise_theta_vit_base_5k_float64_v2}"
combined_csv="${THETA_SUMMARY_CSV:-$repo_root/artifacts/logs/noise_scan/$scan_root_tag/summary.csv}"

base_grid=(
    1.000e-10 1.250e-10 1.500e-10 1.750e-10 2.000e-10 2.500e-10
    3.162e-10 4.000e-10 5.000e-10 6.300e-10 8.000e-10 1.000e-09
)

summary_inputs=()
for theta in "${theta_values[@]}"; do
    if [[ "$theta" != "40" && "$theta" != "400" && "$theta" != "2000" ]]; then
        echo "THETA_VALUES may contain only the manuscript values 40, 400, and 2000" >&2
        exit 2
    fi

    case "$theta" in
        40) scale="50" ;;
        400) scale="5" ;;
        2000) scale="1" ;;
    esac
    scaled_grid="$($python_bin - "$scale" "${base_grid[@]}" <<'PY'
import sys

scale = float(sys.argv[1])
print(" ".join(f"{float(value) * scale:.7g}" for value in sys.argv[2:]))
PY
)"

    child_tag="${scan_root_tag}_theta_${theta}"
    child_logdir="$repo_root/artifacts/logs/noise_scan/$child_tag"
    THETA="$theta" \
    SCAN_PROTOCOL=quick \
    SCAN_TAG="$child_tag" \
    SCAN_MODEL_LABEL="ViT-B/16, theta=$theta" \
    SCAN_FIGURE_PREFIX="$repo_root/artifacts/figures/${child_tag}" \
    TIME_NOISE_STD_FRACS="$scaled_grid" \
    MISMATCH_THETA_STDS="" \
    PUBLISH_PAPER_FIGURE=0 \
    bash "$repo_root/scripts/experiments/noise_scan_vit.sh"
    summary_inputs+=("${theta}=${child_logdir}/summary.csv")
done

"$python_bin" scripts/analysis/summarize_theta_noise_scan.py \
    --input "${summary_inputs[@]}" \
    --output-csv "$combined_csv" \
    --figure-prefix "$figure_prefix"

if [[ "${PUBLISH_PAPER_FIGURE:-0}" == "1" ]]; then
    cp "$figure_prefix.pdf" "$repo_root/paper/figures/noise-theta-vit-base.pdf"
    echo "Published validated appendix figure: paper/figures/noise-theta-vit-base.pdf"
fi
