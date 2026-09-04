#!/bin/bash

# Submit the next dependency-delimited stage of the UBAI theta-selection workflow.
# This controller performs only manifest generation and Slurm submissions; model
# evaluation remains confined to theta_selection_task.sbatch GPU jobs.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 start-selection|post-selection|post-extension|post-confirmation|finalize" >&2
    exit 2
fi

phase="$1"

: "${THETA_SOURCE_COMMIT:?THETA_SOURCE_COMMIT is required}"
: "${THETA_CHECKPOINT_PATH:?THETA_CHECKPOINT_PATH is required}"
: "${THETA_CHECKPOINT_SHA256:?THETA_CHECKPOINT_SHA256 is required}"
: "${THETA_DATASET_MANIFEST:?THETA_DATASET_MANIFEST is required}"
: "${THETA_LOG_DIR:?THETA_LOG_DIR is required}"
: "${THETA_OUTPUT_DIR:?THETA_OUTPUT_DIR is required}"
: "${THETA_MANIFEST_DIR:?THETA_MANIFEST_DIR is required}"
: "${THETA_REMOTE_REPO:?THETA_REMOTE_REPO is required}"
: "${THETA_CONTROL_SCRIPT:?THETA_CONTROL_SCRIPT is required}"

mkdir -p "$THETA_LOG_DIR/slurm" "$THETA_OUTPUT_DIR" "$THETA_MANIFEST_DIR"
workflow_log="$THETA_OUTPUT_DIR/workflow.log"

record() {
    printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$workflow_log"
}

json_value() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$1" "$2"
}

submit_controller() {
    local next_phase="$1"
    local dependency="$2"
    sbatch \
        --parsable \
        --job-name="theta-${next_phase}" \
        --partition=cpu1 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=2G \
        --time=00:10:00 \
        --dependency="afterok:${dependency}" \
        --output="$THETA_LOG_DIR/slurm/%x-%j.out" \
        --error="$THETA_LOG_DIR/slurm/%x-%j.err" \
        --export=ALL \
        --wrap="bash '$THETA_CONTROL_SCRIPT' '$next_phase'"
}

build_manifest() {
    local stage="$1"
    local output="$2"
    shift 2
    python3 "$THETA_REMOTE_REPO/scripts/experiments/ubai/build_theta_selection_manifest.py" \
        --stage "$stage" \
        --output "$output" \
        --source-commit "$THETA_SOURCE_COMMIT" \
        --checkpoint-path "$THETA_CHECKPOINT_PATH" \
        --checkpoint-sha256 "$THETA_CHECKPOINT_SHA256" \
        --dataset-manifest "$THETA_DATASET_MANIFEST" \
        --gpu-selection "$THETA_OUTPUT_DIR/gpu-selection.json" \
        "$@"
}

selected_partition() {
    json_value "$THETA_OUTPUT_DIR/gpu-selection.json" selected_partition
}

submit_array() {
    local manifest="$1"
    local time_limit="$2"
    bash "$THETA_REMOTE_REPO/scripts/experiments/ubai/submit_theta_selection_stage.sh" \
        "$manifest" "$(selected_partition)" "$time_limit"
}

submit_reducer() {
    local mode="$1"
    local manifests="$2"
    local dependency="$3"
    bash "$THETA_REMOTE_REPO/scripts/experiments/ubai/submit_theta_selection_reduce.sh" \
        "$mode" "$manifests" "$dependency" "$THETA_OUTPUT_DIR"
}

active_selection_manifest() {
    if [[ -f "$THETA_MANIFEST_DIR/selection-expanded.tsv" ]]; then
        printf '%s\n' "$THETA_MANIFEST_DIR/selection-expanded.tsv"
    else
        printf '%s\n' "$THETA_MANIFEST_DIR/selection.tsv"
    fi
}

submit_confirmation() {
    local selection_manifest
    local confirmation_manifest="$THETA_MANIFEST_DIR/confirmation.tsv"
    selection_manifest="$(active_selection_manifest)"
    build_manifest confirmation "$confirmation_manifest" \
        --selection-json "$THETA_OUTPUT_DIR/selection.json"
    local array_job reducer_job controller_job
    array_job="$(submit_array "$confirmation_manifest" 03:00:00)"
    reducer_job="$(submit_reducer confirmation "$selection_manifest:$confirmation_manifest" "$array_job")"
    controller_job="$(submit_controller post-confirmation "$reducer_job")"
    record "confirmation array=$array_job reducer=$reducer_job controller=$controller_job"
}

case "$phase" in
    start-selection)
        [[ -n "$(json_value "$THETA_OUTPUT_DIR/gpu-selection.json" selected_family)" ]]
        manifest="$THETA_MANIFEST_DIR/selection.tsv"
        build_manifest selection "$manifest"
        array_job="$(submit_array "$manifest" 03:00:00)"
        reducer_job="$(submit_reducer selection "$manifest" "$array_job")"
        controller_job="$(submit_controller post-selection "$reducer_job")"
        record "selection array=$array_job reducer=$reducer_job controller=$controller_job"
        ;;
    post-selection)
        status="$(json_value "$THETA_OUTPUT_DIR/selection.json" status)"
        if [[ "$status" == "needs_extension" ]]; then
            manifest="$THETA_MANIFEST_DIR/selection-expanded.tsv"
            build_manifest selection "$manifest" --extension
            array_job="$(submit_array "$manifest" 03:00:00)"
            reducer_job="$(submit_reducer selection "$manifest" "$array_job")"
            controller_job="$(submit_controller post-extension "$reducer_job")"
            record "extension array=$array_job reducer=$reducer_job controller=$controller_job"
        elif [[ "$status" == "selected" ]]; then
            submit_confirmation
        else
            record "stopped after selection status=$status"
            exit 1
        fi
        ;;
    post-extension)
        status="$(json_value "$THETA_OUTPUT_DIR/selection.json" status)"
        if [[ "$status" != "selected" ]]; then
            record "stopped after extension status=$status"
            exit 1
        fi
        submit_confirmation
        ;;
    post-confirmation)
        status="$(json_value "$THETA_OUTPUT_DIR/selection.json" status)"
        if [[ "$status" != "confirmed" ]]; then
            record "stopped after confirmation status=$status"
            exit 1
        fi
        selection_manifest="$(active_selection_manifest)"
        confirmation_manifest="$THETA_MANIFEST_DIR/confirmation.tsv"
        full_manifest="$THETA_MANIFEST_DIR/full.tsv"
        build_manifest full "$full_manifest" \
            --selection-json "$THETA_OUTPUT_DIR/selection.json"
        array_job="$(submit_array "$full_manifest" 08:00:00)"
        reducer_job="$(submit_reducer full "$selection_manifest:$confirmation_manifest:$full_manifest" "$array_job")"
        controller_job="$(submit_controller finalize "$reducer_job")"
        record "full array=$array_job reducer=$reducer_job controller=$controller_job"
        ;;
    finalize)
        status="$(json_value "$THETA_OUTPUT_DIR/selection.json" status)"
        if [[ "$status" != "approved" ]]; then
            record "final reducer did not approve selection: status=$status"
            exit 1
        fi
        selected_theta="$(json_value "$THETA_OUTPUT_DIR/selection.json" selected_theta)"
        record "complete selected_theta=$selected_theta"
        : > "$THETA_OUTPUT_DIR/WORKFLOW_COMPLETE"
        ;;
    *)
        echo "Unknown workflow phase: $phase" >&2
        exit 2
        ;;
esac
