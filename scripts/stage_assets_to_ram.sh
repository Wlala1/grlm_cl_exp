#!/bin/bash
# Source this script to stage immutable image assets into RAM before dispatching jobs.

if [ "${USE_RAM_ASSETS:-0}" != "1" ]; then
    return 0 2>/dev/null || exit 0
fi

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_MODEL_DIR=${SOURCE_MODEL_DIR:-${WORK_DIR}/models}
SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-${WORK_DIR}/data}
RAM_ASSET_ROOT=${RAM_ASSET_ROOT:-/dev/shm/grlm_cl_exp_assets}
RAM_MODEL_DIR=${RAM_MODEL_DIR:-${RAM_ASSET_ROOT}/models}
RAM_DATA_DIR=${RAM_DATA_DIR:-${RAM_ASSET_ROOT}/data}

log_ram_stage() {
    echo "[$(date -Is)] $*"
}

copy_tree_once() {
    local src=$1
    local dst=$2
    local label=$3
    local complete="${dst}/.ram_stage_complete"

    if [ "${REFRESH_RAM_ASSETS:-0}" = "1" ] && [ -d "$dst" ]; then
        log_ram_stage "Refreshing staged ${label}: ${dst}"
        rm -rf "$dst"
    fi

    if [ -f "$complete" ]; then
        log_ram_stage "${label} already staged in RAM: ${dst}"
        return
    fi

    if [ ! -d "$src" ]; then
        echo "ERROR: source ${label} directory not found: ${src}" >&2
        return 1
    fi

    log_ram_stage "Staging ${label} into RAM: ${src} -> ${dst}"
    rm -rf "$dst"
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$src"/ "$dst"/
    else
        cp -a "$src"/. "$dst"/
    fi
    touch "$complete"
    log_ram_stage "Finished staging ${label}: ${dst}"
}

mkdir -p "$RAM_ASSET_ROOT"
df -h "$RAM_ASSET_ROOT" || true

copy_tree_once "$SOURCE_MODEL_DIR" "$RAM_MODEL_DIR" "models" || return 1 2>/dev/null || exit 1
copy_tree_once "$SOURCE_DATA_DIR" "$RAM_DATA_DIR" "data" || return 1 2>/dev/null || exit 1

export MODEL_DIR="$RAM_MODEL_DIR"
export DATA_DIR="$RAM_DATA_DIR"

if [ -n "${LLAMA_DIR:-}" ] && [ -d "${LLAMA_DIR}/data" ] && [ -d "${DATA_DIR}/cl_sft" ]; then
    ln -sfn "${DATA_DIR}/cl_sft" "${LLAMA_DIR}/data/grlm_in_domain"
fi

log_ram_stage "Using RAM assets: MODEL_DIR=${MODEL_DIR}, DATA_DIR=${DATA_DIR}"
