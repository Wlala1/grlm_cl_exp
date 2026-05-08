#!/bin/bash
# Prepare all immutable assets inside the Docker image.
set -Eeuo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR=${MODEL_DIR:-${WORK_DIR}/models}
DATA_DIR=${DATA_DIR:-${WORK_DIR}/data}
LLAMA_DIR=${LLAMA_DIR:-${WORK_DIR}/LlamaFactory}
LLAMAFACTORY_REPO=${LLAMAFACTORY_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}
LLAMAFACTORY_REF=${LLAMAFACTORY_REF:-}

if [ -f /run/secrets/hf_token ]; then
    export HF_TOKEN
    HF_TOKEN="$(cat /run/secrets/hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

log() {
    echo "[$(date -Is)] $*"
}

download_model() {
    local repo_id=$1
    local output_dir=$2
    if [ -f "${output_dir}/config.json" ]; then
        log "${repo_id} already exists at ${output_dir}"
        return
    fi
    log "Downloading ${repo_id} -> ${output_dir}"
    mkdir -p "$output_dir"
    hf_download "$repo_id" "$output_dir"
}

hf_download() {
    local repo_id=$1
    local output_dir=$2
    shift 2

    if command -v hf >/dev/null 2>&1; then
        hf download "$repo_id" --local-dir "$output_dir" "$@"
    else
        huggingface-cli download "$repo_id" --local-dir "$output_dir" "$@"
    fi
}

log "=== Preparing GRLM CL image assets ==="
log "Working directory: $WORK_DIR"

log "[Step 0] Installing Python build/download helpers"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade "huggingface_hub[cli]"

log "[Step 1] Installing LlamaFactory"
if [ ! -d "$LLAMA_DIR/src" ]; then
    git clone --depth 1 "$LLAMAFACTORY_REPO" "$LLAMA_DIR"
    if [ -n "$LLAMAFACTORY_REF" ]; then
        cd "$LLAMA_DIR"
        git fetch --depth 1 origin "$LLAMAFACTORY_REF"
        git checkout FETCH_HEAD
        cd "$WORK_DIR"
    fi
else
    log "LlamaFactory already exists, skipping clone"
fi
cd "$LLAMA_DIR"
python3 -m pip install -e ".[torch,metrics]"
python3 -m pip install --upgrade deepspeed setproctitle
cd "$WORK_DIR"

log "[Step 2] Downloading Qwen3 models"
mkdir -p "$MODEL_DIR"
download_model Qwen/Qwen3-0.6B "${MODEL_DIR}/Qwen3-0.6B"
download_model Qwen/Qwen3-1.7B "${MODEL_DIR}/Qwen3-1.7B"
download_model Qwen/Qwen3-4B "${MODEL_DIR}/Qwen3-4B"

log "[Step 3] Preparing Books CL data"
mkdir -p "${DATA_DIR}/cl_sft"
if [ ! -f "${DATA_DIR}/books_id2meta.json" ]; then
    hf_download JazySong/grlm-books-cl-data "$DATA_DIR" --repo-type dataset
fi

if [ -f "${DATA_DIR}/cl_sft.tar.gz" ]; then
    tar xzf "${DATA_DIR}/cl_sft.tar.gz" -C "${DATA_DIR}/cl_sft"
    rm -f "${DATA_DIR}/cl_sft.tar.gz"
fi

if [ -f "${DATA_DIR}/cl_sft/amazon_books_cl_D0_train.json" ]; then
    for cap in h2 h5 h10 h20 h30 h40; do
        ln -sf amazon_books_cl_D0_train.json "${DATA_DIR}/cl_sft/amazon_books_cl_D0_train_${cap}.json"
    done
fi

if [ ! -f "${DATA_DIR}/books_id2meta.json" ]; then
    echo "ERROR: Data not found at ${DATA_DIR}/books_id2meta.json"
    exit 1
fi
if [ ! -f "${DATA_DIR}/books_tid2item_id.json" ]; then
    echo "ERROR: Data not found at ${DATA_DIR}/books_tid2item_id.json"
    exit 1
fi

log "[Step 4] Generating LlamaFactory dataset_info.json"
python3 "${WORK_DIR}/scripts/generate_dataset_info.py" \
    --data_dir "${DATA_DIR}/cl_sft" \
    --output "${LLAMA_DIR}/data/dataset_info.json"

log "[Step 5] Linking data into LlamaFactory"
ln -sfn "${DATA_DIR}/cl_sft" "${LLAMA_DIR}/data/grlm_in_domain"

log "=== Image assets ready ==="
