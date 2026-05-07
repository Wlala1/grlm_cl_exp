#!/bin/bash
# Setup script: download models, prepare data, install dependencies.
# Run this ONCE before starting experiments.
set -e

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${WORK_DIR}/models
DATA_DIR=${WORK_DIR}/data
LLAMA_DIR=${WORK_DIR}/LlamaFactory

echo "=== GRLM CL Experiment Setup ==="
echo "Working directory: $WORK_DIR"

# ============================================================
# Step 1: Install LlamaFactory (if not already present)
# ============================================================
if [ ! -d "$LLAMA_DIR/src" ]; then
    echo "[Step 1] Cloning LlamaFactory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git $LLAMA_DIR
    cd $LLAMA_DIR && pip install -e ".[torch,metrics]" && cd $WORK_DIR
else
    echo "[Step 1] LlamaFactory already exists, skipping."
fi

# ============================================================
# Step 2: Download Qwen3 models from HuggingFace
# ============================================================
echo "[Step 2] Downloading Qwen3 models..."
mkdir -p $MODEL_DIR

# Qwen3-0.6B
if [ ! -d "$MODEL_DIR/Qwen3-0.6B" ]; then
    echo "  Downloading Qwen3-0.6B..."
    huggingface-cli download Qwen/Qwen3-0.6B --local-dir $MODEL_DIR/Qwen3-0.6B
else
    echo "  Qwen3-0.6B already exists."
fi

# Qwen3-1.7B
if [ ! -d "$MODEL_DIR/Qwen3-1.7B" ]; then
    echo "  Downloading Qwen3-1.7B..."
    huggingface-cli download Qwen/Qwen3-1.7B --local-dir $MODEL_DIR/Qwen3-1.7B
else
    echo "  Qwen3-1.7B already exists."
fi

# Qwen3-4B (Instruct version)
if [ ! -d "$MODEL_DIR/Qwen3-4B" ]; then
    echo "  Downloading Qwen3-4B..."
    huggingface-cli download Qwen/Qwen3-4B --local-dir $MODEL_DIR/Qwen3-4B
else
    echo "  Qwen3-4B already exists."
fi

# ============================================================
# Step 3: Download/prepare data
# ============================================================
echo "[Step 3] Preparing data..."
mkdir -p $DATA_DIR/cl_sft

# Option A: Copy from shared storage (uncomment and set SRC if on same cluster)
# SRC=/workspace/jiangzhuosong/GRLM_0
# cp $SRC/in_domain/books/sum_data/books_id2meta.json $DATA_DIR/
# cp $SRC/in_domain/books/sum_data/item_id2tid/books_tid2item_id.json $DATA_DIR/
# cp $SRC/LlamaFactory/data/grlm_in_domain/amazon_books_cl_*.json $DATA_DIR/cl_sft/

# Option B: Download from HuggingFace dataset
if [ ! -f "$DATA_DIR/books_id2meta.json" ]; then
    echo "  Downloading from HuggingFace..."
    huggingface-cli download JazyJiang/grlm-books-cl-data --local-dir $DATA_DIR --repo-type dataset
fi

# Check if data exists
if [ ! -f "$DATA_DIR/books_id2meta.json" ]; then
    echo ""
    echo "ERROR: Data not found at $DATA_DIR/"
    echo "Please ensure you have access to the HuggingFace dataset:"
    echo "  huggingface-cli download JazyJiang/grlm-books-cl-data --local-dir $DATA_DIR --repo-type dataset"
    echo ""
    exit 1
fi

# ============================================================
# Step 4: Setup dataset_info.json for LlamaFactory
# ============================================================
echo "[Step 4] Generating dataset_info.json..."
python3 ${WORK_DIR}/scripts/generate_dataset_info.py \
    --data_dir $DATA_DIR/cl_sft \
    --output $LLAMA_DIR/data/dataset_info.json

# ============================================================
# Step 5: Create symlinks and copy scripts
# ============================================================
echo "[Step 5] Setting up LlamaFactory integration..."
mkdir -p ${WORK_DIR}/results/cl_results_seq
mkdir -p ${WORK_DIR}/checkpoints

# Link data into LlamaFactory expected locations
ln -sfn $DATA_DIR/cl_sft $LLAMA_DIR/data/grlm_in_domain 2>/dev/null || true

# Copy run script into LlamaFactory (it auto-detects paths relative to itself)
cp ${WORK_DIR}/run_books_cl_v2.sh $LLAMA_DIR/run_books_cl_v2.sh

echo ""
echo "=== Setup Complete ==="
echo "To run a single chain:  cd LlamaFactory && bash run_books_cl_v2.sh 06b h10 1"
echo "To run all chains:      bash dispatch_all.sh"
