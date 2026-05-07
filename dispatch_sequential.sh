#!/bin/bash
# Alternative dispatcher: run chains sequentially per GPU.
# Each GPU handles one cap value across all model sizes.
# Total time: ~24-36h per GPU (4-6h for 0.6B + 6-10h for 1.7B + 10-16h for 4B)
# Usage: bash dispatch_sequential.sh

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
LLAMA_DIR=${WORK_DIR}/LlamaFactory
LOG_DIR=${WORK_DIR}/logs
mkdir -p $LOG_DIR

cd $LLAMA_DIR

run_chain() {
    local gpu=$1
    local cap=$2
    echo "[GPU ${gpu}] Starting 06b ${cap}..."
    bash run_books_cl_v2.sh 06b ${cap} ${gpu}
    echo "[GPU ${gpu}] Starting 17b ${cap}..."
    bash run_books_cl_v2.sh 17b ${cap} ${gpu}
    echo "[GPU ${gpu}] Done with cap=${cap}"
}

# GPUs 1-7, one cap per GPU (0.6B + 1.7B only, single GPU models)
CAPS=(h2 h5 h10 h20 h30 h40 full)

for i in "${!CAPS[@]}"; do
    gpu=$((i + 1))
    cap=${CAPS[$i]}
    echo "Launching GPU ${gpu} → cap=${cap} (06b + 17b)"
    nohup bash -c "cd ${LLAMA_DIR} && bash run_books_cl_v2.sh 06b ${cap} ${gpu} && bash run_books_cl_v2.sh 17b ${cap} ${gpu}" \
        > ${LOG_DIR}/gpu${gpu}_${cap}.log 2>&1 &
done

echo ""
echo "=== Launched 7 GPU workers (0.6B + 1.7B) ==="
echo "Each GPU handles: 0.6B then 1.7B for one cap value."
echo "Estimated time: ~10-16h per GPU."
echo ""
echo "After these finish, run 4B chains (2 GPUs each):"
echo "  bash run_books_cl_v2.sh 4b h2 1,2"
echo "  bash run_books_cl_v2.sh 4b h5 3,4"
echo "  bash run_books_cl_v2.sh 4b h10 5,6"
echo "  (then h20, h30, h40, full in next batch)"
