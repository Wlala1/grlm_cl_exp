#!/bin/bash
# Dispatch all 21 CL chains (3 models × 7 caps) across GPUs 1-7.
# Each chain runs ~8-12 hours on H100 80GB.
# Usage: bash dispatch_all.sh

WORK_DIR=$(cd "$(dirname "$0")" && pwd)
LLAMA_DIR=${WORK_DIR}/LlamaFactory
LOG_DIR=${WORK_DIR}/logs
mkdir -p $LOG_DIR

echo "=== Dispatching all CL chains ==="
echo "Logs: $LOG_DIR/"
echo ""

cd $LLAMA_DIR

# ============================================================
# GPU Assignment (7 GPUs available: 1-7)
# 0.6B: 1 GPU each (7 chains → rotate GPUs)
# 1.7B: 1 GPU each (7 chains → rotate GPUs)
# 4B:   2 GPUs each (7 chains → 2 at a time)
# ============================================================

# --- 0.6B chains (7 caps, 1 GPU each) ---
# Run on GPUs 1-7, all in parallel
CAPS_06B=(h2 h5 h10 h20 h30 h40 full)
GPU_06B=(1 2 3 4 5 6 7)

for i in "${!CAPS_06B[@]}"; do
    cap=${CAPS_06B[$i]}
    gpu=${GPU_06B[$i]}
    echo "[0.6B ${cap}] → GPU ${gpu}"
    nohup bash run_books_cl_v2.sh 06b ${cap} ${gpu} \
        > ${LOG_DIR}/06b_${cap}.log 2>&1 &
done

echo ""
echo "=== 0.6B chains launched (7 chains on GPUs 1-7) ==="
echo "Monitor: tail -f ${LOG_DIR}/06b_*.log"
echo ""
echo "Once 0.6B chains finish (~4-6h), run the following for 1.7B:"
echo ""

cat << 'NEXT_BATCH'
# --- 1.7B chains (run after 0.6B finishes) ---
CAPS_17B=(h2 h5 h10 h20 h30 h40 full)
GPU_17B=(1 2 3 4 5 6 7)

for i in "${!CAPS_17B[@]}"; do
    cap=${CAPS_17B[$i]}
    gpu=${GPU_17B[$i]}
    echo "[1.7B ${cap}] → GPU ${gpu}"
    nohup bash run_books_cl_v2.sh 17b ${cap} ${gpu} \
        > logs/17b_${cap}.log 2>&1 &
done

# --- 4B chains (run after 1.7B finishes, uses 2 GPUs each) ---
# Can run 3 chains simultaneously (GPUs 1-2, 3-4, 5-6)
CAPS_4B_BATCH1=(h2 h5 h10)
GPU_4B_BATCH1=("1,2" "3,4" "5,6")

for i in "${!CAPS_4B_BATCH1[@]}"; do
    cap=${CAPS_4B_BATCH1[$i]}
    gpu=${GPU_4B_BATCH1[$i]}
    echo "[4B ${cap}] → GPUs ${gpu}"
    nohup bash run_books_cl_v2.sh 4b ${cap} ${gpu} \
        > logs/4b_${cap}.log 2>&1 &
done

# After first 4B batch finishes:
CAPS_4B_BATCH2=(h20 h30 h40 full)
GPU_4B_BATCH2=("1,2" "3,4" "5,6" "1,2")  # last one waits for first to finish

for i in "${!CAPS_4B_BATCH2[@]}"; do
    cap=${CAPS_4B_BATCH2[$i]}
    gpu=${GPU_4B_BATCH2[$i]}
    echo "[4B ${cap}] → GPUs ${gpu}"
    nohup bash run_books_cl_v2.sh 4b ${cap} ${gpu} \
        > logs/4b_${cap}.log 2>&1 &
done
NEXT_BATCH

echo ""
echo "=== Alternative: Run everything sequentially per GPU ==="
echo "If you prefer to let each GPU run all sizes one after another,"
echo "use: bash dispatch_sequential.sh"
