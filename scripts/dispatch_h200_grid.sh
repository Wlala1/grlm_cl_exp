#!/bin/bash
# Dispatch the full 3x7 Books CL grid on an 8-GPU H200 node.
set -Eeuo pipefail

WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT=${RUN_ROOT:-/runs}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
DISPATCH_LOG=${DISPATCH_LOG:-${LOG_ROOT}/dispatch_h200_grid.log}
RUNNER=${RUNNER:-${WORK_DIR}/run_books_cl_v2.sh}

mkdir -p "$LOG_ROOT"

if [ "${DISPATCH_LOG_REDIRECTED:-0}" != "1" ]; then
    export DISPATCH_LOG_REDIRECTED=1
    exec > >(tee -a "$DISPATCH_LOG") 2>&1
fi

log() {
    echo "[$(date -Is)] $*"
}

PIDS=()

shutdown_dispatch() {
    local sig=$1
    local code=$2
    local pid
    log "Received $sig; forwarding to running workers"
    for pid in "${PIDS[@]:-}"; do
        kill -s "$sig" "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$code"
}

trap 'shutdown_dispatch TERM 143' TERM
trap 'shutdown_dispatch INT 130' INT

run_chain() {
    local model=$1
    local cap=$2
    local gpu_ids=$3
    local chain_log="${LOG_ROOT}/${model}_${cap}.log"
    local result_dir="${RUN_ROOT}/results/cl_results_seq/${model}_${cap}"
    local hist_tag="$cap"
    local period
    if [ "$cap" = "full" ]; then
        hist_tag="hfull"
    fi

    local complete=1
    for period in 0 1 2 3; do
        local recall_file="${result_dir}/seq_recall_${hist_tag}_D${period}.json"
        local results_file="${result_dir}/seq_results_${hist_tag}_D${period}.jsonl"
        if ! python3 -m json.tool "$recall_file" >/dev/null 2>&1 || [ ! -s "$results_file" ]; then
            complete=0
            break
        fi
    done

    if [ "$complete" -eq 1 ]; then
        log "SKIP  ${model} ${cap} GPUs=${gpu_ids} existing results complete"
        python3 "${WORK_DIR}/scripts/collect_cross_scale_table.py" --results-root "${RUN_ROOT}/results" >> "$chain_log" 2>&1 || return 1
        return 0
    fi

    log "START ${model} ${cap} GPUs=${gpu_ids} log=${chain_log}"
    local chain_pid=""
    forward_chain_stop() {
        local sig=$1
        if [ -n "$chain_pid" ] && kill -0 "$chain_pid" 2>/dev/null; then
            log "Received $sig; forwarding to ${model} ${cap} runner $chain_pid"
            kill -s "$sig" "$chain_pid" 2>/dev/null || true
        fi
    }

    trap 'forward_chain_stop TERM' TERM
    trap 'forward_chain_stop INT' INT

    RUN_ROOT="$RUN_ROOT" bash "$RUNNER" "$model" "$cap" "$gpu_ids" >/dev/null 2>&1 &
    chain_pid=$!
    set +e
    wait "$chain_pid"
    local exit_code=$?
    set -e
    trap - TERM INT

    if [ "$exit_code" -eq 0 ]; then
        log "DONE  ${model} ${cap} GPUs=${gpu_ids}"
    else
        log "FAIL  ${model} ${cap} GPUs=${gpu_ids} exit=${exit_code} log=${chain_log}"
    fi
    return "$exit_code"
}

worker() {
    local gpu_ids=$1
    shift
    local failed=0
    local spec model cap

    for spec in "$@"; do
        set -- $spec
        model=$1
        cap=$2
        if ! run_chain "$model" "$cap" "$gpu_ids"; then
            failed=1
        fi
    done
    return "$failed"
}

print_plan() {
    cat <<'PLAN'
Stage 1 single-GPU workers:
  GPU 0: 06b h2   -> 17b h5
  GPU 1: 06b h5   -> 17b h10
  GPU 2: 06b h10  -> 17b h20
  GPU 3: 06b h20  -> 17b h30
  GPU 4: 06b h30  -> 17b h40
  GPU 5: 06b h40  -> 17b full
  GPU 6: 06b full
  GPU 7: 17b h2

Stage 2 two-GPU 4B workers:
  GPUs 0,1: 4b h2  -> 4b h30
  GPUs 2,3: 4b h5  -> 4b h40
  GPUs 4,5: 4b h10 -> 4b full
  GPUs 6,7: 4b h20
PLAN
}

wait_stage() {
    local stage_name=$1
    shift
    local failed=0
    local pid

    for pid in "$@"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [ "$failed" -eq 0 ]; then
        log "${stage_name} complete"
    else
        log "${stage_name} complete with failures"
    fi
    return "$failed"
}

log "=== Dispatching full Books CL grid ==="
log "Run root: $RUN_ROOT"
log "Dispatch log: $DISPATCH_LOG"
print_plan

if [ "${DRY_RUN:-0}" = "1" ]; then
    log "DRY_RUN=1; exiting without launching jobs"
    exit 0
fi

if [ "${USE_RAM_ASSETS:-0}" = "1" ]; then
    # Exports MODEL_DIR and DATA_DIR for all chain workers below.
    source "${WORK_DIR}/scripts/stage_assets_to_ram.sh" || exit 1
fi

TOTAL_FAILED=0

log "=== Stage 1: 0.6B + 1.7B single-GPU chains ==="
PIDS=()
worker 0 "06b h2" "17b h5" &
PIDS+=("$!")
worker 1 "06b h5" "17b h10" &
PIDS+=("$!")
worker 2 "06b h10" "17b h20" &
PIDS+=("$!")
worker 3 "06b h20" "17b h30" &
PIDS+=("$!")
worker 4 "06b h30" "17b h40" &
PIDS+=("$!")
worker 5 "06b h40" "17b full" &
PIDS+=("$!")
worker 6 "06b full" &
PIDS+=("$!")
worker 7 "17b h2" &
PIDS+=("$!")

if ! wait_stage "Stage 1" "${PIDS[@]}"; then
    TOTAL_FAILED=1
fi

log "=== Stage 2: 4B two-GPU chains ==="
PIDS=()
worker "0,1" "4b h2" "4b h30" &
PIDS+=("$!")
worker "2,3" "4b h5" "4b h40" &
PIDS+=("$!")
worker "4,5" "4b h10" "4b full" &
PIDS+=("$!")
worker "6,7" "4b h20" &
PIDS+=("$!")

if ! wait_stage "Stage 2" "${PIDS[@]}"; then
    TOTAL_FAILED=1
fi

python3 "${WORK_DIR}/scripts/collect_cross_scale_table.py" --results-root "${RUN_ROOT}/results" || TOTAL_FAILED=1

if [ "$TOTAL_FAILED" -ne 0 ]; then
    log "=== Dispatch complete with failures. Re-run bash scripts/dispatch_h200_grid.sh to resume failed/incomplete periods. ==="
    exit 1
fi

log "=== Dispatch complete successfully ==="
