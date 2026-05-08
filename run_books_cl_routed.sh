#!/bin/bash
# Books CL routed chain runner with period-level resume and per-eval collection.
# Usage: bash run_books_cl_routed.sh <model_size> <cap> <gpu_ids> [sliding_window] [aux_weight]
set -Eeuo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "Usage: bash run_books_cl_routed.sh <model_size> <cap> <gpu_ids> [sliding_window] [aux_weight]"
    echo "  model_size: 06b, 17b, or 4b"
    echo "  cap: h2, h5, h10, h20, h30, h40, or full"
    echo "  gpu_ids: e.g. 0 or 0,1"
    echo "  sliding_window: token count, default 512"
    echo "  aux_weight: aux loss coefficient, default 0.1"
    exit 2
fi

MODEL_SIZE=$1
CAP=$2
GPU_IDS=$3
SW=${4:-512}
AUX_W=${5:-0.1}
AUX_TAG=${AUX_W//./p}
RUN_ID="${MODEL_SIZE}_${CAP}_sw${SW}_aux${AUX_TAG}"

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_ROOT=${RUN_ROOT:-/runs}
LLAMA_DIR=${LLAMA_DIR:-${WORK_DIR}/LlamaFactory}
DATA_DIR=${DATA_DIR:-${WORK_DIR}/data}
MODEL_DIR=${MODEL_DIR:-${WORK_DIR}/models}
RESULTS_ROOT=${RESULTS_ROOT:-${RUN_ROOT}/results}
RESULT_DIR=${RESULT_DIR:-${RESULTS_ROOT}/cl_results_seq/${RUN_ID}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/${RUN_ID}}
LOG_ROOT=${LOG_ROOT:-${RUN_ROOT}/logs}
CHAIN_LOG_DIR=${CHAIN_LOG_DIR:-${LOG_ROOT}/${RUN_ID}}
CHAIN_LOG=${CHAIN_LOG:-${LOG_ROOT}/${RUN_ID}.log}
STATE_DIR=${STATE_DIR:-${RUN_ROOT}/state}
STATE_FILE=${STATE_FILE:-${STATE_DIR}/${RUN_ID}.json}

EVAL_SCRIPT=${EVAL_SCRIPT:-${WORK_DIR}/eval/s5_books_cl_eval_seq.py}
COLLECT_SCRIPT=${COLLECT_SCRIPT:-${WORK_DIR}/scripts/collect_cross_scale_table.py}
STATE_SCRIPT=${STATE_SCRIPT:-${WORK_DIR}/scripts/chain_state.py}
TID2ITEM=${TID2ITEM:-${DATA_DIR}/books_tid2item_id.json}
ID2META=${ID2META:-${DATA_DIR}/books_id2meta.json}
EVAL_DIR=${EVAL_DIR:-${DATA_DIR}/cl_sft}
MAX_USERS=${MAX_USERS:-5000}

mkdir -p "$RESULT_DIR" "$CHECKPOINT_ROOT" "$CHAIN_LOG_DIR" "$STATE_DIR" "$LOG_ROOT"

if [ "${CHAIN_LOG_REDIRECTED:-0}" != "1" ]; then
    export CHAIN_LOG_REDIRECTED=1
    exec > >(tee -a "$CHAIN_LOG") 2>&1
fi

log() {
    echo "[$(date -Is)] $*"
}

fail() {
    log "ERROR: $*"
    python3 "$STATE_SCRIPT" finish --state-file "$STATE_FILE" --status failed 2>/dev/null || true
    exit 1
}

json_valid() {
    local path=$1
    [ -s "$path" ] && python3 -m json.tool "$path" >/dev/null 2>&1
}

eval_complete() {
    local recall_file=$1
    local results_file=$2
    json_valid "$recall_file" && [ -s "$results_file" ]
}

future_period_incomplete() {
    local period=$1
    local future
    for future in 1 2 3; do
        if [ "$future" -le "$period" ]; then
            continue
        fi
        if ! eval_complete \
            "${RESULT_DIR}/seq_recall_${HIST_TAG}_D${future}.json" \
            "${RESULT_DIR}/seq_results_${HIST_TAG}_D${future}.jsonl"; then
            return 0
        fi
    done
    return 1
}

checkpoint_ready() {
    local path=$1
    [ -d "$path" ] \
        && [ -f "$path/config.json" ] \
        && { compgen -G "${path}/*.safetensors" >/dev/null || compgen -G "${path}/pytorch_model*.bin" >/dev/null; }
}

latest_trainer_checkpoint() {
    local path=$1
    [ -d "$path" ] || return 0
    python3 - "$path" <<'PY'
import os
import re
import sys

root = sys.argv[1]
best = None
for name in os.listdir(root):
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match is None:
        continue
    path = os.path.join(root, name)
    if not os.path.isdir(path):
        continue
    if not os.path.isfile(os.path.join(path, "trainer_state.json")):
        continue
    candidate = (int(match.group(1)), path)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is not None:
    print(best[1])
PY
}

stage_status() {
    local period=$1
    local stage=$2
    python3 "$STATE_SCRIPT" status --state-file "$STATE_FILE" --period "D${period}" --stage "$stage"
}

mark_stage() {
    local period=$1
    local stage=$2
    local status=$3
    shift 3
    python3 "$STATE_SCRIPT" mark \
        --state-file "$STATE_FILE" \
        --period "D${period}" \
        --stage "$stage" \
        --status "$status" \
        "$@"
}

if [ "$MODEL_SIZE" == "06b" ]; then
    MODEL_PATH=${MODEL_DIR}/Qwen3-0.6B
    LR_INIT=7e-5
    LR_FT=3e-5
    NUM_GPUS=1
    BS=32
    GA=2
elif [ "$MODEL_SIZE" == "17b" ]; then
    MODEL_PATH=${MODEL_DIR}/Qwen3-1.7B
    LR_INIT=5e-5
    LR_FT=2e-5
    NUM_GPUS=1
    BS=32
    GA=2
elif [ "$MODEL_SIZE" == "4b" ]; then
    MODEL_PATH=${MODEL_DIR}/Qwen3-4B
    LR_INIT=1e-4
    LR_FT=5e-5
    NUM_GPUS=2
    BS=32
    GA=1
else
    fail "Unknown model size: $MODEL_SIZE (use 06b, 17b, or 4b)"
fi

BS_VAR="BS_${MODEL_SIZE}"
GA_VAR="GA_${MODEL_SIZE}"
BS=${!BS_VAR:-${BS}}
GA=${!GA_VAR:-${GA}}
BS=${BS_OVERRIDE:-${BS}}
GA=${GA_OVERRIDE:-${GA}}
EFFECTIVE_TRAIN_BATCH=$((BS * GA * NUM_GPUS))

case "$CAP" in
    h2|h5|h10|h20|h30|h40|full) ;;
    *) fail "Unknown cap: $CAP (use h2, h5, h10, h20, h30, h40, or full)" ;;
esac

if [ "$CAP" == "full" ]; then
    DATASET_SUFFIX=""
    HIST_TAG="hfull"
    CAP_NUM=""
else
    DATASET_SUFFIX="_${CAP}"
    HIST_TAG="$CAP"
    CAP_NUM=${CAP#h}
fi

python3 "$STATE_SCRIPT" init \
    --state-file "$STATE_FILE" \
    --model-size "${MODEL_SIZE}_routed" \
    --cap "$CAP" \
    --gpu-ids "$GPU_IDS" \
    --run-root "$RUN_ROOT" \
    --result-dir "$RESULT_DIR" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --log-dir "$CHAIN_LOG_DIR" \
    --chain-log-path "$CHAIN_LOG"

log "=== routed ${MODEL_SIZE} cap=${CAP} sw=${SW} aux=${AUX_W} GPUs=${GPU_IDS} ==="
log "Run root: $RUN_ROOT"
log "Results: $RESULT_DIR"
log "State: $STATE_FILE"
log "Chain log: $CHAIN_LOG"
log "Model dir: $MODEL_DIR"
log "Data dir: $DATA_DIR"
log "Train batch: per_device_train_batch_size=$BS gradient_accumulation_steps=$GA effective_train_batch_size=$EFFECTIVE_TRAIN_BATCH"

if [ -d "$LLAMA_DIR/data" ] && [ -d "$DATA_DIR/cl_sft" ]; then
    ln -sfn "$DATA_DIR/cl_sft" "$LLAMA_DIR/data/grlm_in_domain"
fi

if [ "${SMOKE:-0}" = "1" ]; then
    MAX_USERS=${SMOKE_MAX_USERS:-20}
    SMOKE_TRAIN_ARGS=(--max_steps "${SMOKE_MAX_STEPS:-2}")
    log "SMOKE=1: max_steps=${SMOKE_MAX_STEPS:-2}, max_users=$MAX_USERS"
else
    SMOKE_TRAIN_ARGS=()
fi

SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
SAVE_ARGS=(--save_strategy "$SAVE_STRATEGY" --save_only_model false)
if [ -n "${SAVE_STEPS:-}" ]; then
    SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi
if [ "$SAVE_TOTAL_LIMIT" != "0" ]; then
    SAVE_ARGS+=(--save_total_limit "$SAVE_TOTAL_LIMIT")
fi
if [ "$SAVE_TOTAL_LIMIT" = "0" ]; then
    SAVE_TOTAL_LIMIT_LABEL="unlimited"
else
    SAVE_TOTAL_LIMIT_LABEL="$SAVE_TOTAL_LIMIT"
fi
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-1}
log "Checkpointing: save_strategy=$SAVE_STRATEGY save_total_limit=$SAVE_TOTAL_LIMIT_LABEL save_only_model=false keep_checkpoints=$KEEP_CHECKPOINTS"

eval_batch_size() {
    if [ "$CAP" == "full" ]; then
        if [ "$MODEL_SIZE" == "06b" ]; then echo 16
        elif [ "$MODEL_SIZE" == "17b" ]; then echo 8
        else echo 4; fi
    elif [ "$CAP_NUM" -le 10 ]; then
        if [ "$MODEL_SIZE" == "06b" ]; then echo 32
        elif [ "$MODEL_SIZE" == "17b" ]; then echo 16
        else echo 8; fi
    else
        if [ "$MODEL_SIZE" == "06b" ]; then echo 20
        elif [ "$MODEL_SIZE" == "17b" ]; then echo 10
        else echo 6; fi
    fi
}

ACTIVE_TRAIN_PID=""
STOP_REQUESTED=0

forward_train_stop() {
    local sig=$1
    STOP_REQUESTED=1
    if [ -n "$ACTIVE_TRAIN_PID" ] && kill -0 "$ACTIVE_TRAIN_PID" 2>/dev/null; then
        log "Received $sig; forwarding to training process $ACTIVE_TRAIN_PID"
        kill -s "$sig" "$ACTIVE_TRAIN_PID" 2>/dev/null || true
    fi
}

wait_for_active_train() {
    local pid=$1
    local exit_code=0
    ACTIVE_TRAIN_PID="$pid"
    STOP_REQUESTED=0
    trap 'forward_train_stop TERM' TERM
    trap 'forward_train_stop INT' INT

    set +e
    wait "$pid"
    exit_code=$?
    if [ "$STOP_REQUESTED" -eq 1 ] && kill -0 "$pid" 2>/dev/null; then
        log "Waiting for training process $pid to save its shutdown checkpoint"
        wait "$pid"
        exit_code=$?
    fi
    set -e

    trap - TERM INT
    ACTIVE_TRAIN_PID=""
    return "$exit_code"
}

run_train() {
    local period=$1
    local init_model=$2
    local lr=$3
    local epochs=$4
    local output_dir=$5
    local train_log=$6
    local dataset_name="grlm_indomain_books_cl_D${period}${DATASET_SUFFIX}"
    local resume_checkpoint
    resume_checkpoint=$(latest_trainer_checkpoint "$output_dir")

    local train_args=(
        --model_name_or_path "$init_model"
        --do_train
        --dataset "$dataset_name"
        --dataset_dir "${LLAMA_DIR}/data"
        --template qwen3_nothink
        --finetuning_type full
        --output_dir "$output_dir"
        --overwrite_cache
        --per_device_train_batch_size "$BS"
        --gradient_accumulation_steps "$GA"
        --lr_scheduler_type cosine
        --logging_steps 10
        --learning_rate "$lr"
        --num_train_epochs "$epochs"
        --plot_loss
        --bf16
        --report_to none
        --sliding_window "$SW"
        --aux_loss_weight "$AUX_W"
    )
    train_args+=("${SAVE_ARGS[@]}")
    if [ -n "$resume_checkpoint" ]; then
        train_args+=(--resume_from_checkpoint "$resume_checkpoint")
    fi
    train_args+=("${SMOKE_TRAIN_ARGS[@]}")

    log "Training routed D${period} from $init_model"
    if [ -n "$resume_checkpoint" ]; then
        log "Resuming routed D${period} from trainer checkpoint: $resume_checkpoint"
    fi
    log "Train log: $train_log"
    mark_stage "$period" train running --checkpoint-path "$output_dir" --log-path "$train_log"

    {
        echo "[$(date -Is)] train routed D${period} ${MODEL_SIZE} ${CAP}"
        echo "dataset=$dataset_name"
        echo "init_model=$init_model"
        echo "output_dir=$output_dir"
        echo "resume_checkpoint=${resume_checkpoint:-}"
        echo "sliding_window=$SW"
        echo "aux_loss_weight=$AUX_W"
    } >> "$train_log"

    local exit_code=0
    GRLM_GPU_IDS="$GPU_IDS" PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}" WANDB_DISABLED=true DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        python3 "$WORK_DIR/scripts/named_train.py" -m routing.train_with_routing "${train_args[@]}" >> "$train_log" 2>&1 &
    wait_for_active_train "$!" || exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        mark_stage "$period" train failed --exit-code "$exit_code" --checkpoint-path "$output_dir" --log-path "$train_log"
        fail "Training routed D${period} failed with exit code $exit_code. See $train_log"
    fi

    mark_stage "$period" train completed --exit-code 0 --checkpoint-path "$output_dir" --log-path "$train_log"
    log "Training routed D${period} complete"
}

run_eval() {
    local period=$1
    local output_dir=$2
    local eval_log=$3
    local recall_file=$4
    local results_file=$5
    local eval_file="${EVAL_DIR}/amazon_books_cl_D${period}_eval.json"
    local eval_bs
    eval_bs=$(eval_batch_size)

    log "Eval routed D${period}->D$((period + 1)) batch_size=$eval_bs"
    log "Eval log: $eval_log"
    mark_stage "$period" eval running \
        --checkpoint-path "$output_dir" \
        --log-path "$eval_log" \
        --result-recall-path "$recall_file" \
        --result-results-path "$results_file"

    local eval_args=(
        --model "$output_dir"
        --eval_file "$eval_file"
        --tid2item_id "$TID2ITEM"
        --id2meta "$ID2META"
        --num_gpus "$NUM_GPUS"
        --batch_size "$eval_bs"
        --max_users "$MAX_USERS"
        --output_dir "$RESULT_DIR"
        --period "$period"
        --model_size "${MODEL_SIZE}_routed"
        --cap "$CAP"
        --gpu_ids "$GPU_IDS"
    )
    if [ "$CAP" != "full" ]; then
        eval_args+=(--max_hist "$CAP_NUM")
    fi

    {
        echo "[$(date -Is)] eval routed D${period}->D$((period + 1)) ${MODEL_SIZE} ${CAP}"
        echo "model=$output_dir"
        echo "eval_file=$eval_file"
        echo "recall_file=$recall_file"
    } >> "$eval_log"

    local exit_code=0
    CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 "$EVAL_SCRIPT" "${eval_args[@]}" >> "$eval_log" 2>&1 || exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        mark_stage "$period" eval failed \
            --exit-code "$exit_code" \
            --checkpoint-path "$output_dir" \
            --log-path "$eval_log" \
            --result-recall-path "$recall_file" \
            --result-results-path "$results_file"
        fail "Eval routed D${period} failed with exit code $exit_code. See $eval_log"
    fi

    if ! json_valid "$recall_file"; then
        mark_stage "$period" eval failed \
            --exit-code 1 \
            --checkpoint-path "$output_dir" \
            --log-path "$eval_log" \
            --result-recall-path "$recall_file" \
            --result-results-path "$results_file"
        fail "Eval routed D${period} finished but recall JSON is missing or invalid: $recall_file"
    fi

    if [ ! -s "$results_file" ]; then
        mark_stage "$period" eval failed \
            --exit-code 1 \
            --checkpoint-path "$output_dir" \
            --log-path "$eval_log" \
            --result-recall-path "$recall_file" \
            --result-results-path "$results_file"
        fail "Eval routed D${period} finished but per-pair results are missing: $results_file"
    fi

    mark_stage "$period" eval completed \
        --exit-code 0 \
        --checkpoint-path "$output_dir" \
        --log-path "$eval_log" \
        --result-recall-path "$recall_file" \
        --result-results-path "$results_file"
    log "Eval routed D${period} complete"
}

collect_results() {
    local period=$1
    local collect_log="${CHAIN_LOG_DIR}/collect_D${period}.log"

    log "Collecting table after routed D${period} eval"
    mark_stage "$period" collect running --log-path "$collect_log"

    local exit_code=0
    python3 "$COLLECT_SCRIPT" --results-root "$RESULTS_ROOT" >> "$collect_log" 2>&1 || exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        mark_stage "$period" collect failed --exit-code "$exit_code" --log-path "$collect_log"
        fail "Collection after routed D${period} failed with exit code $exit_code. See $collect_log"
    fi

    mark_stage "$period" collect completed --exit-code 0 --log-path "$collect_log"
    log "Collection after routed D${period} complete"
}

PREV_CKPT=""

for PERIOD in 0 1 2 3; do
    OUTPUT_DIR="${CHECKPOINT_ROOT}/D${PERIOD}"
    TRAIN_LOG="${CHAIN_LOG_DIR}/train_D${PERIOD}.log"
    EVAL_LOG="${CHAIN_LOG_DIR}/eval_D${PERIOD}.log"
    RECALL_FILE="${RESULT_DIR}/seq_recall_${HIST_TAG}_D${PERIOD}.json"
    RESULTS_FILE="${RESULT_DIR}/seq_results_${HIST_TAG}_D${PERIOD}.jsonl"
    PERIOD_RESULT_COMPLETE=0

    if eval_complete "$RECALL_FILE" "$RESULTS_FILE"; then
        PERIOD_RESULT_COMPLETE=1
        if ! future_period_incomplete "$PERIOD" || checkpoint_ready "$OUTPUT_DIR"; then
            log "D${PERIOD} eval already complete: $RECALL_FILE"
            mark_stage "$PERIOD" eval completed \
                --exit-code 0 \
                --checkpoint-path "$OUTPUT_DIR" \
                --log-path "$EVAL_LOG" \
                --result-recall-path "$RECALL_FILE" \
                --result-results-path "$RESULTS_FILE"
            collect_results "$PERIOD"
            PREV_CKPT="$OUTPUT_DIR"
            continue
        fi
        log "D${PERIOD} eval exists, but checkpoint is missing and a future period is incomplete; retraining routed D${PERIOD}"
    fi

    TRAIN_STATUS=$(stage_status "$PERIOD" train)
    if [ "$TRAIN_STATUS" = "completed" ] && checkpoint_ready "$OUTPUT_DIR"; then
        log "D${PERIOD} train already complete; reusing checkpoint $OUTPUT_DIR"
    else
        if [ "$PERIOD" -eq 0 ]; then
            INIT_MODEL="$MODEL_PATH"
            LR="$LR_INIT"
            EPOCHS=5
        else
            INIT_MODEL="$PREV_CKPT"
            LR="$LR_FT"
            EPOCHS=3
            if ! checkpoint_ready "$INIT_MODEL"; then
                fail "Cannot start routed D${PERIOD}; previous checkpoint is missing or incomplete: $INIT_MODEL"
            fi
        fi

        RESUME_CKPT=$(latest_trainer_checkpoint "$OUTPUT_DIR")
        if [ -d "$OUTPUT_DIR" ] && [ -z "$RESUME_CKPT" ]; then
            log "Removing incomplete routed D${PERIOD} checkpoint before retrain: $OUTPUT_DIR"
            rm -rf "$OUTPUT_DIR"
        elif [ -n "$RESUME_CKPT" ]; then
            log "Found resumable routed D${PERIOD} trainer checkpoint: $RESUME_CKPT"
        fi
        mkdir -p "$OUTPUT_DIR"
        run_train "$PERIOD" "$INIT_MODEL" "$LR" "$EPOCHS" "$OUTPUT_DIR" "$TRAIN_LOG"
    fi

    if [ "$PERIOD_RESULT_COMPLETE" -eq 1 ]; then
        log "D${PERIOD} eval/result files already exist; keeping them and skipping routed eval"
        mark_stage "$PERIOD" eval completed \
            --exit-code 0 \
            --checkpoint-path "$OUTPUT_DIR" \
            --log-path "$EVAL_LOG" \
            --result-recall-path "$RECALL_FILE" \
            --result-results-path "$RESULTS_FILE"
        collect_results "$PERIOD"
        PREV_CKPT="$OUTPUT_DIR"
        log "=== routed D${PERIOD} checkpoint restored; existing eval preserved ==="
        continue
    fi

    run_eval "$PERIOD" "$OUTPUT_DIR" "$EVAL_LOG" "$RECALL_FILE" "$RESULTS_FILE"
    collect_results "$PERIOD"

    if [ "$KEEP_CHECKPOINTS" != "1" ] && [ "$PERIOD" -ge 2 ]; then
        OLD_CKPT="${CHECKPOINT_ROOT}/D$((PERIOD - 2))"
        if [ -d "$OLD_CKPT" ]; then
            log "Deleting old routed checkpoint: $OLD_CKPT"
            rm -rf "$OLD_CKPT"
        fi
    fi

    PREV_CKPT="$OUTPUT_DIR"
    log "=== routed D${PERIOD} complete ==="
done

if [ "$KEEP_CHECKPOINTS" = "1" ]; then
    log "All routed periods complete; retaining chain checkpoints: $CHECKPOINT_ROOT"
else
    log "All routed periods complete; deleting chain checkpoints: $CHECKPOINT_ROOT"
    rm -rf "$CHECKPOINT_ROOT"
fi
python3 "$STATE_SCRIPT" finish --state-file "$STATE_FILE" --status completed
log "===== routed ${MODEL_SIZE} cap=${CAP} sw=${SW} aux=${AUX_W} all periods complete ====="
log "Results in: $RESULT_DIR"
