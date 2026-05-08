#!/bin/bash
# Host entrypoint: run the full Books CL grid inside Docker.
set -Eeuo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAG=${IMAGE_TAG:-sglang-mini}
RUN_DIR=${RUN_DIR:-${WORK_DIR}/runs}
RUN_ROOT=/runs
USE_RAM_ASSETS=${USE_RAM_ASSETS:-1}
DOCKER_STOP_TIMEOUT=${DOCKER_STOP_TIMEOUT:-900}

if [ "${GRLM_IN_CONTAINER:-0}" = "1" ] || [ -f /.dockerenv ]; then
    export RUN_ROOT=${RUN_ROOT:-/runs}
    exec bash "${WORK_DIR}/scripts/dispatch_h200_grid.sh"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker command not found. Install Docker and NVIDIA Container Toolkit on the H200 host."
    exit 1
fi

if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "ERROR: Docker image not found: $IMAGE_TAG"
    echo "Build it first with:"
    echo "  bash setup.sh"
    exit 1
fi

mkdir -p "$RUN_DIR"

ENV_ARGS=(
    -e RUN_ROOT="$RUN_ROOT"
)

for name in MAX_USERS SMOKE SMOKE_MAX_USERS SMOKE_MAX_STEPS DRY_RUN USE_RAM_ASSETS RAM_ASSET_ROOT REFRESH_RAM_ASSETS BS_OVERRIDE GA_OVERRIDE BS_06b GA_06b BS_17b GA_17b BS_4b GA_4b SAVE_STRATEGY SAVE_STEPS SAVE_TOTAL_LIMIT KEEP_CHECKPOINTS GRLM_SECOND_SIGNAL_EXITS DOCKER_STOP_TIMEOUT; do
    if [ -n "${!name:-}" ]; then
        ENV_ARGS+=(-e "${name}=${!name}")
    fi
done

echo "=== GRLM CL Docker Run ==="
echo "Image: $IMAGE_TAG"
echo "Runs dir: $RUN_DIR -> /runs"
echo "Use RAM assets: $USE_RAM_ASSETS"
echo "Docker stop timeout: ${DOCKER_STOP_TIMEOUT}s"
if [ "$USE_RAM_ASSETS" = "1" ]; then
    echo "RAM asset root: ${RAM_ASSET_ROOT:-/dev/shm/grlm_cl_exp_assets}"
fi

exec docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --stop-timeout "$DOCKER_STOP_TIMEOUT" \
    -v "${RUN_DIR}:${RUN_ROOT}" \
    "${ENV_ARGS[@]}" \
    "$IMAGE_TAG" \
    bash /opt/grlm_cl_exp/scripts/dispatch_h200_grid.sh
