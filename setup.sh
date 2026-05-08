#!/bin/bash
# Build the Docker image containing code, LlamaFactory, Qwen3 models, and Books CL data.
set -Eeuo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAG=${IMAGE_TAG:-sglang-mini}
BASE_IMAGE=${BASE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}
LLAMAFACTORY_REPO=${LLAMAFACTORY_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}
LLAMAFACTORY_REF=${LLAMAFACTORY_REF:-}

echo "=== GRLM CL Docker Build ==="
echo "Working directory: $WORK_DIR"
echo "Image tag: $IMAGE_TAG"
echo "Base image: $BASE_IMAGE"
echo "LlamaFactory repo: $LLAMAFACTORY_REPO"
if [ -n "$LLAMAFACTORY_REF" ]; then
    echo "LlamaFactory ref: $LLAMAFACTORY_REF"
fi

BUILD_ARGS=(
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "LLAMAFACTORY_REPO=${LLAMAFACTORY_REPO}"
)
if [ -n "$LLAMAFACTORY_REF" ]; then
    BUILD_ARGS+=(--build-arg "LLAMAFACTORY_REF=${LLAMAFACTORY_REF}")
fi

SECRET_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
    SECRET_ARGS=(--secret id=hf_token,env=HF_TOKEN)
    echo "HF_TOKEN detected: passing as BuildKit secret"
else
    echo "HF_TOKEN not set: build will use public HuggingFace access"
fi

cd "$WORK_DIR"
DOCKER_BUILDKIT=1 docker build \
    "${BUILD_ARGS[@]}" \
    "${SECRET_ARGS[@]}" \
    -t "$IMAGE_TAG" \
    .

echo ""
echo "=== Docker Build Complete ==="
echo "Run all experiments:"
echo "  bash dispatch_all.sh"
