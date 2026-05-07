# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ARG LLAMAFACTORY_REPO=https://github.com/hiyouga/LLaMA-Factory.git
ARG LLAMAFACTORY_REF=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    GRLM_IN_CONTAINER=1 \
    RUN_ROOT=/runs \
    LLAMA_DIR=/opt/grlm_cl_exp/LlamaFactory \
    DATA_DIR=/opt/grlm_cl_exp/data \
    MODEL_DIR=/opt/grlm_cl_exp/models \
    RESULTS_ROOT=/runs/results \
    LLAMAFACTORY_REPO=${LLAMAFACTORY_REPO} \
    LLAMAFACTORY_REF=${LLAMAFACTORY_REF}

WORKDIR /opt/grlm_cl_exp

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        rsync \
        tini \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/grlm_cl_exp

RUN chmod +x \
        /opt/grlm_cl_exp/setup.sh \
        /opt/grlm_cl_exp/dispatch_all.sh \
        /opt/grlm_cl_exp/run_books_cl_v2.sh \
        /opt/grlm_cl_exp/scripts/*.sh \
        /opt/grlm_cl_exp/scripts/*.py \
        /opt/grlm_cl_exp/eval/*.py

RUN --mount=type=secret,id=hf_token,required=false \
    bash /opt/grlm_cl_exp/scripts/prepare_image_assets.sh

RUN mkdir -p /runs \
    && chmod -R a+rX /opt/grlm_cl_exp \
    && chmod -R a+rwX /opt/grlm_cl_exp/LlamaFactory/data \
    && chmod -R a+rwX /runs

VOLUME ["/runs"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "/opt/grlm_cl_exp/scripts/dispatch_h200_grid.sh"]
