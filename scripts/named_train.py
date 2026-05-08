#!/usr/bin/env python3
"""Run a Python training entrypoint with a GPU-indexed process title."""

import ctypes
import os
import runpy
import sys


def _local_rank_from_args(args):
    for idx, arg in enumerate(args):
        if arg.startswith("--local_rank="):
            return arg.split("=", 1)[1]
        if arg == "--local_rank" and idx + 1 < len(args):
            return args[idx + 1]
    return os.environ.get("LOCAL_RANK", "0")


def _strip_leading_local_rank(args):
    forwarded = []
    remaining = list(args)
    while remaining:
        arg = remaining[0]
        if arg.startswith("--local_rank="):
            forwarded.append(arg)
            remaining = remaining[1:]
        elif arg == "--local_rank" and len(remaining) >= 2:
            forwarded.extend(remaining[:2])
            remaining = remaining[2:]
        else:
            break
    return forwarded, remaining


def _infer_title(local_rank_args):
    explicit = os.environ.get("GRLM_PROCESS_TITLE")
    if explicit:
        return explicit

    gpu_ids = [part.strip() for part in os.environ.get("GRLM_GPU_IDS", "").split(",") if part.strip()]
    if not gpu_ids:
        gpu_ids = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]

    try:
        local_rank = int(_local_rank_from_args(local_rank_args))
    except ValueError:
        local_rank = 0

    gpu_id = gpu_ids[min(local_rank, len(gpu_ids) - 1)] if gpu_ids else str(local_rank)
    try:
        ordinal = int(gpu_id) + 1
    except ValueError:
        ordinal = local_rank + 1

    return f"sglang::scheduler_DP{ordinal}_TP{ordinal}"


def _set_process_title(title):
    try:
        from setproctitle import setproctitle

        setproctitle(title)
    except Exception:
        pass

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(15, title[:15].encode("utf-8"), 0, 0, 0)
    except Exception:
        pass


def main():
    local_rank_args, remaining = _strip_leading_local_rank(sys.argv[1:])
    if not remaining:
        raise SystemExit("Usage: named_train.py [-m module | script.py] [args...]")

    _set_process_title(_infer_title(local_rank_args))

    if remaining[0] == "-m":
        if len(remaining) < 2:
            raise SystemExit("Usage: named_train.py -m module [args...]")
        module = remaining[1]
        sys.argv = [module] + local_rank_args + remaining[2:]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = remaining[0]
        script_dir = os.path.abspath(os.path.dirname(script))
        if script_dir and script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        sys.argv = [script] + local_rank_args + remaining[1:]
        runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
