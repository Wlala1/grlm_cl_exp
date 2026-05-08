#!/usr/bin/env python3
"""Run LlamaFactory training with shutdown-aware checkpointing."""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.dirname(SCRIPT_DIR)
LLAMA_SRC = os.path.join(WORK_DIR, "LlamaFactory", "src")

for path in (SCRIPT_DIR, LLAMA_SRC):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from shutdown_checkpoint import (  # noqa: E402
    SaveOnSignalCallback,
    exit_if_shutdown_requested,
    install_signal_handlers,
)
from llamafactory.train.tuner import run_exp  # noqa: E402


def main() -> None:
    install_signal_handlers()
    run_exp(args=sys.argv[1:], callbacks=[SaveOnSignalCallback()])
    exit_if_shutdown_requested()


if __name__ == "__main__":
    main()
