#!/usr/bin/env python3
"""Trainer callback that saves a checkpoint before a graceful shutdown."""

from __future__ import annotations

import os
import signal
from typing import Any

from transformers import TrainerCallback


_REQUESTED = False
_SIGNUM: int | None = None


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _handle_signal(signum: int, _frame: Any) -> None:
    global _REQUESTED, _SIGNUM
    if not _REQUESTED:
        _REQUESTED = True
        _SIGNUM = signum
        print(
            f"[shutdown-checkpoint] received {_signal_name(signum)}; "
            "will save at the next optimizer step.",
            flush=True,
        )
        return

    if os.environ.get("GRLM_SECOND_SIGNAL_EXITS", "1") == "1":
        print(
            f"[shutdown-checkpoint] received {_signal_name(signum)} again; exiting immediately.",
            flush=True,
        )
        os._exit(128 + signum)


def install_signal_handlers() -> None:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, _handle_signal)
        except (AttributeError, OSError, ValueError):
            pass


def shutdown_requested() -> bool:
    return _REQUESTED


def exit_if_shutdown_requested() -> None:
    if _REQUESTED:
        signum = _SIGNUM or signal.SIGTERM
        print(
            "[shutdown-checkpoint] checkpoint requested during shutdown; "
            f"exiting with {128 + signum} so the period can resume later.",
            flush=True,
        )
        raise SystemExit(128 + signum)


class SaveOnSignalCallback(TrainerCallback):
    """Ask Trainer to save and stop after a shutdown signal.

    The save happens through Trainer's normal checkpoint path, so model weights,
    trainer state, scheduler state, and optimizer state are persisted unless
    save_only_model is explicitly enabled.
    """

    def _maybe_stop(self, control: Any) -> Any:
        if _REQUESTED:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._maybe_stop(control)

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        return self._maybe_stop(control)
