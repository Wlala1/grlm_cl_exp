#!/usr/bin/env python3
"""Maintain per-chain period status for resumable GRLM CL runs."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone


PERIODS = [f"D{i}" for i in range(4)]
STAGES = {"train", "eval", "collect"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def default_state(args):
    periods = {}
    for period in PERIODS:
        periods[period] = {
            "train_status": "pending",
            "eval_status": "pending",
            "collected_status": "pending",
            "checkpoint_path": None,
            "train_log_path": None,
            "eval_log_path": None,
            "collect_log_path": None,
            "result_recall_path": None,
            "result_results_path": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "last_stage": None,
            "last_log_path": None,
        }
    return {
        "schema_version": 1,
        "model_size": args.model_size,
        "cap": args.cap,
        "gpu_ids": args.gpu_ids,
        "run_root": args.run_root,
        "result_dir": args.result_dir,
        "checkpoint_root": args.checkpoint_root,
        "log_dir": args.log_dir,
        "chain_log_path": args.chain_log_path,
        "chain_status": "pending",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "periods": periods,
    }


def load_state(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def write_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state["updated_at"] = utc_now()
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def ensure_periods(state):
    state.setdefault("periods", {})
    for period in PERIODS:
        state["periods"].setdefault(period, default_state(argparse.Namespace(
            model_size=state.get("model_size"),
            cap=state.get("cap"),
            gpu_ids=state.get("gpu_ids"),
            run_root=state.get("run_root"),
            result_dir=state.get("result_dir"),
            checkpoint_root=state.get("checkpoint_root"),
            log_dir=state.get("log_dir"),
            chain_log_path=state.get("chain_log_path"),
        ))["periods"][period])


def cmd_init(args):
    state = load_state(args.state_file)
    if state is None:
        state = default_state(args)
    else:
        ensure_periods(state)
        state.update({
            "model_size": args.model_size,
            "cap": args.cap,
            "gpu_ids": args.gpu_ids,
            "run_root": args.run_root,
            "result_dir": args.result_dir,
            "checkpoint_root": args.checkpoint_root,
            "log_dir": args.log_dir,
            "chain_log_path": args.chain_log_path,
        })
    write_state(args.state_file, state)


def cmd_status(args):
    state = load_state(args.state_file)
    if state is None:
        print("pending")
        return
    period = state.get("periods", {}).get(args.period, {})
    key = "collected_status" if args.stage == "collect" else f"{args.stage}_status"
    print(period.get(key, "pending"))


def cmd_mark(args):
    if args.stage not in STAGES:
        raise SystemExit(f"Unknown stage: {args.stage}")
    if args.period not in PERIODS:
        raise SystemExit(f"Unknown period: {args.period}")

    state = load_state(args.state_file)
    if state is None:
        raise SystemExit(f"State file does not exist: {args.state_file}")
    ensure_periods(state)

    period = state["periods"][args.period]
    status_key = "collected_status" if args.stage == "collect" else f"{args.stage}_status"
    period[status_key] = args.status
    period["last_stage"] = args.stage
    if args.exit_code is not None:
        period["exit_code"] = args.exit_code

    stage_started = f"{args.stage}_started_at"
    stage_finished = f"{args.stage}_finished_at"
    if args.status == "running":
        state["chain_status"] = "running"
        period["started_at"] = period.get("started_at") or utc_now()
        period[stage_started] = utc_now()
    elif args.status in {"completed", "failed", "skipped"}:
        period[stage_finished] = utc_now()
        if args.status == "failed":
            state["chain_status"] = "failed"
        if args.stage == "collect" and args.status == "completed":
            period["finished_at"] = utc_now()

    if args.checkpoint_path:
        period["checkpoint_path"] = args.checkpoint_path
    if args.log_path:
        log_key = f"{args.stage}_log_path"
        if args.stage == "collect":
            log_key = "collect_log_path"
        period[log_key] = args.log_path
        period["last_log_path"] = args.log_path
    if args.result_recall_path:
        period["result_recall_path"] = args.result_recall_path
    if args.result_results_path:
        period["result_results_path"] = args.result_results_path

    write_state(args.state_file, state)


def cmd_finish(args):
    state = load_state(args.state_file)
    if state is None:
        raise SystemExit(f"State file does not exist: {args.state_file}")
    state["chain_status"] = args.status
    state["finished_at"] = utc_now()
    write_state(args.state_file, state)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--state-file", required=True)
    init.add_argument("--model-size", required=True)
    init.add_argument("--cap", required=True)
    init.add_argument("--gpu-ids", required=True)
    init.add_argument("--run-root", required=True)
    init.add_argument("--result-dir", required=True)
    init.add_argument("--checkpoint-root", required=True)
    init.add_argument("--log-dir", required=True)
    init.add_argument("--chain-log-path", required=True)
    init.set_defaults(func=cmd_init)

    status = subparsers.add_parser("status")
    status.add_argument("--state-file", required=True)
    status.add_argument("--period", required=True)
    status.add_argument("--stage", required=True, choices=sorted(STAGES))
    status.set_defaults(func=cmd_status)

    mark = subparsers.add_parser("mark")
    mark.add_argument("--state-file", required=True)
    mark.add_argument("--period", required=True)
    mark.add_argument("--stage", required=True, choices=sorted(STAGES))
    mark.add_argument("--status", required=True,
                      choices=["pending", "running", "completed", "failed", "skipped"])
    mark.add_argument("--exit-code", type=int)
    mark.add_argument("--checkpoint-path")
    mark.add_argument("--log-path")
    mark.add_argument("--result-recall-path")
    mark.add_argument("--result-results-path")
    mark.set_defaults(func=cmd_mark)

    finish = subparsers.add_parser("finish")
    finish.add_argument("--state-file", required=True)
    finish.add_argument("--status", required=True, choices=["completed", "failed"])
    finish.set_defaults(func=cmd_finish)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"chain_state.py error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
