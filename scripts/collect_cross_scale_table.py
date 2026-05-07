#!/usr/bin/env python3
"""Collect Cross-Scale History Noise Analysis table from CL recall files."""
import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux training hosts have fcntl.
    fcntl = None


MODEL_ORDER = ["06b", "17b", "4b"]
MODEL_LABELS = {
    "06b": "Qwen3-0.6B",
    "17b": "Qwen3-1.7B",
    "4b": "Qwen3-4B",
}
CAP_ORDER = ["h2", "h5", "h10", "h20", "h30", "h40", "full"]
PERIODS = [0, 1, 2, 3]
GROUPS = [1, 2, 3, 4, 5]


@contextmanager
def file_lock(lock_path):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def hist_tag(cap):
    return "hfull" if cap == "full" else cap


def recall_path(cl_results_dir, model, cap, period):
    return os.path.join(
        cl_results_dir,
        f"{model}_{cap}",
        f"seq_recall_{hist_tag(cap)}_D{period}.json",
    )


def extract_group_recall(payload, group):
    groups = payload.get("groups", {})
    group_payload = groups.get(f"group{group}") or groups.get(f"group_{group}")
    if not group_payload:
        return None
    metrics = group_payload.get("metrics", {})
    value = metrics.get("recall@20")
    if value is None:
        return None
    return float(value)


def load_cells(cl_results_dir, model, cap):
    cells = {}
    sources = {}
    for group in GROUPS:
        for period in PERIODS:
            key = f"G{group}_D{period}"
            path = recall_path(cl_results_dir, model, cap, period)
            if not os.path.exists(path):
                cells[key] = None
                continue
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
                cells[key] = extract_group_recall(payload, group)
                sources[key] = path
            except Exception as exc:
                print(f"[warn] failed to read {path}: {exc}", file=sys.stderr)
                cells[key] = None
    return cells, sources


def build_rows(cl_results_dir):
    rows = []
    for model in MODEL_ORDER:
        for cap in CAP_ORDER:
            cells, sources = load_cells(cl_results_dir, model, cap)
            values = [value for value in cells.values() if value is not None]
            complete = len(values) == len(GROUPS) * len(PERIODS)
            avg = sum(values) / len(values) if complete else None
            rows.append({
                "model": model,
                "cap": cap,
                "method": f"{MODEL_LABELS[model]} ({cap.replace('h', 'h=') if cap != 'full' else 'full'})",
                "complete": complete,
                "completed_cells": len(values),
                "expected_cells": len(GROUPS) * len(PERIODS),
                "avg": avg,
                "cells": cells,
                "sources": sources,
            })
    return rows


def pct(value, digits=2):
    if value is None:
        return ""
    return f"{value * 100:.{digits}f}"


def table_headers():
    headers = ["Method"]
    for group in GROUPS:
        for period in PERIODS:
            headers.append(f"Group {group} D{period}->{period + 1}")
    headers.append("Avg")
    return headers


def row_for_csv(row):
    out = [row["method"]]
    for group in GROUPS:
        for period in PERIODS:
            out.append(pct(row["cells"][f"G{group}_D{period}"]))
    out.append(pct(row["avg"]))
    return out


def atomic_write(path, writer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    writer(tmp_path)
    os.replace(tmp_path, path)


def write_csv(path, rows):
    def writer(tmp_path):
        with open(tmp_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(table_headers())
            for row in rows:
                csv_writer.writerow(row_for_csv(row))
    atomic_write(path, writer)


def write_json(path, rows, cl_results_dir):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "Recall@20",
        "unit": "fraction",
        "display_outputs": {
            "csv": "percent",
            "markdown": "percent",
            "json": "fraction",
        },
        "source_results_dir": cl_results_dir,
        "models": MODEL_ORDER,
        "caps": CAP_ORDER,
        "groups": GROUPS,
        "periods": [f"D{period}->D{period + 1}" for period in PERIODS],
        "rows": rows,
    }

    def writer(tmp_path):
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    atomic_write(path, writer)


def write_markdown(path, rows):
    headers = table_headers()
    lines = [
        "# Cross-Scale History Noise Analysis",
        "",
        "Metric: Recall@20 (%). Empty cells are not completed yet.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row_for_csv(row)) + " |")
    lines.append("")

    def writer(tmp_path):
        with open(tmp_path, "w") as f:
            f.write("\n".join(lines))
    atomic_write(path, writer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", "/runs/results"))
    parser.add_argument("--cl-results-dir", default=None)
    parser.add_argument("--output-prefix", default="cross_scale_history_noise_analysis")
    args = parser.parse_args()

    results_root = args.results_root
    cl_results_dir = args.cl_results_dir or os.path.join(results_root, "cl_results_seq")
    lock_path = os.path.join(results_root, ".collect_cross_scale_table.lock")

    with file_lock(lock_path):
        rows = build_rows(cl_results_dir)
        csv_path = os.path.join(results_root, f"{args.output_prefix}.csv")
        json_path = os.path.join(results_root, f"{args.output_prefix}.json")
        md_path = os.path.join(results_root, f"{args.output_prefix}.md")
        write_csv(csv_path, rows)
        write_json(json_path, rows, cl_results_dir)
        write_markdown(md_path, rows)

    completed = sum(row["completed_cells"] for row in rows)
    expected = sum(row["expected_cells"] for row in rows)
    print(f"Collected {completed}/{expected} cells")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
