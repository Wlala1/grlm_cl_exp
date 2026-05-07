"""
Recompute CL recall using per-target evaluation (Tiger-style).
Instead of "any hit per user", each (user, target) pair is independently evaluated.
Also groups by history_len (accumulated history, like Tiger) instead of period_t_count.

Usage:
    python recompute_cl_recall.py --results_dir /path/to/cl_results --eval_dir /path/to/eval_jsons
"""
import json
import os
import argparse
import numpy as np
from collections import defaultdict


def recompute_single(user_results_path, eval_json_path, output_path):
    """Recompute per-target recall for one (model, cap, period) combo."""
    # Load user results (has iids = top-20 candidates)
    user_results = {}
    with open(user_results_path) as f:
        for line in f:
            r = json.loads(line)
            user_results[r["user_id"]] = r

    # Load eval JSON (has target_item_ids per user)
    with open(eval_json_path) as f:
        eval_data = json.load(f)

    # Build user_id -> target_item_ids mapping
    eval_targets = {}
    for sample in eval_data:
        eval_targets[sample["user_id"]] = sample["target_item_ids"]

    # Per-target evaluation
    all_pairs = []  # (history_len, hit_at_per_target)
    total_pairs = 0
    hits = [0] * 20

    for user_id, result in user_results.items():
        iids = result["iids"]
        history_len = result["history_len"]
        targets = eval_targets.get(user_id, [])

        for target_id in targets:
            total_pairs += 1
            hit_pos = -1
            for k_idx, iid in enumerate(iids):
                if iid == target_id:
                    hit_pos = k_idx + 1
                    for pos in range(k_idx, 20):
                        hits[pos] += 1
                    break
            all_pairs.append({
                "user_id": user_id,
                "history_len": history_len,
                "period_t_count": result.get("period_t_count", 0),
                "target_id": target_id,
                "hit_pos": hit_pos,
            })

    # Overall recall
    recall = {
        "recall@1": hits[0] / total_pairs if total_pairs > 0 else 0,
        "recall@5": hits[4] / total_pairs if total_pairs > 0 else 0,
        "recall@10": hits[9] / total_pairs if total_pairs > 0 else 0,
        "recall@20": hits[19] / total_pairs if total_pairs > 0 else 0,
    }

    # Group by history_len (Tiger-style: group1 = longest history)
    all_pairs.sort(key=lambda x: x["history_len"], reverse=True)
    n = len(all_pairs)
    group_size = n // 5
    group_results = {}

    for g in range(1, 6):
        if g < 5:
            group_pairs = all_pairs[(g-1)*group_size : g*group_size]
        else:
            group_pairs = all_pairs[(g-1)*group_size:]

        g_total = len(group_pairs)
        g_hits = [0] * 20
        for p in group_pairs:
            if p["hit_pos"] > 0:
                for pos in range(p["hit_pos"] - 1, 20):
                    g_hits[pos] += 1

        g_recall = {
            "recall@1": g_hits[0] / g_total if g_total > 0 else 0,
            "recall@5": g_hits[4] / g_total if g_total > 0 else 0,
            "recall@10": g_hits[9] / g_total if g_total > 0 else 0,
            "recall@20": g_hits[19] / g_total if g_total > 0 else 0,
        }

        hist_lens = [p["history_len"] for p in group_pairs]
        group_results[f"group{g}"] = {
            "n_pairs": g_total,
            "history_len_range": [int(max(hist_lens)), int(min(hist_lens))],
            "metrics": g_recall,
        }

    output = {
        "metric_type": "per_target (Tiger-style)",
        "group_by": "history_len",
        "total_pairs": total_pairs,
        "n_users": len(user_results),
        "metrics": recall,
        "groups": group_results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  R@1={recall['recall@1']:.4f} R@5={recall['recall@5']:.4f} "
          f"R@10={recall['recall@10']:.4f} R@20={recall['recall@20']:.4f} "
          f"({total_pairs} pairs, {len(user_results)} users)")
    for g in range(1, 6):
        gr = group_results[f"group{g}"]
        m = gr["metrics"]
        print(f"    G{g} ({gr['n_pairs']} pairs, hlen {gr['history_len_range'][0]}-{gr['history_len_range'][1]}): "
              f"R@20={m['recall@20']:.4f}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='/workspace/jiangzhuosong/GRLM_0/in_domain/books/cl_results')
    parser.add_argument('--eval_dir', type=str,
                        default='/workspace/jiangzhuosong/GRLM_0/LlamaFactory/data/grlm_in_domain')
    args = parser.parse_args()

    # Find all user_results files
    for model_cap_dir in sorted(os.listdir(args.results_dir)):
        dir_path = os.path.join(args.results_dir, model_cap_dir)
        if not os.path.isdir(dir_path):
            continue

        for fname in sorted(os.listdir(dir_path)):
            if not fname.startswith("user_results_D") or not fname.endswith(".jsonl"):
                continue

            # Parse: user_results_D{period}_{hist_tag}.jsonl
            # e.g., user_results_D0_hfull.jsonl, user_results_D2_hfull.jsonl
            parts = fname.replace("user_results_", "").replace(".jsonl", "")
            # parts like "D0_hfull" or "D2_h20"
            period_str = parts.split("_")[0]  # "D0", "D2", etc.
            period = int(period_str[1:])

            # Corresponding eval file
            eval_file = os.path.join(args.eval_dir, f"amazon_books_cl_D{period}_eval.json")
            if not os.path.exists(eval_file):
                print(f"[skip] {model_cap_dir}/{fname} — eval file not found: {eval_file}")
                continue

            user_results_path = os.path.join(dir_path, fname)
            output_path = os.path.join(dir_path, fname.replace("user_results_", "recall_pertarget_"))
            output_path = output_path.replace(".jsonl", ".json")

            print(f"\n{model_cap_dir}/{fname} (period D{period}):")
            recompute_single(user_results_path, eval_file, output_path)


if __name__ == "__main__":
    main()
