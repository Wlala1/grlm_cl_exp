"""
s5_books_cl_eval_seq.py
CL evaluation: Sequential per-target eval (Tiger-style).
For each user, predict D_{t+1} items one by one with sliding history window.
Each target gets its own beam search with growing history.
Recall@K = fraction of (user, target) pairs where target is in top-K.
"""
import re
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time
from datetime import datetime, timezone

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def create_reverse_mapping(original_dict):
    rm = {}
    w2k = defaultdict(list)
    for key_str, ids in original_dict.items():
        words = [w.strip().lower() for w in key_str.split(',')]
        rm[key_str] = {'words': words, 'ids': ids}
        for w in words:
            w2k[w].append(key_str)
    return rm, w2k


def get_iid_by_tid(content, tid2item_id_local, reverse_mapping_local, word_to_keys_local):
    iids = []
    tids = content.replace("[", "").replace("]", "").split(", ")
    tid_key = ",".join(tids)
    if tid_key in tid2item_id_local:
        iids.extend(tid2item_id_local[tid_key])
    else:
        candidate_scores = defaultdict(float)
        for i, query_word in enumerate(tids):
            position_weight = 1.0 / (i + 1)
            for candidate_word, candidate_keys in word_to_keys_local.items():
                similarity = 0.0
                if query_word == candidate_word:
                    similarity = 1.0
                elif query_word in candidate_word or candidate_word in query_word:
                    similarity = 0.8
                if similarity > 0:
                    for candidate_key in candidate_keys:
                        candidate_scores[candidate_key] += similarity * position_weight
            if len(candidate_scores) > 1000:
                break
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        for candidate_key, score in sorted_candidates[:5]:
            iids.extend(reverse_mapping_local[candidate_key]['ids'])
        iids = iids[:1]
    return iids


def parse_prompt_items(prompt_text):
    """Parse prompt into list of individual item strings."""
    pattern = r"Item text ID: \[.*?\] Title: .*?\.\n?"
    items = re.findall(pattern, prompt_text)
    return [item.strip() for item in items]


def format_target_item(target_tid, title):
    """Format a target item as history entry."""
    keywords = ", ".join(target_tid)
    return f"Item text ID: [{keywords}] Title: {title}."


def process_single_gpu(rank, data_slice, output_queue, model_name, batch_size,
                       tid2item_id_path, id2meta_path, max_hist):
    torch.cuda.set_device(rank)

    with open(tid2item_id_path, 'r') as f:
        local_tid2item = json.load(f)
    local_rm, local_w2k = create_reverse_mapping(local_tid2item)

    with open(id2meta_path, 'r') as f:
        id2meta = json.load(f)

    print(f"Rank {rank}: Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=f"cuda:{rank}"
    )
    model.eval()
    print(f"Rank {rank}: Processing {len(data_slice)} users, batch_size={batch_size}")

    INSTRUCTION = (
        "Based on the user's historical product interaction sequence, predict the next "
        "product's characteristic words. \nEach product is represented by exactly 5 "
        "characteristic words enclosed in square brackets []. The historical sequence "
        "shows the user's interaction pattern.\n"
    )

    local_hits = [0] * 20
    local_total = 0
    local_results = []

    # Initialize user states for batched processing
    user_states = []
    for sample in data_slice:
        history_items = parse_prompt_items(sample["prompt"])
        user_states.append({
            "user_id": sample["user_id"],
            "current_history": list(history_items),
            "base_history_len": len(history_items),
            "target_ids": sample["target_item_ids"],
            "target_tids": sample["target_tids"],
            "target_idx": 0,
        })

    total_pairs = sum(len(s["target_ids"]) for s in user_states)
    pbar = tqdm(total=total_pairs, desc=f"GPU {rank}")

    while user_states:
        # Batch: take up to batch_size users' current targets
        batch_states = user_states[:batch_size]

        # Build batch inputs
        batch_texts = []
        for state in batch_states:
            if max_hist is not None:
                prompt_items = state["current_history"][-max_hist:]
            else:
                prompt_items = state["current_history"]

            prompt_text = "\n".join(prompt_items) + "\n"
            full_prompt = INSTRUCTION + prompt_text + "Item text ID: "
            messages = [{"role": "user", "content": full_prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_texts.append(text)

        model_inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=8192, return_attention_mask=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=30, do_sample=False,
                num_beams=20, num_return_sequences=20,
                pad_token_id=tokenizer.eos_token_id,
            )

        num_seqs = generated_ids.shape[0] // len(batch_states)

        # Process each user in batch
        finished_indices = []
        for batch_idx, state in enumerate(batch_states):
            t_idx = state["target_idx"]
            target_id = state["target_ids"][t_idx]
            target_tid = state["target_tids"][t_idx]

            # Extract candidates from this user's beam outputs
            contents = []
            for seq_idx in range(batch_idx * num_seqs, (batch_idx + 1) * num_seqs):
                input_len = model_inputs.input_ids[batch_idx].shape[0]
                output_ids = generated_ids[seq_idx][input_len:].tolist()
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                pattern = r'\[(.*?)\]'
                cons = re.findall(pattern, content)
                for c in cons:
                    cs = "[" + c + "]"
                    if cs not in contents:
                        contents.append(cs)

            # Map TID contents to item_ids
            iids = []
            for content in contents:
                iid = get_iid_by_tid(content, local_tid2item, local_rm, local_w2k)
                for i in iid:
                    if i not in iids:
                        iids.append(i)
                    if len(iids) >= 20:
                        break
                if len(iids) >= 20:
                    break
            iids = iids[:20]

            # Check hit
            local_total += 1
            hit_pos = -1
            for k_idx, iid in enumerate(iids):
                if iid == target_id:
                    hit_pos = k_idx + 1
                    for pos in range(k_idx, 20):
                        local_hits[pos] += 1
                    break

            if max_hist is not None:
                eval_hist_len = min(len(state["current_history"]), max_hist)
            else:
                eval_hist_len = len(state["current_history"])

            local_results.append({
                "user_id": state["user_id"],
                "history_len": state["base_history_len"],
                "eval_history_len": eval_hist_len,
                "target_idx": t_idx,
                "target_id": target_id,
                "hit_pos": hit_pos,
                "iids": iids,
            })

            # Add target to history
            title = id2meta.get(target_id, {}).get("title", "Unknown")
            target_item_str = format_target_item(target_tid, title)
            state["current_history"].append(target_item_str)

            # Advance to next target or mark finished
            state["target_idx"] += 1
            if state["target_idx"] >= len(state["target_ids"]):
                finished_indices.append(batch_idx)

            pbar.update(1)

        # Remove finished users (iterate in reverse to preserve indices)
        for idx in sorted(finished_indices, reverse=True):
            user_states.pop(idx)

    pbar.close()

    output_queue.put((rank, local_hits, local_total, local_results))
    print(f"Rank {rank}: Done. {local_total} pairs evaluated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True, help='CL eval JSON')
    parser.add_argument('--tid2item_id', type=str, required=True)
    parser.add_argument('--id2meta', type=str, required=True, help='books_id2meta.json')
    parser.add_argument('--max_hist', type=int, default=None)
    parser.add_argument('--max_users', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./cl_results')
    parser.add_argument('--period', type=int, default=None,
                        help='Training period index. Used to write D-specific outputs.')
    parser.add_argument('--model_size', type=str, default=None)
    parser.add_argument('--cap', type=str, default=None)
    parser.add_argument('--gpu_ids', type=str, default=None)
    args = parser.parse_args()

    print(f"Loading eval data: {args.eval_file}")
    eval_data = json.load(open(args.eval_file))
    print(f"Total users: {len(eval_data)}")

    if args.max_users:
        random.shuffle(eval_data)
        eval_data = eval_data[:args.max_users]
        print(f"Sampled {len(eval_data)} users")

    total_targets = sum(len(s["target_item_ids"]) for s in eval_data)
    print(f"Total (user, target) pairs: {total_targets}")

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    chunk_size = len(eval_data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = len(eval_data) if i == num_gpus - 1 else start + chunk_size
        data_chunks.append(eval_data[start:end])

    output_queue = mp.Queue()
    processes = []
    start_time = time.time()

    for rank in range(num_gpus):
        p = mp.Process(
            target=process_single_gpu,
            args=(rank, data_chunks[rank], output_queue, args.model, args.batch_size,
                  args.tid2item_id, args.id2meta, args.max_hist)
        )
        processes.append(p)
        p.start()

    all_hits = [0] * 20
    all_total = 0
    all_results = []

    for _ in range(num_gpus):
        rank, hits, total, results = output_queue.get()
        print(f"GPU {rank}: {total} pairs evaluated")
        for i in range(20):
            all_hits[i] += hits[i]
        all_total += total
        all_results.extend(results)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    recall = {
        "recall@1": all_hits[0] / all_total if all_total > 0 else 0,
        "recall@5": all_hits[4] / all_total if all_total > 0 else 0,
        "recall@10": all_hits[9] / all_total if all_total > 0 else 0,
        "recall@20": all_hits[19] / all_total if all_total > 0 else 0,
    }

    hist_tag = f"h{args.max_hist}" if args.max_hist else "hfull"
    period_tag = f"_D{args.period}" if args.period is not None else ""
    eval_transition = None
    if args.period is not None:
        eval_transition = f"D{args.period}->D{args.period + 1}"

    print(f"\n{'='*50}")
    if eval_transition:
        print(f"Sequential CL Eval ({hist_tag}, {eval_transition}, {all_total} pairs, {elapsed:.0f}s)")
    else:
        print(f"Sequential CL Eval ({hist_tag}, {all_total} pairs, {elapsed:.0f}s)")
    print(f"{'='*50}")
    for k, v in recall.items():
        print(f"  {k}: {v:.4f}")

    # Group by history_len (Group 1 = longest accumulated history)
    all_results.sort(key=lambda x: x["history_len"], reverse=True)
    n = len(all_results)
    group_size = n // 5
    group_results = {}

    for g in range(1, 6):
        if g < 5:
            group_data = all_results[(g-1)*group_size : g*group_size]
        else:
            group_data = all_results[(g-1)*group_size:]

        g_total = len(group_data)
        g_hits = [0] * 20
        for r in group_data:
            if r["hit_pos"] > 0:
                for pos in range(r["hit_pos"] - 1, 20):
                    g_hits[pos] += 1

        g_recall = {
            "recall@1": g_hits[0] / g_total if g_total > 0 else 0,
            "recall@5": g_hits[4] / g_total if g_total > 0 else 0,
            "recall@10": g_hits[9] / g_total if g_total > 0 else 0,
            "recall@20": g_hits[19] / g_total if g_total > 0 else 0,
        }

        hist_lens = [r["history_len"] for r in group_data]
        group_results[f"group{g}"] = {
            "n_pairs": g_total,
            "history_len_range": [int(max(hist_lens)), int(min(hist_lens))],
            "metrics": g_recall,
        }
        print(f"  Group {g} ({g_total} pairs, hlen {max(hist_lens)}-{min(hist_lens)}): "
              f"R@1={g_recall['recall@1']:.4f} R@5={g_recall['recall@5']:.4f} "
              f"R@10={g_recall['recall@10']:.4f} R@20={g_recall['recall@20']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save per-pair results
    results_file = os.path.join(args.output_dir, f"seq_results_{hist_tag}{period_tag}.jsonl")
    with open(results_file, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r) + '\n')
    print(f"Saved {len(all_results)} pair results to {results_file}")

    recall_file = os.path.join(args.output_dir, f"seq_recall_{hist_tag}{period_tag}.json")
    with open(recall_file, 'w') as f:
        json.dump({
            "eval_type": "sequential_per_target",
            "model_size": args.model_size,
            "cap": args.cap,
            "train_period": args.period,
            "eval_transition": eval_transition,
            "gpu_ids": args.gpu_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": recall,
            "n_pairs": all_total,
            "n_users": len(eval_data),
            "elapsed_s": elapsed,
            "group_by": "history_len",
            "groups": group_results,
        }, f, indent=2)
    print(f"Saved to {recall_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
