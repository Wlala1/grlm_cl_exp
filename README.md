# GRLM Continual Learning Experiments (Books)

Sequential CL training + Tiger-style sequential evaluation for GRLM (Qwen3) on Amazon Books.

## Requirements

- NVIDIA GPU (H100 80GB recommended)
- Python 3.10+, PyTorch 2.x, CUDA 12+
- ~50GB disk for models, ~10GB for data, ~30GB for checkpoints during training

## Quick Start

```bash
# 1. Clone and setup
git clone <this-repo>
cd grlm_cl_exp

# 2. Install dependencies & download models/data
bash setup.sh

# 3. Run a single chain (e.g., 0.6B, h=10, GPU 1)
cd LlamaFactory
bash run_books_cl_v2.sh 06b h10 1

# 4. Run all chains in parallel (uses GPUs 1-7)
bash dispatch_all.sh
```

## Experiment Design

**CL Protocol:** D0→D1→D2→D3 sequential fine-tuning on Amazon Books (300K users, 142K items, 5 periods).

**Models:** Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B-Instruct

**History caps:** h=2, 5, 10, 20, 30, 40, full (sliding window size at eval time)

**Evaluation:** Tiger-style sequential per-target prediction. For each user's D_{t+1} items (chronologically ordered), predict one at a time with a sliding history window. Each (user, target) pair is independently evaluated. Recall@K = fraction of pairs where target appears in top-K beam search candidates.

**Grouping:** Users grouped into 5 quintiles by accumulated history length (Group 1 = longest history, most susceptible to history noise).

## Directory Structure

```
grlm_cl_exp/
├── setup.sh                    # Download models + data
├── dispatch_all.sh             # Launch all 21 chains on GPUs 1-7
├── LlamaFactory/
│   ├── run_books_cl_v2.sh      # Single chain: train 4 periods + sequential eval
│   └── data/
│       └── dataset_info.json   # Dataset registry (auto-populated by setup.sh)
├── eval/
│   ├── s5_books_cl_eval_seq.py # Sequential eval script (Tiger-style)
│   └── recompute_cl_recall.py  # Post-hoc recompute from existing results
├── data/                       # Downloaded by setup.sh (not in git)
│   ├── books_id2meta.json
│   ├── books_tid2item_id.json
│   └── cl_sft/                 # Train + eval JSONs for all periods × caps
└── results/                    # Output directory
    └── cl_results_seq/
```

## Single Chain Usage

```bash
bash run_books_cl_v2.sh <model_size> <cap> <gpu_ids>
```

- `model_size`: `06b`, `17b`, or `4b`
- `cap`: `h2`, `h5`, `h10`, `h20`, `h30`, `h40`, or `full`
- `gpu_ids`: e.g., `1` (single GPU) or `4,5` (multi-GPU for 4B)

Each chain runs 4 periods sequentially (D0→D1→D2→D3), with training + eval at each step. Results saved to `results/cl_results_seq/{model}_{cap}/`.

## Hyperparameters

| Model | D0 lr | D1+ lr | D0 epochs | D1+ epochs | GPUs |
|-------|-------|--------|-----------|------------|------|
| 0.6B  | 7e-5  | 3e-5   | 5         | 3          | 1    |
| 1.7B  | 5e-5  | 2e-5   | 5         | 3          | 1    |
| 4B    | 1e-4  | 5e-5   | 5         | 3          | 2    |

## Output

Each eval produces:
- `seq_recall_{hist_tag}.json`: Overall + per-group Recall@1/5/10/20
- `seq_results_{hist_tag}.jsonl`: Per-(user,target) pair hit details
