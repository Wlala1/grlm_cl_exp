# GRLM Continual Learning Experiments (Books)

Sequential CL training + Tiger-style sequential evaluation for GRLM (Qwen3) on Amazon Books.

## Requirements

- NVIDIA GPU with 80GB VRAM (H100 recommended)
- 7+ GPUs to run all chains in parallel (or fewer GPUs with longer wall-clock time)
- Python 3.10+, PyTorch 2.x, CUDA 12+
- `huggingface-cli` installed (`pip install huggingface_hub`)
- ~50GB disk for models, ~10GB for data, ~100GB for checkpoints during training

## Quick Start

```bash
# 1. Clone repo
git clone git@github.com:JazyJiang/grlm_cl_exp.git
cd grlm_cl_exp

# 2. Build Docker image (installs LlamaFactory + downloads models/data into the image)
#    Set HF_TOKEN first if HuggingFace access requires authentication.
bash setup.sh

# 3. Run experiments (see dispatch options below)
bash dispatch_all.sh
```

## What `setup.sh` Does

`setup.sh` builds the Docker image `sglang-mini`.

Inside the image it:
1. Clones [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and installs it
2. Downloads Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B from HuggingFace → `/opt/grlm_cl_exp/models/`
3. Downloads experiment data from `JazySong/grlm-books-cl-data` → `/opt/grlm_cl_exp/data/`
4. Extracts `cl_sft.tar.gz` into `/opt/grlm_cl_exp/data/cl_sft/`
5. Creates D0 symlinks (D0 train is identical across all history caps)
6. Generates `dataset_info.json` for LlamaFactory

Useful build overrides:

```bash
HF_TOKEN=... bash setup.sh
IMAGE_TAG=my-grlm:books BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel bash setup.sh
LLAMAFACTORY_REF=<commit-or-tag> bash setup.sh
```

Run the image directly:

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD/runs:/runs" \
  sglang-mini
```

## Experiment Design

**CL Protocol:** D0→D1→D2→D3 sequential fine-tuning on Amazon Books (300K users, 142K items, 5 time periods).

**Models:** Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B

**History caps:** h=2, 5, 10, 20, 30, 40, full (sliding window size for input history)

**Evaluation:** Tiger-style sequential per-target prediction. For each user's D_{t+1} items (chronologically ordered), predict one at a time with a sliding history window. Each (user, target) pair is independently evaluated. Recall@K = fraction of pairs where target appears in top-K beam search candidates.

**Grouping:** Users grouped into 5 quintiles by accumulated history length (Group 1 = longest history, most susceptible to history noise).

## Running Experiments

### Option A: Run all chains on an 8×H200 node

```bash
bash dispatch_all.sh
```

This launches the full 3×7 grid with period-level resume. Runtime outputs are written under `${RUN_ROOT:-/runs}`:
- logs: `/runs/logs/`
- state: `/runs/state/`
- epoch and period checkpoints: `/runs/checkpoints/`
- results and summary tables: `/runs/results/`

By default, `dispatch_all.sh` stages image assets into RAM before launching jobs:
- source in image: `/opt/grlm_cl_exp/{models,data}`
- RAM copy: `/dev/shm/grlm_cl_exp_assets/{models,data}`
- disable with `USE_RAM_ASSETS=0 bash dispatch_all.sh`
- change RAM path with `RAM_ASSET_ROOT=/dev/shm/my_grlm_assets bash dispatch_all.sh`
- force recopy after rebuilding the image with `REFRESH_RAM_ASSETS=1 bash dispatch_all.sh`

Re-running the same command skips completed period evals and resumes from the first incomplete period:

```bash
bash dispatch_all.sh
```

Existing results are protected. A period is considered complete only when both files exist:
- `seq_recall_{tag}_D{t}.json` is valid JSON
- `seq_results_{tag}_D{t}.jsonl` is non-empty

If a period's result files are complete, eval is not re-run or overwritten. If its checkpoint is missing but a later period still needs it, the script may retrain that period only to restore the checkpoint chain, while preserving the existing eval/result files.

### Option B: Sequential per GPU (simpler, fewer GPUs OK)

```bash
bash dispatch_sequential.sh
```

Each GPU runs 0.6B then 1.7B for one cap value. 4B chains run afterward.

### Option C: Run a single chain manually

```bash
bash run_books_cl_v2.sh <model_size> <cap> <gpu_ids>
```

Examples:
```bash
bash run_books_cl_v2.sh 06b h10 1       # 0.6B, cap=10, GPU 1
bash run_books_cl_v2.sh 17b full 3      # 1.7B, full history, GPU 3
bash run_books_cl_v2.sh 4b h20 4,5      # 4B, cap=20, GPUs 4+5
```

Parameters:
- `model_size`: `06b`, `17b`, or `4b`
- `cap`: `h2`, `h5`, `h10`, `h20`, `h30`, `h40`, or `full`
- `gpu_ids`: e.g., `1` (single GPU) or `4,5` (multi-GPU, required for 4B)

Each chain trains D0→D1→D2→D3 sequentially, with eval after each period. Checkpoints are retained by default for restartability; set `KEEP_CHECKPOINTS=0` to restore the older auto-cleanup behavior.

### Monitoring Progress

```bash
# Watch a specific chain's log
tail -f /runs/logs/06b_h10.log

# Watch a period-specific train/eval log
tail -f /runs/logs/06b_h10/train_D0.log
tail -f /runs/logs/06b_h10/eval_D0.log

# Check which chains are done (results appear when eval finishes)
ls /runs/results/cl_results_seq/*/seq_recall_*.json

# Current Cross-Scale table
ls /runs/results/cross_scale_history_noise_analysis.*

# GPU utilization
nvidia-smi
```

## Hyperparameters

| Model | D0 lr | D1+ lr | D0 epochs | D1+ epochs | GPUs | Effective train batch |
|-------|-------|--------|-----------|------------|------|-----------------------|
| 0.6B  | 7e-5  | 3e-5   | 5         | 3          | 1    | 16×4×1 = 64          |
| 1.7B  | 5e-5  | 2e-5   | 5         | 3          | 1    | 16×4×1 = 64          |
| 4B    | 1e-4  | 5e-5   | 5         | 3          | 2    | 16×2×2 = 64          |

D0 trains from the pretrained model (more epochs + higher lr). D1+ fine-tunes from the previous period's checkpoint (lower lr to reduce forgetting).

Training saves full Trainer checkpoints every epoch by default (`SAVE_STRATEGY=epoch`, `SAVE_TOTAL_LIMIT=2`), so optimizer/scheduler state can resume from the latest `checkpoint-*`. Set `SAVE_TOTAL_LIMIT=0` to disable pruning.

## Directory Structure

```
grlm_cl_exp/
├── setup.sh                    # One-time setup (downloads everything)
├── run_books_cl_v2.sh          # Single chain: train 4 periods + eval
├── dispatch_all.sh             # Launch all 21 chains in parallel
├── dispatch_sequential.sh      # Alternative: sequential per GPU
├── scripts/
│   └── generate_dataset_info.py
├── eval/
│   ├── s5_books_cl_eval_seq.py # Sequential eval (Tiger-style)
│   └── recompute_cl_recall.py  # Re-compute recall from saved results
├── LlamaFactory/               # Cloned by setup.sh (not in git)
├── models/                     # Downloaded by setup.sh (not in git)
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   └── Qwen3-4B/
├── data/                       # Downloaded by setup.sh (not in git)
│   ├── books_id2meta.json      # Item metadata (142K items, keywords+title)
│   ├── books_tid2item_id.json  # TID → item_id mapping for eval
│   └── cl_sft/                 # Train + eval JSONs per period × cap
├── checkpoints/                # Training checkpoints (auto-cleaned)
├── logs/                       # Training + eval logs
└── results/
    └── cl_results_seq/         # Final results
        ├── 06b_h10/
        │   ├── seq_recall_h10_D0.json
        │   ├── seq_results_h10_D0.jsonl
        │   └── ...
        └── ...
```

## Output Format

Each eval produces (per period):
- `seq_recall_{tag}_D{t}.json`: Recall@1/5/10/20, overall + per-group breakdown
- `seq_results_{tag}_D{t}.jsonl`: Per-(user, target) hit details for further analysis

After every eval, the collector refreshes:
- `/runs/results/cross_scale_history_noise_analysis.csv`
- `/runs/results/cross_scale_history_noise_analysis.json`
- `/runs/results/cross_scale_history_noise_analysis.md`

Each chain also records period-level resume state in `/runs/state/{model}_{cap}.json`.

Example `seq_recall_h10_D0.json`:
```json
{
  "overall": {"recall@1": 0.032, "recall@5": 0.089, "recall@10": 0.134, "recall@20": 0.187},
  "group_1": {"recall@1": 0.021, ...},
  "group_2": {...},
  ...
}
```

## Troubleshooting

- **`huggingface-cli: command not found`**: Run `pip install huggingface_hub`
- **CUDA OOM during eval**: Reduce eval batch size by editing `run_books_cl_v2.sh` (EVAL_BS variables around line 138)
- **Disk full during training**: Each checkpoint is 2-7GB depending on model size. Checkpoints are retained by default; use `SAVE_TOTAL_LIMIT=N` to keep only the latest N epoch checkpoints, or `KEEP_CHECKPOINTS=0` to delete old period checkpoints after they are no longer needed.
- **Want to re-run eval only (without retraining)**: Use `recompute_cl_recall.py` on existing `seq_results_*.jsonl` files
- **Network issues downloading from HuggingFace**: Set `HF_ENDPOINT` or use a proxy: `export https_proxy=http://your-proxy:port`
