"""
Generate dataset_info.json for LlamaFactory.
Creates entries for all CL periods × history caps.

Usage:
    python scripts/generate_dataset_info.py \
        --data_dir data/cl_sft \
        --output LlamaFactory/data/dataset_info.json
"""
import os
import json
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing CL SFT JSON files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for dataset_info.json')
    args = parser.parse_args()

    dataset_info = {}

    # Find all training JSON files
    train_files = sorted(glob(os.path.join(args.data_dir, 'amazon_books_cl_*_train*.json')))

    for fpath in train_files:
        fname = os.path.basename(fpath)
        # e.g. amazon_books_cl_D0_train.json or amazon_books_cl_D1_train_h10.json
        name_no_ext = fname.replace('.json', '')

        # Convert filename to dataset key: amazon_books_cl_D0_train -> grlm_indomain_books_cl_D0
        # amazon_books_cl_D1_train_h10 -> grlm_indomain_books_cl_D1_h10
        parts = name_no_ext.replace('amazon_books_cl_', '').replace('_train', '')
        # parts is now like "D0" or "D1_h10"
        dataset_key = f"grlm_indomain_books_cl_{parts}"

        dataset_info[dataset_key] = {
            "file_name": os.path.join("grlm_in_domain", fname),
            "columns": {
                "prompt": "prompt",
                "response": "response"
            }
        }

    # Also check if there's existing dataset_info to merge with
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            existing = json.load(f)
        # Keep existing entries, add/overwrite CL entries
        existing.update(dataset_info)
        dataset_info = existing

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Generated {len(dataset_info)} dataset entries -> {args.output}")
    # Show CL entries
    cl_entries = [k for k in dataset_info if 'cl_D' in k]
    print(f"CL entries: {len(cl_entries)}")
    for k in sorted(cl_entries):
        print(f"  {k}: {dataset_info[k]['file_name']}")


if __name__ == "__main__":
    main()
