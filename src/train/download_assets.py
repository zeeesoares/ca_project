#!/usr/bin/env python3

import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

from datasets import load_dataset


def download_assets(assets_dir: str):
    models_to_download = {
        "bert-base-uncased": (AutoModelForMaskedLM, "bert-base-uncased"),
        "Qwen/Qwen2.5-1.5B": (AutoModelForCausalLM, "qwen-1.5B")
    }

    for repo_id, (model_class, folder_name) in models_to_download.items():
        dest_path = os.path.join(assets_dir, "models", folder_name)
        os.makedirs(dest_path, exist_ok=True)

        print(f"\n--- Download: {repo_id} ---")
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            trust_remote_code=True
        )

        model = model_class.from_pretrained(
            repo_id,
            trust_remote_code=True
        )

        tokenizer.save_pretrained(dest_path)
        model.save_pretrained(dest_path)
        print(f"Saved in: {dest_path}")

    dataset_dest = os.path.join(assets_dir, "datasets", "wikitext")
    os.makedirs(dataset_dest, exist_ok=True)

    print("\n--- Download: Wikitext ---")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset.save_to_disk(dataset_dest)
    print(f"Dataset saved in: {dataset_dest}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Download models and datasets for training."
    )

    arg_parser.add_argument(
        "--assets-dir",
        type=str,
        required=True,
        help="Directory to save downloaded assets (models and datasets)."
    )

    args = arg_parser.parse_args()

    download_assets(assets_dir=args.assets_dir)
