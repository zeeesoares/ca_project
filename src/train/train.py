#!/usr/bin/env python3

import argparse
import sys
import torch

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.train.model import build_model
from src.train.dataset import build_dataloader
from src.train.trainer import Trainer

from src.torch_ext.checkpoint import Checkpoint


def parse_args():
    """
    Parses the following command-line arguments for training configuration:
        --checkpoint-pfs-dir (str, required)
        --checkpoint-local-dir (str, optional, default="/tmp/")
        --profile (flag)
        --checkpoint-interval (int, optional)
        --total-steps (int, optional)
        --warmup-steps (int, optional)
    """

    DEFAULT_CHECKPOINT_LOCAL_DIR = "/tmp/"
    DEFAULT_CHECKPOINT_INTERVAL  = 10
    DEFAULT_TOTAL_STEPS          = 50
    DEFAULT_WARMUP_STEPS         = 10

    parser = argparse.ArgumentParser(
        description=(
            "Train a model with orchestrated checkpoint migration to PFS."
        ),
        add_help=False
    )

    parser.add_argument(
        "-h", "--help",
        action="help",
        help="Show this help message and exit."
    )

    parser.add_argument(
        "--checkpoint-pfs-dir",
        type=str,
        required=True,
        help="PFS directory where migrated checkpoints should be stored.",
    )

    parser.add_argument(
        "--checkpoint-local-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_LOCAL_DIR,
        help=(
            "Local directory where checkpoints are saved before migration. "
            f"Default: {DEFAULT_CHECKPOINT_LOCAL_DIR}"
        ),
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable PyTorch profiler."
        ),
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=(
            "Number of steps between checkpoint saves. "
            f"Default: {DEFAULT_CHECKPOINT_INTERVAL}"
        ),
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_TOTAL_STEPS,
        help=(
            "Total number of training steps to perform. "
            f"Default: {DEFAULT_TOTAL_STEPS}"
        ),
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help=(
            "Number of warmup steps for the learning rate scheduler. "
            f"Default: {DEFAULT_WARMUP_STEPS}"
        ),
    )

    return parser.parse_args()


def main(
    checkpoint_pfs_dir: str,
    checkpoint_local_dir: str,
    profile: bool,
    checkpoint_interval: int,
    total_steps: int,
    warmup_steps: int,
) -> None:
    # Argument validation
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval <= 0")

    if total_steps <= 0:
        raise ValueError("total_steps <= 0")

    if warmup_steps < 0:
        raise ValueError("warmup_steps < 0")

    if warmup_steps > total_steps:
        raise ValueError("warmup_steps > total_steps")

    if checkpoint_interval > total_steps:
        raise ValueError("checkpoint_interval > total_steps")

    # Training setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model("bert-base-uncased")
    dataloader = build_dataloader(
        batch_size=8,
        model_name="bert-base-uncased",
        dataset_name="wikitext"
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    checkpoint = Checkpoint(
        total_epochs=(total_steps // checkpoint_interval),
        checkpoint_pfs_dir=checkpoint_pfs_dir,
        append_job_id=True,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        checkpoint=checkpoint,
        device=device,
        checkpoint_interval=checkpoint_interval,
        checkpoint_local_dir=checkpoint_local_dir,
        enable_profiler=profile,
    )

    trainer.train(max_steps=total_steps)


if __name__ == "__main__":
    args = parse_args()

    try:
        main(
            checkpoint_pfs_dir=args.checkpoint_pfs_dir,
            checkpoint_local_dir=args.checkpoint_local_dir,
            profile=args.profile,
            checkpoint_interval=args.checkpoint_interval,
            total_steps=args.total_steps,
            warmup_steps=args.warmup_steps,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(1)
