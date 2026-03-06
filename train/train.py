import argparse
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from train.model import build_model
from train.dataset import build_dataloader
from train.trainer import Trainer

from checkpointing.torch_manager import TorchCheckpointManager
from checkpointing.async_wrapper import AsyncCheckpointWrapper
from checkpointing.compression_wrapper import CompressionCheckpointWrapper


def build_checkpoint_stack(save_dir,
                           use_async=False,
                           use_compression=False):

    checkpoint = TorchCheckpointManager(save_dir)

    if use_compression:
        checkpoint = CompressionCheckpointWrapper(checkpoint)

    if use_async:
        checkpoint = AsyncCheckpointWrapper(checkpoint)

    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-async", action="store_true")
    parser.add_argument("--enable-compression", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model()
    dataloader = build_dataloader(batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    total_steps = 50
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps,
    )

    checkpoint = build_checkpoint_stack(
        save_dir="./checkpoints",
        use_async=args.enable_async,
        use_compression=args.enable_compression,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        checkpoint=checkpoint,
        device=device,
        checkpoint_interval=total_steps // 5,
        enable_profiler=args.profile,
    )

    trainer.train(max_steps=total_steps)


if __name__ == "__main__":
    main()
