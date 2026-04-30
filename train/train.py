import argparse
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from train.model import build_model
from train.dataset import build_dataloader
from train.trainer import Trainer

from torch_ext.checkpoint import Checkpoint


def main(total_steps, checkpoint_interval):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model()
    dataloader = build_dataloader(batch_size=8)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    total_steps = total_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps,
    )

    checkpoint = Checkpoint()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        checkpoint=checkpoint,
        device=device,
        checkpoint_interval=checkpoint_interval,
        enable_profiler=args.profile,
    )

    trainer.train(max_steps=total_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=10)

    args = parser.parse_args()
    main(args.total_steps, args.checkpoint_interval)
