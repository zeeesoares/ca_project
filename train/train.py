import argparse
import os
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from train.model import build_model
from train.dataset import build_dataloader
from train.trainer import Trainer

from torch_ext.checkpoint import Checkpoint


def main(profile, total_steps, checkpoint_interval, test_baseline=False, checkpoint_pfs_dir="/tmp"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    job_id = os.environ.get('SLURM_JOB_ID', os.getpid())
    
    unique_pfs_dir = os.path.join(checkpoint_pfs_dir, f"job_{job_id}")
    
    if not os.path.exists(unique_pfs_dir):
        os.makedirs(unique_pfs_dir, exist_ok=True)

    model = build_model("bert-base-uncased")
    dataloader = build_dataloader(batch_size=8, model_name="bert-base-uncased", dataset_name="wikitext")

    optimizer = AdamW(model.parameters(), lr=5e-5)

    total_steps = total_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps,
    )

    checkpoint = Checkpoint(total_epochs=total_steps // checkpoint_interval)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        checkpoint=checkpoint,
        device=device,
        checkpoint_interval=checkpoint_interval,
        checkpoint_pfs_dir=unique_pfs_dir,
        enable_profiler=profile,
    )

    trainer.train(max_steps=total_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--total-steps", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--checkpoint-pfs-dir", type=str, default="/tmp")

    args = parser.parse_args()
    main(args.profile, args.total_steps, args.checkpoint_interval, checkpoint_pfs_dir=args.checkpoint_pfs_dir)
