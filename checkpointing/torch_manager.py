import os
import torch
from .interface import CheckpointManager


class TorchCheckpointManager(CheckpointManager):
    """
    Baseline checkpoint manager using PyTorch's torch.save
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save(self, step, model, optimizer, scheduler):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{step}.pt")
        state = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None
        }
        torch.save(state, checkpoint_path)

    def load(self, path, model, optimizer=None, scheduler=None):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler and checkpoint["scheduler_state"]:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint["step"]
