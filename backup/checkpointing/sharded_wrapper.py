import os
import torch
from .decorator import CheckpointDecorator
import torch.distributed as dist


class ShardedCheckpointWrapper(CheckpointDecorator):
    """
    Saves only local shard in distributed training.
    """

    def save(self, step, model, optimizer, scheduler):
        rank = dist.get_rank() if dist.is_initialized() else 0
        save_dir = os.path.join(self.manager.save_dir, f"rank_{rank}")
        os.makedirs(save_dir, exist_ok=True)

        checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
        state = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None
        }
        torch.save(state, checkpoint_path)
