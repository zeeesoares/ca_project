import os
import torch
from .decorator import CheckpointDecorator


class CompressionCheckpointWrapper(CheckpointDecorator):
    """
    Simple checkpoint compression:
    Converts FP32 model weights to FP16 before saving.
    """

    def save(self, step, model, optimizer, scheduler):

        # Get original state
        original_model_state = model.state_dict()

        compressed_model_state = {}

        for name, tensor in original_model_state.items():

            if tensor.dtype == torch.float32:
                compressed_model_state[name] = tensor.half()
            else:
                compressed_model_state[name] = tensor

        state = {
            "step": step,
            "model_state": compressed_model_state,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None
        }

        save_path = os.path.join(
            self.manager.save_dir,
            f"checkpoint_{step}.pt"
        )

        torch.save(state, save_path)

    def load(self, path, model, optimizer=None, scheduler=None):

        checkpoint = torch.load(path)

        compressed_state = checkpoint["model_state"]
        restored_state = {}

        for name, tensor in compressed_state.items():

            if tensor.dtype == torch.float16:
                restored_state[name] = tensor.float()
            else:
                restored_state[name] = tensor

        model.load_state_dict(restored_state)

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        if scheduler and checkpoint["scheduler_state"]:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint["step"]
