import os
import shutil
import sys
import time
import torch
import uuid
from pathlib import Path

from src.protocol.migrater.client import MigraterClient


DEFAULT_CHECKPOINT_NAME = "checkpoint.pt"


def make_big_state(base_state: dict, size_gb: int = 10) -> dict:
    """
    Add a large padding tensor to a checkpoint state dictionary.

    This helper is mainly useful for testing checkpoint migration behavior with
    large checkpoint files. The input dictionary is modified in place by adding
    a "__padding__" key containing a float32 tensor of approximately size_gb
    GB.

    Arguments:
        base_state(dict): Checkpoint state dictionary to augment.
        size_gb(int): Approximate size, in gigabytes, of the padding tensor.

    Returns:
        dict: The same checkpoint state dictionary, with an added
            "__padding__" tensor.
    """

    num_elements = int(size_gb * 1e9 / 4)  # float32 = 4 bytes
    big_tensor = torch.ones(num_elements, dtype=torch.float32)

    base_state["__padding__"] = big_tensor
    return base_state


def derive_alt_checkpoint_path(checkpoint_pfs_path: str) -> str:
    """
    Derive an alternate checkpoint path by appending "_alt" to the original
    filename before the extension.

    Example:
        "/pfs/checkpoint.pt" -> "/pfs/checkpoint_alt.pt"

    Arguments:
        checkpoint_pfs_path(str): Original checkpoint PFS path.

    Returns:
        str: Derived alternate checkpoint PFS path.
    """

    path = Path(checkpoint_pfs_path)
    return str(path.with_name(f"{path.stem}_alt{path.suffix}"))


class Checkpoint:
    """
    Checkpoint manager that saves checkpoints locally and coordinates migration
    to a persistent file system.

    The class alternates between a primary and an alternate PFS destination
    path after each successful checkpoint handoff. This reduces the risk of
    overwriting the latest valid PFS checkpoint before a newer checkpoint has
    been safely handled.
    """

    def __init__(
        self,
        total_epochs: int,
        checkpoint_pfs_dir: str,
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
        append_job_id: bool = False,
    ) -> None:
        """
        Initialize checkpoint migration state.

        Arguments:
            total_epochs(int): Total number of checkpoints expected for the
                training run. This value is forwarded to the Migrater service
                so it can reason about checkpoint progress.
            checkpoint_pfs_dir(str): PFS directory where migrated checkpoints
                should be stored.
            checkpoint_name(str): Base filename used for the PFS checkpoint.
                Defaults to "checkpoint.pt".
            append_job_id(bool): If True, append the current SLURM job
                ID to checkpoint_name before deriving the primary and alternate
                PFS checkpoint paths. If SLURM_JOB_ID is not set, the filename
                is left unchanged.
        """

        self.migrater = MigraterClient()

        if append_job_id:
            checkpoint_name = self._append_job_id(checkpoint_name)

        self.checkpoint_pfs_dir = Path(checkpoint_pfs_dir)
        self.checkpoint_pfs_path = str(
            self.checkpoint_pfs_dir / checkpoint_name
        )
        self.checkpoint_pfs_path_alt = derive_alt_checkpoint_path(
            self.checkpoint_pfs_path
        )

        self.use_alt_pfs_path = False

        self.total_epochs = total_epochs
        self.epoch = 0

    def _append_job_id(self, filename: str) -> str:
        """
        Append the current SLURM job ID or process ID to the checkpoint
        filename.

        This can help avoid filename collisions when multiple jobs are writing
        to the same PFS checkpoint directory. If the SLURM_JOB_ID environment
        variable is not set, the current process ID is used as a fallback.

        Arguments:
            filename(str): Original checkpoint filename.

        Returns:
            str: Modified filename with the SLURM job ID or process ID appended
                before the extension.
        """

        job_id = os.getenv("SLURM_JOB_ID", os.getpid())

        path = Path(filename)
        return str(path.with_name(f"{path.stem}_{job_id}{path.suffix}"))

    def _fallback_save(
        self,
        checkpoint_local_path: str,
        checkpoint_pfs_path: str
    ) -> bool:
        """
        Copy a local checkpoint directly to PFS as an atomic fallback.

        This is used when the Migrater service cannot be notified. The copy is
        first written to a temporary file in the target directory, then
        promoted to the final checkpoint path with os.replace(). This avoids
        exposing a partially copied checkpoint at the final PFS path.

        Arguments:
            checkpoint_local_path(str): Local path of the checkpoint file.
            checkpoint_pfs_path(str): Target PFS path for the checkpoint file.

        Returns:
            bool: Whether the checkpoint was successfully copied to PFS.
        """

        tmp_target = None

        try:
            target = Path(checkpoint_pfs_path)
            target.parent.mkdir(parents=True, exist_ok=True)

            tmp_target = target.with_name(
                f".{target.name}.{uuid.uuid4().hex}.tmp"
            )

            shutil.copy2(checkpoint_local_path, tmp_target)
            os.replace(tmp_target, target)

            print(f"Checkpoint saved to PFS as a fallback: {target}")
            return True

        except Exception as copy_error:
            print(
                f"Error saving checkpoint to PFS as a fallback: {copy_error}",
                file=sys.stderr,
            )

            if tmp_target is not None:
                try:
                    Path(tmp_target).unlink(missing_ok=True)
                except Exception:
                    pass

            return False

    def _save_handler(
        self,
        checkpoint_local_path: str,
        checkpoint_pfs_path: str,
    ) -> bool:
        """
        Notify the Migrater that a local checkpoint is ready for migration.

        A timestamp and epoch number are sent together with the local
        checkpoint path and target PFS path. If the Migrater notification
        fails, this method falls back to copying the checkpoint directly to
        PFS.

        The internal epoch counter is only advanced if either the Migrater was
        successfully notified or the fallback copy succeeded.

        Arguments:
            checkpoint_local_path(str): Local path of the checkpoint file.
            checkpoint_pfs_path(str): Target PFS path for the checkpoint file.

        Returns:
            bool: Whether the checkpoint was successfully handed off to the
                Migrater or saved to PFS via fallback.
        """

        # Associate a timestamp with the checkpoint completion time,
        # which can be used by the Migrater to determine checkpoint freshness.
        timestamp = time.time()
        next_epoch = self.epoch + 1

        # Send the checkpoint info to Migrater service.
        try:
            self.migrater.notify_checkpoint_saved(
                checkpoint_local_path,
                checkpoint_pfs_path,
                timestamp,
                next_epoch,
                self.total_epochs
            )
            self.epoch = next_epoch
            return True

        except Exception as e:
            print(f"Error notifying Migrater: {e}", file=sys.stderr)

            # Fallback to direct copy if Migrater notification fails.
            if self._fallback_save(checkpoint_local_path, checkpoint_pfs_path):
                self.epoch = next_epoch
                return True

            return False

    def save(
        self,
        state: dict,
        checkpoint_local_path: str,
    ) -> None:
        """
        Save a checkpoint locally and request migration to PFS.

        The checkpoint is first written to checkpoint_local_path using
        torch.save(). After the local save completes, the Migrater service is
        notified so it can move or copy the checkpoint to PFS.

        The PFS target alternates between checkpoint.pt and checkpoint_alt.pt
        inside the configured PFS checkpoint directory.

        Arguments:
            state(dict): Checkpoint state to save.
            checkpoint_local_path(str): Local path where the checkpoint file is
                written before migration.

        Raises:
            RuntimeError: If the checkpoint could not be handed off to the
                Migrater and the fallback PFS copy also failed.
        """

        local_path = Path(checkpoint_local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(state, local_path)

        target_pfs_path = (
            self.checkpoint_pfs_path_alt
            if self.use_alt_pfs_path
            else self.checkpoint_pfs_path
        )

        # NOTE: if checkpoint saving becomes asynchronous in the future,
        # call `_save_handler` only after the local save has completed.
        if self._save_handler(str(local_path), target_pfs_path):
            self.use_alt_pfs_path = not self.use_alt_pfs_path
            return

        raise RuntimeError(
            f"Failed to migrate checkpoint to PFS path: {target_pfs_path}"
        )
