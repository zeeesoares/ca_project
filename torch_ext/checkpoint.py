import torch
import time
from protocol.migrater.client import MigraterClient


class Checkpoint:

    def __init__(self):

        self.migrater = MigraterClient()

        # INFO instead of passing both path at each save, alternatively we can
        # pass one or both of them in the constructor

    def _save_handler(self,
                      checkpoint_local_path: str,
                      checkpoint_pfs_path: str) -> None:

        # Associate a timestamp with the checkpoint completion time,
        # which can be used by the migrater to determine checkpoint freshness.
        timestamp = time.time()

        # Send the checkpoint info to migrater service
        self.migrater.notify_checkpoint_saved(checkpoint_local_path,
                                              checkpoint_pfs_path,
                                              timestamp)

    def save(self,
             state: dict,
             checkpoint_local_path: str,
             checkpoint_pfs_path: str) -> None:

        torch.save(state, checkpoint_local_path)

        # Since `torch.save()` is synchronous, notify migrater immediately
        # NOTE if `torch.save()` is asynchronous in the future,
        # we need to pass the `_save_handler` as a callback to it
        self._save_handler(checkpoint_local_path, checkpoint_pfs_path)
