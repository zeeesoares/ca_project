import os
import sys
import time
import torch

from protocol.migrater.client import MigraterClient


class Checkpoint:

    def __init__(self):

        self.migrater = MigraterClient()

        # INFO instead of passing both paths at each save, alternatively we can
        # pass one or both of them in the constructor

    def _save_handler(self,
                      checkpoint_local_path: str,
                      checkpoint_pfs_path: str) -> None:

        # Associate a timestamp with the checkpoint completion time,
        # which can be used by the migrater to determine checkpoint freshness.
        timestamp = time.time()

        # Send the checkpoint info to migrater service
        try:
            self.migrater.notify_checkpoint_saved(checkpoint_local_path,
                                                  checkpoint_pfs_path,
                                                  timestamp)
        except Exception as e:
            print(f"Error notifying migrater: {e}")

            # We could implement a retry mechanism here in the future if needed
            # For now, lets save the checkpoint to the PFS directly to ensure it's not lost
            try:
                # Copy the checkpoint to PFS as a fallback
                if os.system(f"cp {checkpoint_local_path} {checkpoint_pfs_path}") != 0:
                    print(f"Error: Failed to copy checkpoint to PFS: {checkpoint_pfs_path}",
                          file=sys.stderr)
                else:
                    print(f"Checkpoint saved to PFS as a fallback: {checkpoint_pfs_path}")
            except Exception as e:
                print(f"Error saving checkpoint to PFS as a fallback: {e}")
                # Depending on the criticality, we might want to raise an exception here
                # or implement additional fallback mechanisms

    def save(self,
             state: dict,
             checkpoint_local_path: str,
             checkpoint_pfs_path: str) -> None:

        torch.save(state, checkpoint_local_path)

        # Since `torch.save()` is synchronous, notify migrater immediately
        # NOTE if `torch.save()` is asynchronous in the future,
        # we need to pass the `_save_handler` as a callback to it
        self._save_handler(checkpoint_local_path, checkpoint_pfs_path)
