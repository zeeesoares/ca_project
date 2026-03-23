from abc import ABC, abstractmethod


class CheckpointManager(ABC):
    """
    Abstract interface for checkpointing strategies.
    """

    @abstractmethod
    def save(self, step, model, optimizer, scheduler):
        """Persist training state."""
        pass

    @abstractmethod
    def load(self, path, model, optimizer=None, scheduler=None):
        """Restore training state."""
        pass
