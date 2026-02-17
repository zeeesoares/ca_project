import threading
from .decorator import CheckpointDecorator


class AsyncCheckpointWrapper(CheckpointDecorator):
    """
    Makes any CheckpointManager async (non-blocking save)
    """

    def save(self, step, model, optimizer, scheduler):
        def async_save():
            self.manager.save(step, model, optimizer, scheduler)
        thread = threading.Thread(target=async_save)
        thread.start()
