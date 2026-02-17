from .interface import CheckpointManager


class CheckpointDecorator(CheckpointManager):
    """
    Base class for checkpoint decorators / wrappers.
    """

    def __init__(self, manager: CheckpointManager):
        self.manager = manager

    def save(self, step, model, optimizer, scheduler):
        self.manager.save(step, model, optimizer, scheduler)

    def load(self, path, model, optimizer=None, scheduler=None):
        return self.manager.load(path, model, optimizer, scheduler)
