from .decorator import CheckpointDecorator


class IncrementalCheckpointWrapper(CheckpointDecorator):
    """
    Saves only parameter differences (delta checkpointing)
    """

    def __init__(self, manager):
        super().__init__(manager)
        self.prev_state = None

    def save(self, step, model, optimizer, scheduler):
        current_state = model.state_dict()
        delta_state = {}

        if self.prev_state is None:
            delta_state = current_state
        else:
            for k in current_state:
                delta_state[k] = current_state[k] - self.prev_state[k]

        # Temporarily replace model state with delta
        model.load_state_dict(delta_state)
        self.manager.save(step, model, optimizer, scheduler)
        model.load_state_dict(current_state)

        self.prev_state = current_state
