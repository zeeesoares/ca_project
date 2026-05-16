import time
import torch
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloader,
        checkpoint,
        device: str = "cpu",
        checkpoint_interval: int = 500,
        enable_profiler: bool = False,
        checkpoint_local_dir: str = "/tmp/",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.checkpoint = checkpoint
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.enable_profiler = enable_profiler
        self.checkpoint_local_dir = Path(checkpoint_local_dir)

    def train(self, max_steps: int) -> None:
        self.model.train()
        step = 0
        start_time = time.time()

        profiler_activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            profiler_activities.append(ProfilerActivity.CUDA)

        prof_context = profile(
            activities=profiler_activities,
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=max_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./logs/profiler_results"
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) if self.enable_profiler else nullcontext()

        with prof_context as prof:
            while step < max_steps:
                for batch in self.dataloader:
                    with record_function("## TRAINING_STEP ##"):
                        batch = {
                            k: v.to(self.device)
                            for k, v in batch.items()
                        }

                        outputs = self.model(**batch)
                        loss = outputs.loss
                        loss.backward()

                        self.optimizer.step()

                        if self.scheduler:
                            self.scheduler.step()

                        self.optimizer.zero_grad()

                    step += 1

                    if step % 50 == 0:
                        print(f"Step {step} | Loss: {loss.item():.4f}")

                    if step % self.checkpoint_interval == 0:
                        print(f"Saving checkpoint at step {step}")

                        with record_function("## CHECKPOINT_SAVE ##"):
                            state = {
                                "step": step,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict":
                                    self.optimizer.state_dict(),
                                "scheduler_state_dict":
                                    self.scheduler.state_dict()
                                    if self.scheduler else None
                            }

                            self.checkpoint.save(
                                state=state,
                                checkpoint_local_path=(
                                    self._checkpoint_local_path_for_step(step)
                                ),
                            )

                    if self.enable_profiler:
                        prof.step()

                    if step >= max_steps:
                        break

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f} seconds")

    def _checkpoint_local_path_for_step(self, step: int) -> str:
        """
        Return the local checkpoint file path for a given training step.

        Example:
            /tmp/ -> /tmp/checkpoint_100.pt
            /tmp/checkpoints -> /tmp/checkpoints/checkpoint_100.pt
        """

        return str(self.checkpoint_local_dir / f"checkpoint_{step}.pt")
