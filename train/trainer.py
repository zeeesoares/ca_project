import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext


class Trainer:
    def __init__(self, model, optimizer, scheduler, dataloader, checkpoint,
                 device="cpu", checkpoint_interval=500, enable_profiler=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.checkpoint = checkpoint
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.enable_profiler = enable_profiler

    def train(self, max_steps):
        self.model.train()
        step = 0
        start_time = time.time()

        prof_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=max_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler_results'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        ) if self.enable_profiler else nullcontext()

        with prof_context as prof:
            while step < max_steps:
                for batch in self.dataloader:
                    with record_function("## TRAINING_STEP ##"):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
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
                            self.checkpoint.save(step, self.model, self.optimizer, self.scheduler)

                    if self.enable_profiler:
                        prof.step()

                    if step >= max_steps:
                        break

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f} seconds")
