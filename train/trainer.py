import time


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloader,
        checkpoint,
        device="cpu",
        checkpoint_interval=500,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.checkpoint = checkpoint
        self.device = device
        self.checkpoint_interval = checkpoint_interval

    def train(self, max_steps):

        self.model.train()
        step = 0

        start_time = time.time()

        while step < max_steps:
            for batch in self.dataloader:

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
                    self.checkpoint.save(
                        step,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                    )

                if step >= max_steps:
                    break

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f} seconds")
