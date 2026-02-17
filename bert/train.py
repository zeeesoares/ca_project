import time
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Classe para medir o tempo de checkpointing, Podemos extender o Trainer para funcionalidades novas.
class TrackerTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        start_time = time.time()
        
        super().save_model(output_dir, _internal_call)
        
        save_duration = time.time() - start_time
        checkpointing_log(save_duration)
        print(f"Checkpointing Time: {save_duration:.2f} seconds")

def checkpointing_log(time):
    with open("bert_checkpoints/checkpointing_log.txt", "a") as log_file:
        log_file.write(f"Checkpoint saved in {time:.2f} seconds\n")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm_probability=0.15
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_checkpoints",
    per_device_train_batch_size=8,
    save_strategy="steps",
    save_steps=10,             # Save every 50 steps to observe I/O impact
    save_total_limit=2,        # Limit checkpoints to save disk space
    num_train_epochs=1,
    report_to="none",          # Disable external logging for cleaner output
    logging_steps=10           # Log training loss more frequently
)

trainer = TrackerTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training and benchmarking...")
total_start_time = time.time()

train_results = trainer.train()

total_duration = time.time() - total_start_time

print("-" * 30)
print(f"Total Training Runtime: {total_duration:.2f} seconds")
print(f"Final Training Loss: {train_results.metrics['train_loss']:.4f}")
print("-" * 30)