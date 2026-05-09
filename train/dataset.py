from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def build_dataloader(batch_size=8, model_name="bert-base-uncased", dataset_name="wikitext"):
    base_path = "/tmp"
    tokenizer_path = f"{base_path}/models/{model_name}"
    dataset_path = f"{base_path}/datasets/{dataset_name}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(tokenizer_path)

    try:
        raw_dataset = load_from_disk(dataset_path)
    except Exception:
        print(f"A descarregar dataset {dataset_name}...")
        raw_dataset = load_dataset(dataset_name, "wikitext-2-raw-v1")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = raw_dataset["train"].map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    tokenized.set_format("torch")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    return DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)