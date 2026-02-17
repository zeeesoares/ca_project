from datasets import load_dataset
from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


def build_dataloader(batch_size=8):

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized = dataset["train"].map(
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

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    return dataloader
