from transformers import BertForMaskedLM


def build_model():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return model
