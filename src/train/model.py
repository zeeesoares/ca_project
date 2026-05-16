from transformers import AutoModelForMaskedLM


def build_model(model_name="bert-base-uncased"):
    model_path = f"/tmp/{model_name}"

    try:
        print(f"Loading model from: {model_path}")
        return AutoModelForMaskedLM.from_pretrained(model_path)

    except Exception:
        print(f"Local model not found. Downloading {model_name}...")
        return AutoModelForMaskedLM.from_pretrained(model_name)
