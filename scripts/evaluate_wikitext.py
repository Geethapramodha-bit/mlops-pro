import math
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "data/processed/wikitext_eval.csv"

MODEL_NAME = "distilgpt2"

def main():
    print("Loading WikiText evaluation data...")
    df = pd.read_csv(DATA_PATH)

    print("Samples:", len(df))

    print("Loading Transformer model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    print("Evaluating...")

    for text in df["text"]:
        if not isinstance(text, str) or not text.strip():
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"]
            )

        num_tokens = inputs["input_ids"].size(1)

        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    print("\nWikiText Transformer Evaluation")
    print("--------------------------------")
    print("Model:", MODEL_NAME)
    print("Samples:", len(df))
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()