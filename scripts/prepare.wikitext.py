from datasets import load_from_disk
import pandas as pd
import os

DATA_PATH = "data/external/wikitext2"
OUTPUT_PATH = "data/processed/wikitext_eval.csv"

def main():
    ds = load_from_disk(DATA_PATH)

    test_df = ds["test"].to_pandas()

    # Remove empty lines
    test_df = test_df[test_df["text"].str.strip() != ""]

    # Keep a small evaluation sample for development
    test_df = test_df.sample(
        n=min(500, len(test_df)),
        random_state=42
    ).reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)

    test_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("WikiText evaluation dataset created.")
    print("Samples:", len(test_df))
    print("Columns:", test_df.columns.tolist())
    print("\nFirst example:")
    print(test_df.iloc[0]["text"])


if __name__ == "__main__":
    main()