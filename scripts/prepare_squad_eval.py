from datasets import load_from_disk
import pandas as pd
import os

SQUAD_PATH = "data/external/squad2"
OUTPUT_PATH = "data/processed/squad_eval.csv"

SAMPLE_SIZE = 50


def main():
    dataset = load_from_disk(SQUAD_PATH)

    validation_df = dataset["validation"].to_pandas()

    evaluation_df = validation_df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)

    evaluation_df["answer"] = evaluation_df["answers"].apply(
        lambda x: x["text"][0] if len(x["text"]) > 0 else ""
    )

    result = evaluation_df[
        ["id", "title", "context", "question", "answer"]
    ].copy()

    os.makedirs("data/processed", exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("SQuAD evaluation dataset created.")
    print("Shape:", result.shape)
    print(
        "Answerable questions:",
        (result["answer"] != "").sum()
    )
    print(
        "Unanswerable questions:",
        (result["answer"] == "").sum()
    )

    print("\nFirst example:")
    print(result.iloc[0].to_string())


if __name__ == "__main__":
    main()