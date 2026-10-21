from datasets import load_dataset, load_from_disk
import pandas as pd
import os

OUTPUT_PATH = "data/raw/intents.csv"
CLINC_PATH = "data/external/clinc150"


def main():
    # Load CLINC150 data
    dataset = load_from_disk(CLINC_PATH)

    # Load intent-name mapping
    intent_dataset = load_dataset(
        "DeepPavlov/clinc150",
        "intents"
    )

    intent_df = intent_dataset["intents"].to_pandas()

    # Create label -> intent mapping
    label_to_intent = dict(
        zip(
            intent_df["id"],
            intent_df["name"]
        )
    )

    rows = []

    # Convert train, validation and test data
    for split in ["train", "validation", "test"]:
        split_df = dataset[split].to_pandas()

        for _, row in split_df.iterrows():
            label = row["label"]

            # Skip OOD examples represented by None
            if pd.isna(label):
                continue

            label = int(label)

            rows.append({
                "text": row["utterance"],
                "intent": label_to_intent[label],
                "split": split
            })

    result = pd.DataFrame(rows)

    os.makedirs("data/raw", exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("CLINC150 conversion completed.")
    print("Dataset shape:", result.shape)
    print("\nColumns:")
    print(result.columns.tolist())

    print("\nIntent count:")
    print(result["intent"].nunique())

    print("\nSplit distribution:")
    print(result["split"].value_counts())

    print("\nFirst 10 rows:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()