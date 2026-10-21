from datasets import load_from_disk
import pandas as pd
import json
import os


XLAM_PATH = "data/external/xlam_function_calling_60k"
OUTPUT_PATH = "data/processed/xlam_agent_eval.csv"
SAMPLE_SIZE = 100


def main():

    print("Loading xLAM dataset...")

    dataset = load_from_disk(XLAM_PATH)

    test_df = dataset["test"].to_pandas()

    print("Test samples:", len(test_df))

    evaluation_df = (
        test_df
        .sample(n=SAMPLE_SIZE, random_state=42)
        .reset_index(drop=True)
    )

    rows = []

    for _, row in evaluation_df.iterrows():

        answers = json.loads(row["answers"])
        tools = json.loads(row["tools"])

        rows.append({
            "id": row["id"],
            "query": row["query"],

            # Store ALL available tool definitions
            "available_tools": json.dumps(tools),

            # Store ALL expected tool calls
            "expected_calls": json.dumps(answers)
        })

    result = pd.DataFrame(rows)

    os.makedirs("data/processed", exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nEvaluation dataset created.")

    print("Samples:", len(result))

    print(
        "Average expected tool calls:",
        result["expected_calls"]
        .apply(lambda x: len(json.loads(x)))
        .mean()
    )

    print(
        "Maximum expected tool calls:",
        result["expected_calls"]
        .apply(lambda x: len(json.loads(x)))
        .max()
    )

    print("\nColumns:")
    print(result.columns.tolist())

    print("\nFirst example:")
    print(result.iloc[0]["query"])

    print("\nExpected calls:")
    print(
        json.dumps(
            json.loads(result.iloc[0]["expected_calls"]),
            indent=2
        )
    )


if __name__ == "__main__":
    main()