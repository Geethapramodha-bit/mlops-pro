import os
import re

import mlflow
import pandas as pd
from ollama import chat


DATA_PATH = "data/processed/squad_eval.csv"
MODEL_NAME = "llama3:latest"
REPORT_PATH = "reports/squad_results.csv"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, reference: str) -> int:
    return int(
        normalize_text(prediction)
        == normalize_text(reference)
    )


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens or not reference_tokens:
        return float(
            prediction_tokens == reference_tokens
        )

    common = set(prediction_tokens) & set(reference_tokens)

    if not common:
        return 0.0

    common_count = sum(
        min(
            prediction_tokens.count(token),
            reference_tokens.count(token)
        )
        for token in common
    )

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a question-answering system.

Answer the question using ONLY the provided context.

If the context does not contain enough information
to answer the question, respond with:
"NO ANSWER"

Context:
{context}

Question:
{question}

Answer:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


def main():
    df = pd.read_csv(DATA_PATH)

    print("SQuAD evaluation dataset loaded.")
    print("Total questions:", len(df))

    results = []

    for index, row in df.iterrows():

        prediction = generate_answer(
            question=row["question"],
            context=row["context"]
        )

        
        reference = row["answer"]

        # Pandas reads empty CSV cells as NaN.
        if pd.isna(reference) or str(reference).strip() == "":
              expected = "NO ANSWER"
        else:
            expected = str(reference).strip()

        em = exact_match(
            prediction,
            expected
        )

        f1 = token_f1(
            prediction,
            expected
        )

        results.append({
            "id": row["id"],
            "question": row["question"],
            "reference": expected,
            "prediction": prediction,
            "exact_match": em,
            "f1": f1
        })

        if (index + 1) % 10 == 0:
            print(
                f"Processed {index + 1}/{len(df)}"
            )

    results_df = pd.DataFrame(results)

    os.makedirs("reports", exist_ok=True)

    results_df.to_csv(
        REPORT_PATH,
        index=False
    )

    answerable = df["answer"].fillna("").astype(str).str.strip() != ""
    answerable_results = results_df[answerable]
    unanswerable_results = results_df[~answerable]

    answerable_em = (
        answerable_results["exact_match"].mean()
        if len(answerable_results)
        else 0.0
    )

    answerable_f1 = (
        answerable_results["f1"].mean()
        if len(answerable_results)
        else 0.0
    )

    unanswerable_em = (
        unanswerable_results["exact_match"].mean()
        if len(unanswerable_results)
        else 0.0
    )

    overall_em = results_df["exact_match"].mean()
    overall_f1 = results_df["f1"].mean()

    print("\nQA Evaluation")
    print("----------------")
    print("Overall Exact Match:", overall_em)
    print("Overall F1         :", overall_f1)

    print("\nAnswerable Questions")
    print("----------------")
    print("Count              :", len(answerable_results))
    print("Exact Match        :", answerable_em)
    print("F1                 :", answerable_f1)

    print("\nUnanswerable Questions")
    print("----------------")
    print("Count              :", len(unanswerable_results))
    print("Correct Abstention :", unanswerable_em)

    # MLflow
    mlflow.set_experiment("squad-qa-evaluation")

    with mlflow.start_run():

        mlflow.log_param(
            "dataset",
            "SQuAD 2.0"
        )

        mlflow.log_param(
            "model",
            MODEL_NAME
        )

        mlflow.log_param(
            "evaluation_samples",
            len(df)
        )

        mlflow.log_param(
            "answerable_samples",
            int(answerable.sum())
        )

        mlflow.log_param(
            "unanswerable_samples",
            int((~answerable).sum())
        )

        mlflow.log_metric(
            "overall_exact_match",
            overall_em
        )

        mlflow.log_metric(
            "overall_f1",
            overall_f1
        )

        mlflow.log_metric(
            "answerable_exact_match",
            answerable_em
        )

        mlflow.log_metric(
            "answerable_f1",
            answerable_f1
        )

        mlflow.log_metric(
            "unanswerable_correct_abstention",
            unanswerable_em
        )

    print("\nResults saved to:")
    print(REPORT_PATH)

    print("\nMLflow QA evaluation completed.")


if __name__ == "__main__":
    main()