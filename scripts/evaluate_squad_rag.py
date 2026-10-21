import re

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from ollama import chat


SQUAD_PATH = "data/processed/squad_eval.csv"
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "squad_evaluation"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "llama3:latest"

# Development test only
SAMPLE_SIZE = 50


def normalize_text(text):
    """Normalize text for evaluation."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def token_f1(prediction, reference):
    """
    Calculate token-level F1.

    This compares the tokens in the generated answer
    with the tokens in the reference answer.
    """

    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens or not reference_tokens:
        return int(prediction_tokens == reference_tokens)

    common_tokens = set(prediction_tokens) & set(reference_tokens)

    if not common_tokens:
        return 0.0

    common_count = sum(
        min(
            prediction_tokens.count(token),
            reference_tokens.count(token)
        )
        for token in common_tokens
    )

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def generate_answer(question, context):
    """Generate an answer using only the retrieved context."""

    prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the provided context.

Important rules:

1. If the context directly contains the answer, answer the question.
2. Do not use outside knowledge.
3. Do not infer an answer from unrelated information.
4. If the context does not contain enough information to answer
   the question, respond with exactly:

NO ANSWER

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


def is_no_answer(answer):
    """Check whether the model abstained."""

    normalized = normalize_text(answer)

    return normalized in {
        "no answer",
        "noanswer"
    }


def main():

    print("Loading SQuAD evaluation data...")

    df = pd.read_csv(SQUAD_PATH)

    # Only use 5 questions during development.
    df = df.head(SAMPLE_SIZE)

    print("Questions to evaluate:", len(df))

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True
    )

    print("Loading ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    retrieval_results = []

    answer_f1_scores = []
    answerable_f1_scores = []

    correct_answers = 0

    answerable_count = 0
    unanswerable_count = 0

    correct_abstentions = 0

    # -----------------------------
    # Evaluate questions
    # -----------------------------

    for index, row in df.iterrows():

        question = row["question"]

        reference = row["answer"]

        if pd.isna(reference) or str(reference).strip() == "":
            reference = "NO ANSWER"
        else:
            reference = str(reference).strip()

        print("\n" + "=" * 70)
        print(f"Question {index + 1}/{len(df)}")
        print("=" * 70)

        print("\nQuestion:")
        print(question)

        # -----------------------------
        # Retrieval
        # -----------------------------

        question_embedding = embedding_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=1
        )

        retrieved_context = results["documents"][0][0]

        print("\nRetrieved context:")
        print(retrieved_context[:500])

        original_context = row["context"]

        retrieval_correct = (
            retrieved_context == original_context
        )

        retrieval_results.append(
            retrieval_correct
        )

        # -----------------------------
        # LLM answer
        # -----------------------------

        answer = generate_answer(
            question,
            retrieved_context
        )

        print("\nReference answer:")
        print(reference)

        print("\nLLM answer:")
        print(answer)

        print(
            "\nRetrieval correct:",
            retrieval_correct
        )

        # -----------------------------
        # Answer evaluation
        # -----------------------------

        reference_is_answerable = (
            normalize_text(reference)
            != "no answer"
        )

        model_abstained = is_no_answer(answer)

        if reference_is_answerable:

            answerable_count += 1

            f1 = token_f1(
                answer,
                reference
            )

            answerable_f1_scores.append(f1)
            answer_f1_scores.append(f1)

            # Consider a strong token overlap as a correct answer.
            answer_correct = f1 >= 0.5

            if answer_correct:
                correct_answers += 1

            print("\nQuestion type: ANSWERABLE")
            print("F1:", f1)
            print("Answer correct:", answer_correct)

        else:

            unanswerable_count += 1

            if model_abstained:
                correct_abstentions += 1
                correct_answers += 1

            print("\nQuestion type: UNANSWERABLE")
            print("Model abstained:", model_abstained)
            print(
                "Correct abstention:",
                model_abstained
            )

            # Unanswerable questions receive:
            # F1 = 1 if model correctly says NO ANSWER
            # F1 = 0 otherwise.
            f1 = 1.0 if model_abstained else 0.0

            answer_f1_scores.append(f1)

    # -----------------------------
    # Final metrics
    # -----------------------------

    retrieval_accuracy = (
        sum(retrieval_results)
        / len(retrieval_results)
    )

    overall_f1 = (
        sum(answer_f1_scores)
        / len(answer_f1_scores)
    )

    if len(answerable_f1_scores) > 0:

        answerable_f1 = (
            sum(answerable_f1_scores)
            / len(answerable_f1_scores)
        )

    else:

        answerable_f1 = 0.0

    overall_answer_accuracy = (
        correct_answers
        / len(df)
    )

    if unanswerable_count > 0:

        abstention_accuracy = (
            correct_abstentions
            / unanswerable_count
        )

    else:

        abstention_accuracy = 0.0

    # -----------------------------
    # Results
    # -----------------------------

    print("\n")
    print("=" * 70)
    print("SQuAD RAG EVALUATION RESULTS")
    print("=" * 70)

    print("\nDataset")
    print("-" * 30)

    print(
        "Questions evaluated :",
        len(df)
    )

    print(
        "Answerable          :",
        answerable_count
    )

    print(
        "Unanswerable        :",
        unanswerable_count
    )

    print("\nRetrieval")
    print("-" * 30)

    print(
        "Correct retrieval   :",
        sum(retrieval_results),
        "/",
        len(retrieval_results)
    )

    print(
        "Retrieval accuracy  :",
        round(retrieval_accuracy, 4)
    )

    print("\nAnswer Quality")
    print("-" * 30)

    print(
        "Overall answer accuracy :",
        round(overall_answer_accuracy, 4)
    )

    print(
        "Overall F1              :",
        round(overall_f1, 4)
    )

    print("\nAnswerable Questions")
    print("-" * 30)

    print(
        "Answerable F1           :",
        round(answerable_f1, 4)
    )

    print("\nUnanswerable Questions")
    print("-" * 30)

    print(
        "Correct abstentions     :",
        correct_abstentions,
        "/",
        unanswerable_count
    )

    print(
        "Abstention accuracy     :",
        round(abstention_accuracy, 4)
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()