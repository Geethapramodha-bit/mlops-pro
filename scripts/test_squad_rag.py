import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from ollama import chat


SQUAD_PATH = "data/processed/squad_eval.csv"
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "squad_evaluation"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "llama3:latest"


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a question-answering assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information,
respond with "NO ANSWER".

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
    df = pd.read_csv(SQUAD_PATH)

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    for i in range(5):
        question = df.iloc[i]["question"]
        reference = df.iloc[i]["answer"]

        question_embedding = embedding_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=1
        )

        retrieved_context = results["documents"][0][0]

        answer = generate_answer(
            question,
            retrieved_context
        )

        print("\n" + "=" * 70)

        print("Question:")
        print(question)

        print("\nReference answer:")
        print(
            reference
            if pd.notna(reference) and str(reference).strip()
            else "NO ANSWER"
        )

        print("\nRetrieved context:")
        print(retrieved_context[:500])

        print("\nLLM answer:")
        print(answer)


if __name__ == "__main__":
    main()