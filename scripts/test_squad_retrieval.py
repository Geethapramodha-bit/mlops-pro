import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


SQUAD_PATH = "data/processed/squad_eval.csv"
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "squad_evaluation"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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

    # Test the first 5 questions.
    for i in range(5):
        question = df.iloc[i]["question"]
        original_context = df.iloc[i]["context"]

        question_embedding = embedding_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=1
        )

        retrieved_context = results["documents"][0][0]

        print("\n" + "=" * 70)
        print("Question:")
        print(question)

        print("\nOriginal context starts with:")
        print(original_context[:300])

        print("\nRetrieved context starts with:")
        print(retrieved_context[:300])

        print(
            "\nCorrect context retrieved:",
            retrieved_context == original_context
        )

        print(
            "Distance:",
            results["distances"][0][0]
        )


if __name__ == "__main__":
    main()