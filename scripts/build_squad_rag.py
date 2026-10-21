import os

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


SQUAD_PATH = "data/processed/squad_eval.csv"
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "squad_evaluation"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    df = pd.read_csv(SQUAD_PATH)

    print("SQuAD evaluation data loaded.")
    print("Rows:", len(df))

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Delete an old evaluation collection if it exists.
    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("Existing SQuAD collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    documents = df["context"].tolist()

    ids = [
        f"squad_{i}"
        for i in range(len(documents))
    ]

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print("\nSQuAD RAG collection created.")
    print("Collection:", COLLECTION_NAME)
    print("Documents:", collection.count())


if __name__ == "__main__":
    main()