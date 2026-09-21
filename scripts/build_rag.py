import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = "data/raw/knowledge_base.txt"
CHROMA_PATH = "data/processed/chroma_db"


def load_documents():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    documents = [
        doc.strip()
        for doc in text.split("\n\n")
        if doc.strip()
    ]

    return documents


def main():
    documents = load_documents()

    print("Documents loaded:", len(documents))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embedding_model.encode(documents).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name="support_knowledge"
    )

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print("Documents stored in ChromaDB:", collection.count())
    print("RAG database created successfully!")


if __name__ == "__main__":
    main()