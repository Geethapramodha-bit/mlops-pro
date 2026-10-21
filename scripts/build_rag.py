import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = "data/raw/knowledge_base.txt"
CHROMA_PATH = "data/processed/chroma_db"


def load_documents():
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    sections = text.split("\n\n")

    documents = []
    categories = []

    current_category = None

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if section.endswith("SUPPORT"):
            current_category = section.replace(" SUPPORT", "").lower()
            continue

        documents.append(section)
        categories.append(current_category)

    return documents, categories


def main():
    documents, categories = load_documents()

    print("Documents loaded:", len(documents))
    print("Categories:", categories)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embedding_model.encode(documents).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Remove the old collection so stale documents are deleted
    try:
        client.delete_collection("support_knowledge")
        print("Old ChromaDB collection deleted.")
    except Exception:
        print("No existing collection to delete.")

    collection = client.create_collection(
        name="support_knowledge"
    )

    ids = [f"doc_{i}" for i in range(len(documents))]

    metadatas = [
        {"category": category}
        for category in categories
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Documents stored in ChromaDB:", collection.count())
    print("RAG database created successfully!")


if __name__ == "__main__":
    main()