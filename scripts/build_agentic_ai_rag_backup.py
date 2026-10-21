import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("data/raw/agentic_ai_dataset.json")
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "agentic_ai_knowledge"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_text(text):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    documents = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    chunks = []
    metadatas = []
    ids = []

    for doc_index, item in enumerate(documents):
        for chunk_index, chunk in enumerate(chunk_text(item["text"])):
            chunks.append(chunk)
            metadatas.append({
                "source": item["source"],
                "source_url": item["source_url"],
                "document_index": doc_index,
                "chunk_index": chunk_index,
                "topic": "agentic_ai",
            })
            ids.append(f"agentic_doc_{doc_index}_chunk_{chunk_index}")

    if not chunks:
        raise ValueError("No text chunks were generated from the dataset.")

    print("Source documents:", len(documents))
    print("Chunks generated:", len(chunks))

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        local_files_only=True,
    )

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Previous Agentic AI collection replaced.")
    except Exception:
        print("No previous Agentic AI collection found.")

    collection = client.create_collection(name=COLLECTION_NAME)

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print("Collection:", COLLECTION_NAME)
    print("Chunks indexed:", collection.count())
    print("Embedding dimension:", len(embeddings[0]))
    print("Agentic AI RAG indexing completed.")


if __name__ == "__main__":
    main()
