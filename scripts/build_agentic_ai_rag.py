
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATASET_PATH = Path("data/raw/agentic_ai_dataset.json")
LOCAL_DOCS_DIR = Path("data/raw/agentic_ai")
CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "agentic_ai_knowledge"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40


def clean_text(text):
    # Remove HTML comments and license headers
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"^\s*!--Copyright.*?-->\s*",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove Markdown images
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)

    # Keep Markdown link text but remove its URL
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    # Remove HTML/MDX tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text):
    text = clean_text(text)
    sections = re.split(r"(?m)(?=^#{1,6}\s+)", text)

    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        words = section.split()
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


def load_documents():
    # Load the original Hugging Face dataset
    documents = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    # Add local Agentic AI knowledge files
    for file_path in sorted(LOCAL_DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8-sig").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        documents.append({
            "source": f"local/{file_path.name}",
            "source_url": "",
            "text": text,
        })

    return documents


def main():
    documents = load_documents()

    chunks = []
    metadatas = []
    ids = []

    for doc_index, item in enumerate(documents):
        document_chunks = chunk_text(item["text"])

        print(
            f"Document {doc_index + 1}: "
            f"{item['source']} -> {len(document_chunks)} chunks"
        )

        for chunk_index, chunk in enumerate(document_chunks):
            chunks.append(chunk)

            metadatas.append({
                "source": item["source"],
                "source_url": item.get("source_url", ""),
                "document_index": doc_index,
                "chunk_index": chunk_index,
                "topic": "agentic_ai",
            })

            ids.append(
                f"agentic_doc_{doc_index}_chunk_{chunk_index}"
            )

    if not chunks:
        raise ValueError("No text chunks were generated.")

    print("\nSource documents:", len(documents))
    print("Total chunks generated:", len(chunks))

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

    # Replace only the Agentic AI collection
    # Support and evaluation collections remain untouched.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Previous Agentic AI collection deleted.")
    except Exception:
        print("No previous Agentic AI collection found.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Agentic AI knowledge base"},
    )

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))

        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print("\nCollection:", COLLECTION_NAME)
    print("Chunks indexed:", collection.count())
    print("Embedding dimension:", len(embeddings[0]))
    print("Agentic AI RAG indexing completed.")


if __name__ == "__main__":
    main()
