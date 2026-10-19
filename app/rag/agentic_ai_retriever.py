
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/processed/chroma_db"
COLLECTION_NAME = "agentic_ai_knowledge"


class AgenticAIRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=True,
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    def search(self, query: str, top_k: int = 3):
        if not query.strip():
            return []

        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved_items = []

        for i, document in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]

            retrieved_items.append({
                "text": document,
                "source": metadata.get("source", "Unknown"),
                "source_url": metadata.get("source_url", ""),
                "distance": results["distances"][0][i],
            })

        return retrieved_items
