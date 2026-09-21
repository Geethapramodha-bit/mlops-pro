import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/processed/chroma_db"


class Retriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=True
        )
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name="support_knowledge"
        )

    def search(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        return results["documents"][0]