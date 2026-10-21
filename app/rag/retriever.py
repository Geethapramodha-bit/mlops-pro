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

    def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 3
    ):
        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        query_args = {
            "query_embeddings": query_embedding,
            "n_results": top_k
        }

        if category:
            query_args["where"] = {
                "category": category
            }

        results = self.collection.query(**query_args)

        return results["documents"][0]