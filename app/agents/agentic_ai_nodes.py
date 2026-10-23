
from app.rag.agentic_ai_retriever import AgenticAIRetriever
from app.agents.agentic_ai_state import AgenticAIState

retriever = AgenticAIRetriever()

# Initial threshold; validate it against more test questions.
MAX_DISTANCE = 1.0


def retrieve_context(state: AgenticAIState) -> dict:
    question = state["question"].strip()

    if not question:
        return {
            "retrieved_context": [],
            "action": "fallback",
        }

    documents = retriever.search(question, top_k=3)

    # Keep only documents that pass the relevance threshold.
    relevant_documents = [
        document
        for document in documents
        if document["distance"] <= MAX_DISTANCE
    ]

    if not relevant_documents:
        return {
            "retrieved_context": [],
            "action": "fallback",
        }

    return {
        "retrieved_context": relevant_documents,
        "action": "answer",
    }
