from app.rag.agentic_ai_retriever import AgenticAIRetriever
from app.agents.agentic_ai_state import AgenticAIState


retriever = AgenticAIRetriever()


def retrieve_context(state: AgenticAIState) -> dict:
    question = state["question"]

    documents = retriever.search(
        question,
        top_k=3,
    )

    return {
        "retrieved_context": documents,
        "action": "answer" if documents else "fallback",
    }