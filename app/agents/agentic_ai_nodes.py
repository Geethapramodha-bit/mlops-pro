
from app.rag.agentic_ai_retriever import AgenticAIRetriever
from app.agents.agentic_ai_state import AgenticAIState
from app.agents.agentic_ai_smalltalk import get_smalltalk_response
from app.services.agentic_ai_llm_service import is_follow_up

retriever = AgenticAIRetriever()

# Initial threshold; validate it against more test questions.
MAX_DISTANCE = 1.0


def retrieve_context(state: AgenticAIState) -> dict:
    question = state["question"].strip()

    # Handle empty input.
    if not question:
        return {
            "retrieved_context": [],
            "action": "fallback",
        }

    # Handle greetings and small talk before retrieval.
    smalltalk_reply = get_smalltalk_response(question)

    if smalltalk_reply is not None:
        return {
            "retrieved_context": [],
            "response": smalltalk_reply,
            "action": "smalltalk",
        }

    # Identify follow-ups that may rely on earlier messages.
    history = state.get("conversation_history", [])
    follow_up = is_follow_up(question) and bool(history)

    # Retrieve relevant knowledge-base documents.
    documents = retriever.search(question, top_k=3)

    # Keep only documents that pass the relevance threshold.
    relevant_documents = [
        document
        for document in documents
        if document["distance"] <= MAX_DISTANCE
    ]

    # A follow-up can use the previous conversation even when
    # the current wording does not retrieve new documents.
    if follow_up:
        return {
            "retrieved_context": relevant_documents,
            "action": "answer",
        }

    if not relevant_documents:
        return {
            "retrieved_context": [],
            "action": "fallback",
        }

    return {
        "retrieved_context": relevant_documents,
        "action": "answer",
    }
