
from langgraph.graph import StateGraph, START, END

from app.agents.agentic_ai_state import AgenticAIState
from app.agents.agentic_ai_nodes import retrieve_context
from app.services.agentic_ai_llm_service import (
    generate_agentic_ai_response,
)


def generate_answer(state: AgenticAIState) -> dict:
    response = generate_agentic_ai_response(
        state["question"],
        state["retrieved_context"],
        state.get("conversation_history", []),
    )

    return {
        "response": response,
        "action": "completed",
    }


def smalltalk_response(state: AgenticAIState) -> dict:
    return {
        "response": state["response"],
        "action": "completed",
    }


def fallback_response(state: AgenticAIState) -> dict:
    return {
        "response": (
            "I could not find enough relevant information "
            "in my knowledge base to answer this question."
        ),
        "action": "fallback",
    }


def build_agentic_ai_graph():
    graph = StateGraph(AgenticAIState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("smalltalk_response", smalltalk_response)
    graph.add_node("fallback_response", fallback_response)

    graph.add_edge(START, "retrieve_context")

    graph.add_conditional_edges(
        "retrieve_context",
        lambda state: state["action"],
        {
            "answer": "generate_answer",
            "smalltalk": "smalltalk_response",
            "fallback": "fallback_response",
        },
    )

    graph.add_edge("generate_answer", END)
    graph.add_edge("smalltalk_response", END)
    graph.add_edge("fallback_response", END)

    return graph.compile()


agentic_ai_graph = build_agentic_ai_graph()
