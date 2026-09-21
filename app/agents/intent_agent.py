from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.services.intent_classifier import IntentClassifier
from app.rag.retriever import Retriever
from app.services.llm_service import generate_response


class AgentState(TypedDict):
    text: str
    intent: str
    route: str
    context: list[str]
    response: str


classifier = IntentClassifier()
retriever = Retriever()


def classify_intent(state: AgentState):
    intent = classifier.predict(state["text"])
    return {"intent": intent}


def route_by_intent(state: AgentState):
    intent = state["intent"]

    if intent == "account":
        route = "account"
    elif intent == "payment":
        route = "payment"
    elif intent == "refund":
        route = "refund"
    elif intent == "technical":
        route = "technical"
    else:
        route = "unknown"

    return {"route": route}


def retrieve_context(state: AgentState):
    context = retriever.search(
        state["text"],
        top_k=3
    )

    return {"context": context}


def generate_answer(state: AgentState):
    response = generate_response(
        state["text"],
        state["context"]
    )

    return {"response": response}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("route_by_intent", route_by_intent)
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_edge("classify_intent", "route_by_intent")
graph_builder.add_edge("route_by_intent", "retrieve_context")
graph_builder.add_edge("retrieve_context", "generate_answer")
graph_builder.add_edge("generate_answer", END)

intent_agent = graph_builder.compile()