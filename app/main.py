
from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.intent_agent import intent_agent
from app.agents.agentic_ai_graph import agentic_ai_graph
from app.services.conversation_logger import (
    log_conversation,
    update_feedback,
)


app = FastAPI(title="Agentic RAG MLOps")


class ChatRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback: int


class AgenticAIRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Agentic RAG MLOps system is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# Existing support chatbot endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    result = intent_agent.invoke({
        "text": request.text,
        "intent": "",
        "route": "",
        "context": [],
        "response": "",
    })

    conversation_id = log_conversation(
        user_query=request.text,
        intent=result["intent"],
        response=result["response"],
    )

    return {
        "conversation_id": conversation_id,
        "text": request.text,
        "intent": result["intent"],
        "route": result["route"],
        "response": result["response"],
    }


# Existing feedback endpoint
@app.post("/feedback")
def feedback(request: FeedbackRequest):
    update_feedback(
        conversation_id=request.conversation_id,
        feedback=request.feedback,
    )

    return {
        "conversation_id": request.conversation_id,
        "feedback": request.feedback,
        "message": "Feedback recorded successfully",
    }


# Agentic AI endpoint using LangGraph
@app.post("/agentic-ai")
def agentic_ai_chat(request: AgenticAIRequest):
    result = agentic_ai_graph.invoke({
        "question": request.text,
        "retrieved_context": [],
        "response": "",
        "action": "",
    })

    return {
        "text": request.text,
        "sources": [
            item["source"]
            for item in result["retrieved_context"]
        ],
        "response": result["response"],
        "action": result["action"],
    }
