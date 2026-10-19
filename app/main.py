
from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.intent_agent import intent_agent
from app.services.conversation_logger import (
    log_conversation,
    update_feedback,
)
from app.rag.agentic_ai_retriever import AgenticAIRetriever
from app.services.agentic_ai_llm_service import (
    generate_agentic_ai_response,
)


app = FastAPI(title="Agentic RAG MLOps")


class ChatRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback: int


class AgenticAIRequest(BaseModel):
    text: str


# Initialize the Agentic AI retriever
agentic_ai_retriever = AgenticAIRetriever()


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


# New Agentic AI RAG endpoint
@app.post("/agentic-ai")
def agentic_ai_chat(request: AgenticAIRequest):
    retrieved_context = agentic_ai_retriever.search(
        request.text,
        top_k=3,
    )

    response = generate_agentic_ai_response(
        request.text,
        retrieved_context,
    )

    return {
        "text": request.text,
        "sources": [
            item["source"] for item in retrieved_context
        ],
        "response": response,
    }
