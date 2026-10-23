
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.intent_agent import intent_agent
from app.agents.agentic_ai_graph import agentic_ai_graph
from app.services.conversation_logger import (
    log_conversation,
    update_feedback,
)

from app.api.auth import router as auth_router
from app.api.conversations import router as conversations_router
from app.api.auth import get_current_user
from app.services.database import Base, engine
from app.services import models
from app.services.models import User


app = FastAPI(title="Agentic RAG MLOps")


# Allow the React frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create database tables.
Base.metadata.create_all(bind=engine)


# Register API routers.
app.include_router(auth_router)
app.include_router(conversations_router)


class ChatRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback: int = Field(ge=0, le=1)


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


# Support chatbot endpoint.
# Authentication is required.
@app.post("/chat")
def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
):
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


# Feedback endpoint.
# Authentication is required.
@app.post("/feedback")
def feedback(
    request: FeedbackRequest,
    user: User = Depends(get_current_user),
):
    update_feedback(
        conversation_id=request.conversation_id,
        feedback=request.feedback,
    )

    return {
        "conversation_id": request.conversation_id,
        "feedback": request.feedback,
        "message": "Feedback recorded successfully",
    }


# Agentic AI endpoint using LangGraph.
# Authentication is required.
@app.post("/agentic-ai")
def agentic_ai_chat(
    request: AgenticAIRequest,
    user: User = Depends(get_current_user),
):
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
