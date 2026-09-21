from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.intent_agent import intent_agent
from app.services.conversation_logger import log_conversation


app = FastAPI(title="Agentic RAG MLOps")


class ChatRequest(BaseModel):
    text: str
    feedback: int = 0


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


@app.post("/chat")
def chat(request: ChatRequest):

    result = intent_agent.invoke({
        "text": request.text,
        "intent": "",
        "route": "",
        "context": [],
        "response": ""
    })

    log_conversation(
        user_query=request.text,
        intent=result["intent"],
        response=result["response"],
        feedback=request.feedback
    )

    return {
        "text": request.text,
        "intent": result["intent"],
        "route": result["route"],
        "response": result["response"]
    }