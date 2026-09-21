from fastapi import FastAPI
from pydantic import BaseModel

from app.services.intent_classifier import IntentClassifier


app = FastAPI(title="Agentic RAG MLOps")


# Load the trained intent classifier
intent_classifier = IntentClassifier()


class IntentRequest(BaseModel):
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


@app.post("/predict-intent")
def predict_intent(request: IntentRequest):

    intent = intent_classifier.predict(
        request.text
    )

    return {
        "text": request.text,
        "intent": intent
    }