
import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# Prevent model and retriever initialization during API smoke tests.
mock_intent_agent_module = MagicMock()
mock_intent_agent_module.intent_agent = MagicMock()

sys.modules["app.agents.intent_agent"] = mock_intent_agent_module

from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Agentic RAG MLOps system is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

