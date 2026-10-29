
import os
import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# Set a test-only secret before importing the application.
os.environ["BOTPLAYER_SESSION_SECRET"] = "test-only-secret-for-ci"

# Prevent intent classifier initialization during API smoke tests.
mock_intent_agent_module = MagicMock()
mock_intent_agent_module.intent_agent = MagicMock()
sys.modules["app.agents.intent_agent"] = mock_intent_agent_module

# Prevent Agentic AI graph and retriever initialization during API smoke tests.
mock_agentic_ai_graph_module = MagicMock()
mock_agentic_ai_graph_module.agentic_ai_graph = MagicMock()
sys.modules["app.agents.agentic_ai_graph"] = mock_agentic_ai_graph_module

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
