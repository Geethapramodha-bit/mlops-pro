from app.services.agentic_ai_llm_service import (
    is_follow_up,
    resolve_follow_up,
)


def test_recognizes_follow_up_question():
    assert is_follow_up("Explain the second component.") is True
    assert is_follow_up("What about the other one?") is True
    assert is_follow_up("Give an example.") is True


def test_rejects_independent_question():
    assert is_follow_up("What is Agentic AI?") is False
    assert is_follow_up("Explain machine learning.") is False


def test_resolves_other_component():
    history = [
        {
            "role": "user",
            "content": "What are the main components of Agentic AI?",
        },
        {
            "role": "assistant",
            "content": "The main components are AI agents and the tools they use.",
        },
        {
            "role": "user",
            "content": "Explain the second component in more detail.",
        },
        {
            "role": "assistant",
            "content": "The second component is tools, including databases, APIs, and algorithms.",
        },
    ]

    result = resolve_follow_up("What about the other one?", history)

    assert "AI agents" in result
    assert "tools component" in result


def test_empty_history_preserves_question():
    question = "What about the other one?"

    assert resolve_follow_up(question, []) == question


def test_unrelated_history_preserves_question():
    history = [
        {
            "role": "user",
            "content": "What is machine learning?",
        },
        {
            "role": "assistant",
            "content": "Machine learning allows systems to learn from data.",
        },
    ]

    question = "What about the other one?"

    assert resolve_follow_up(question, history) == question
