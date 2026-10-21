from typing import TypedDict


class AgenticAIState(TypedDict):
    question: str
    retrieved_context: list[dict]
    response: str
    action: str