
import ollama


MODEL_NAME = "llama3:latest"


def generate_agentic_ai_response(
    question: str,
    retrieved_context: list[dict],
) -> str:
    if not question.strip():
        return "Please enter a question."

    if not retrieved_context:
        return "I could not find relevant information in the knowledge base."

    context_parts = []

    for item in retrieved_context:
        source = item.get("source", "Unknown")
        source_url = item.get("source_url", "")
        text = item.get("text", "")

        context_parts.append(
            f"Source: {source}\n"
            f"Source URL: {source_url}\n"
            f"Content:\n{text}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""
You are an AI assistant specializing in Agentic AI,
large language models, tools, and retrieval-augmented generation.

Answer the user's question using the provided context.
Explain concepts clearly and accurately.
Do not invent facts that are not supported by the context.
If the context is insufficient, state that clearly.
Do not claim to have executed tools or actions unless you actually have.

Context:
{context_text}

User question:
{question}

Answer:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]
