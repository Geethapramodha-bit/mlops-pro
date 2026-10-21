
import ollama


MODEL_NAME = "llama3:latest"


def generate_agentic_ai_response(
    question: str,
    retrieved_context: list[dict],
) -> str:
    if not question.strip():
        return "Please enter a question."

    if not retrieved_context:
        return (
            "I could not find relevant information "
            "in the knowledge base."
        )

    context_parts = []

    for item in retrieved_context:
        source = item.get("source", "Unknown")
        source_url = item.get("source_url", "")
        text = item.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"Source: {source}\n"
            f"Source URL: {source_url}\n"
            f"Content:\n{text}"
        )

    if not context_parts:
        return (
            "I could not find relevant information "
            "in the knowledge base."
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""
You are a careful AI assistant specializing in Agentic AI,
large language models, AI agents, tools, and RAG.

STRICT RULES:
1. Answer only when the provided context contains information
   that directly supports the answer.
2. Retrieved documents may be irrelevant to the question.
   Do not assume that retrieved context is automatically relevant.
3. If the context does not contain enough relevant information,
   reply exactly:
   "I could not find enough relevant information in my knowledge base to answer this question."
4. Never invent facts, sources, images, variables, tools, or tool outputs.
5. Never generate pretend code showing tools being executed.
6. Do not claim to execute tools or perform actions.
7. Answer in clear, concise language.
8. Do not repeat the same point unnecessarily.
9. Use only the supplied context for factual claims.

RETRIEVED CONTEXT:
<context>
{context_text}
</context>

USER QUESTION:
<question>
{question}
</question>

Before answering, determine whether the context actually supports
the requested information. If it does not, use the exact fallback
response specified in rule 3.

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
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()
