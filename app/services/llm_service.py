import ollama

MODEL_NAME = "llama3:latest"


def generate_response(question: str, context: list[str]) -> str:
    context_text = "\n\n".join(context)

    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer,
say that you do not have enough information.

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
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]