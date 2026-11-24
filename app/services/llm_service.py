
import os
import ollama
import httpx


MODEL_NAME = "llama3:latest"

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "ollama"
).strip().lower()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

ollama_client = ollama.Client(host=OLLAMA_HOST)


def generate_response(question: str, context: list[str]) -> str:
    context_text = "\n\n".join(context)

    prompt = f"""
You are a customer support assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that you do not have enough information.

Context:
{context_text}

User question:
{question}

Answer:
"""

    try:
        if LLM_PROVIDER == "groq":
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError("GROQ_API_KEY is missing.")

            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "temperature": 0,
                },
                timeout=90.0,
            )

            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"].strip()

        elif LLM_PROVIDER == "ollama":
            response = ollama_client.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options={"temperature": 0},
            )

            answer = response["message"]["content"].strip()

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
            )

        return answer or "I do not have enough information to answer."

    except Exception as error:
        print(f"Support chatbot {LLM_PROVIDER} error: {error}")

        return (
            "The AI model is temporarily unavailable. "
            "Please try again later."
        )
