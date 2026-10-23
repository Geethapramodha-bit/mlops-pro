
from app.services.llm_service import generate_response

question = "I forgot my password. What should I do?"

context = [
    "For password problems, contact customer support "
    "for assistance with account recovery."
]

response = generate_response(question, context)

print("\nLLM Response:")
print(response)
