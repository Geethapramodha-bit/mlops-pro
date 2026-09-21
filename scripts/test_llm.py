from app.services.llm_service import generate_response


prompt = """
You are a helpful customer support assistant.

Answer this question briefly:

I forgot my password. What should I do?
"""

response = generate_response(prompt)

print("\nLLM Response:")
print(response)