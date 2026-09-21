from app.rag.retriever import Retriever
from app.services.llm_service import generate_response


retriever = Retriever()

questions = [
    "I forgot my password. What should I do?",
    "My payment was declined. What should I do?",
    "I want a refund. How can I get it?"
]


for question in questions:

    context = retriever.search(question, top_k=3)

    answer = generate_response(
        question,
        context
    )

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    print("\nRETRIEVED CONTEXT:")
    for item in context:
        print("-", item)

    print("\nLLM ANSWER:")
    print(answer)