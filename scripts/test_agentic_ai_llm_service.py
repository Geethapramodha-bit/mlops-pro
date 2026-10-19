from app.rag.agentic_ai_retriever import AgenticAIRetriever
from app.services.agentic_ai_llm_service import generate_agentic_ai_response


def main():
    question = "What are AI agents and how do they use tools?"

    retriever = AgenticAIRetriever()
    retrieved_context = retriever.search(question, top_k=3)

    print("\n--- Retrieved Documents ---")
    for item in retrieved_context:
        print("Source:", item["source"])
        print("Distance:", round(item["distance"], 4))

    print("\n--- Agentic AI Response ---")
    answer = generate_agentic_ai_response(
        question,
        retrieved_context,
    )
    print(answer)


if __name__ == "__main__":
    main()