
from app.agents.agentic_ai_graph import agentic_ai_graph


def main():
    print("\n" + "=" * 45)
    print("     Agentic AI Knowledge Assistant")
    print("=" * 45)
    print("Ask me about:")
    print("  - Agentic AI and AI agents")
    print("  - Large Language Models (LLMs)")
    print("  - Transformers")
    print("  - Retrieval-Augmented Generation (RAG)")
    print("  - LangGraph")
    print("\nType 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in ("exit", "quit"):
                print("Assistant: Goodbye!")
                break

            if not question:
                print("Assistant: Please enter a question.\n")
                continue

            result = agentic_ai_graph.invoke({
                "question": question,
                "retrieved_context": [],
                "response": "",
                "action": "",
            })

            answer = result.get(
                "response",
                "I couldn't find enough relevant information "
                "in my knowledge base to answer that question."
            )

            print(f"\nAssistant: {answer}\n")

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}\n")


if __name__ == "__main__":
    main()
