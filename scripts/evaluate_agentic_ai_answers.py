
from app.rag.agentic_ai_retriever import AgenticAIRetriever
from app.services.agentic_ai_llm_service import (
    generate_agentic_ai_response,
)


TEST_CASES = [
    {
        "question": "What are AI agents and how do they use tools?",
        "expected": "answer",
    },
    {
        "question": "How can I create a custom tool for an AI agent?",
        "expected": "answer",
    },
    {
        "question": "How does retrieval augmented generation work?",
        "expected": "answer",
    },
    {
        "question": "What is the capital of France?",
        "expected": "refusal",
    },
]

FALLBACK = (
    "I could not find enough relevant information "
    "in my knowledge base to answer this question."
)


def main():
    retriever = AgenticAIRetriever()

    supported_total = 0
    supported_answered = 0
    unsupported_total = 0
    unsupported_refused = 0
    unsupported_answered = 0

    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        expected = test_case["expected"]

        print(f"\n{'=' * 60}")
        print(f"Test {index}: {question}")

        retrieved_context = retriever.search(
            question,
            top_k=3,
        )

        print("\nRetrieved sources:")
        for item in retrieved_context:
            print(f"- {item['source']}")

        answer = generate_agentic_ai_response(
            question,
            retrieved_context,
        )

        print("\nGenerated answer:")
        print(answer)

        refused = FALLBACK.lower() in answer.lower()

        if expected == "answer":
            supported_total += 1

            if not refused:
                supported_answered += 1
                print("\nEvaluation: ANSWER GENERATED")
            else:
                print("\nEvaluation: UNEXPECTED REFUSAL")

        else:
            unsupported_total += 1

            if refused:
                unsupported_refused += 1
                print("\nEvaluation: CORRECT REFUSAL")
            else:
                unsupported_answered += 1
                print("\nEvaluation: UNSUPPORTED ANSWER")

    print(f"\n{'=' * 60}")
    print("ANSWER EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    print(f"Total questions: {len(TEST_CASES)}")
    print(f"Expected-answer questions: {supported_total}")
    print(f"Answers generated: {supported_answered}")
    print(f"Expected-refusal questions: {unsupported_total}")
    print(f"Correct refusals: {unsupported_refused}")
    print(f"Unsupported answers: {unsupported_answered}")

    if supported_total:
        print(
            "Answer generation rate: "
            f"{supported_answered / supported_total:.2%}"
        )

    if unsupported_total:
        print(
            "Correct refusal rate: "
            f"{unsupported_refused / unsupported_total:.2%}"
        )


if __name__ == "__main__":
    main()
