
from app.rag.agentic_ai_retriever import AgenticAIRetriever


TEST_CASES = [
    {
        "question": "What are AI agents and how do they use tools?",
        "expected_keywords": ["transformers_agents.md", "agent.md"],
    },
    {
        "question": "How can I create a custom tool for an AI agent?",
        "expected_keywords": ["custom_tools.md"],
    },
    {
        "question": "How does retrieval augmented generation work?",
        "expected_keywords": ["rag.md"],
    },
]


def main():
    retriever = AgenticAIRetriever()

    total = len(TEST_CASES)
    passed = 0

    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        expected = test_case["expected_keywords"]

        results = retriever.search(question, top_k=3)

        retrieved_sources = [
            item["source"].split("/")[-1]
            for item in results
        ]

        matched = any(
            expected_source in retrieved_sources
            for expected_source in expected
        )

        if matched:
            passed += 1

        print(f"\nTest {index}: {question}")
        print("Expected source:", expected)
        print("Retrieved sources:", retrieved_sources)
        print("Result:", "PASS" if matched else "FAIL")

    print("\n========== Evaluation Summary ==========")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Source hit rate: {passed / total:.2%}")


if __name__ == "__main__":
    main()
