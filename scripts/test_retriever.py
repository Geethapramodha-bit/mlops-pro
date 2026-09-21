from app.rag.retriever import Retriever


retriever = Retriever()

queries = [
    "I forgot my password",
    "My payment was declined",
    "I want a refund",
    "The application keeps crashing"
]

for query in queries:
    print("\nQuery:", query)

    results = retriever.search(query)

    for i, result in enumerate(results, start=1):
        print(f"{i}. {result}")