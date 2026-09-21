from app.agents.intent_agent import intent_agent


test_messages = [
    "I forgot my password",
    "My payment was declined",
    "I want my money back",
    "The website is crashing"
]


for message in test_messages:

    result = intent_agent.invoke({
        "text": message,
        "intent": "",
        "route": "",
        "context": [],
        "response": ""
    })

    print("\n" + "=" * 60)
    print("Query:", message)
    print("Intent:", result["intent"])
    print("Route:", result["route"])

    print("\nRetrieved Context:")
    for item in result["context"]:
        print("-", item)

    print("\nFinal LLM Response:")
    print(result["response"])