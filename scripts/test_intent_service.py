from app.services.intent_classifier import IntentClassifier


classifier = IntentClassifier()

test_messages = [
    "I forgot my password",
    "My payment was declined",
    "Please give me a refund",
    "The website is not working"
]


for message in test_messages:
    intent = classifier.predict(message)

    print(f"{message} -> {intent}")