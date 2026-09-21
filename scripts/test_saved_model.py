import joblib


# Load saved vectorizer and model
vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

model = joblib.load(
    "models/intent_classifier.pkl"
)

print("Model and vectorizer loaded successfully!")


# New user inputs
new_texts = [
    "I forgot my password",
    "My card payment was declined",
    "I want my money back",
    "The application is crashing"
]


# Convert text using saved vectorizer
new_text_tfidf = vectorizer.transform(
    new_texts
)


# Predict using saved model
predictions = model.predict(
    new_text_tfidf
)


print("\nPredictions:")

for text, prediction in zip(
    new_texts,
    predictions
):
    print(f"{text} -> {prediction}")