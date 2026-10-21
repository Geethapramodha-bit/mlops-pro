import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/raw/support_intents.csv"

MODEL_PATH = "models/support_intent_classifier.pkl"
VECTORIZER_PATH = "models/support_tfidf_vectorizer.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    print("Support Intent Model")
    print("--------------------")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel saved:", MODEL_PATH)
    print("Vectorizer saved:", VECTORIZER_PATH)


if __name__ == "__main__":
    main()