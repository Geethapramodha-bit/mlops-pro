import joblib


MODEL_PATH = "models/intent_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


class IntentClassifier:

    def __init__(self):
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.model = joblib.load(MODEL_PATH)

    def predict(self, text: str) -> str:

        text_tfidf = self.vectorizer.transform([text])

        prediction = self.model.predict(text_tfidf)

        return prediction[0]