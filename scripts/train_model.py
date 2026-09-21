import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


DATA_PATH = "data/raw/intents.csv"


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["intent"]


    # -----------------------------
    # 2. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    # -----------------------------
    # 3. TF-IDF
    # -----------------------------
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)


    # -----------------------------
    # 4. Start MLflow experiment
    # -----------------------------
    mlflow.set_experiment("intent-classification")

    with mlflow.start_run():

        # -----------------------------
        # 5. Create and train model
        # -----------------------------
        model = LogisticRegression()

        model.fit(X_train_tfidf, y_train)


        # -----------------------------
        # 6. Save model and vectorizer
        # -----------------------------
        joblib.dump(
            vectorizer,
            "models/tfidf_vectorizer.pkl"
        )

        joblib.dump(
            model,
            "models/intent_classifier.pkl"
        )

        print("\nModel and vectorizer saved successfully!")


        # -----------------------------
        # 7. Predictions
        # -----------------------------
        y_pred = model.predict(X_test_tfidf)


        # -----------------------------
        # 8. Evaluation metrics
        # -----------------------------
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )


        # -----------------------------
        # 9. Log parameters to MLflow
        # -----------------------------
        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "tfidf_features",
            X_train_tfidf.shape[1]
        )

        mlflow.log_param(
            "training_samples",
            len(X_train)
        )

        mlflow.log_param(
            "testing_samples",
            len(X_test)
        )


        # -----------------------------
        # 10. Log metrics to MLflow
        # -----------------------------
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )


        # -----------------------------
        # 11. Log model to MLflow
        # -----------------------------
        mlflow.sklearn.log_model(
            model,
            name="intent_classifier"
        )


        # -----------------------------
        # 12. Print evaluation results
        # -----------------------------
        print("\nModel training completed!")

        print("\nAccuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)


        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )


        # -----------------------------
        # 13. Compare actual vs predicted
        # -----------------------------
        print("\nActual intents:")
        print(y_test.to_list())

        print("\nPredicted intents:")
        print(y_pred.tolist())


        # -----------------------------
        # 14. Test new user inputs
        # -----------------------------
        new_texts = [
            "I cannot sign into my account",
            "Someone charged my card",
            "Please return my money",
            "The website keeps crashing"
        ]


        # Convert new text using
        # the SAME TF-IDF vectorizer
        new_text_tfidf = vectorizer.transform(
            new_texts
        )


        # Predict intents
        new_predictions = model.predict(
            new_text_tfidf
        )


        print("\nNew user inputs:")

        for text, prediction in zip(
            new_texts,
            new_predictions
        ):
            print(
                f"{text} -> {prediction}"
            )


if __name__ == "__main__":
    main()