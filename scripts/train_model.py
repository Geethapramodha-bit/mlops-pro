
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from app.mlops.evaluation_gate import check_evaluation_gate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


DATA_PATH = "data/raw/intents.csv"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATH = "models/intent_classifier.pkl"


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    X_train = train_df["text"]
    y_train = train_df["intent"]

    X_validation = validation_df["text"]
    y_validation = validation_df["intent"]

    X_test = test_df["text"]
    y_test = test_df["intent"]

    print("Dataset loaded")
    print("----------------")
    print("Training samples   :", len(train_df))
    print("Validation samples :", len(validation_df))
    print("Test samples       :", len(test_df))
    print("Number of intents  :", df["intent"].nunique())

    # --------------------------------------------------
    # 2. TF-IDF
    # --------------------------------------------------
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_validation_tfidf = vectorizer.transform(X_validation)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTF-IDF")
    print("----------------")
    print("Training matrix   :", X_train_tfidf.shape)
    print("Validation matrix :", X_validation_tfidf.shape)
    print("Test matrix       :", X_test_tfidf.shape)
    print("Vocabulary size   :", len(vectorizer.vocabulary_))

    # --------------------------------------------------
    # 3. MLflow experiment
    # --------------------------------------------------
    # Use SQLite instead of the deprecated filesystem backend.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("intent-classification")

    with mlflow.start_run():

        # --------------------------------------------------
        # 4. Train model
        # --------------------------------------------------
        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(
            X_train_tfidf,
            y_train
        )

        print("\nModel training completed.")

        # --------------------------------------------------
        # 5. Validation evaluation
        # --------------------------------------------------
        validation_predictions = model.predict(
            X_validation_tfidf
        )

        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )

        validation_precision = precision_score(
            y_validation,
            validation_predictions,
            average="weighted",
            zero_division=0
        )

        validation_recall = recall_score(
            y_validation,
            validation_predictions,
            average="weighted",
            zero_division=0
        )

        validation_f1 = f1_score(
            y_validation,
            validation_predictions,
            average="weighted",
            zero_division=0
        )

        print("\nValidation Results")
        print("----------------")
        print("Accuracy :", validation_accuracy)
        print("Precision:", validation_precision)
        print("Recall   :", validation_recall)
        print("F1-score :", validation_f1)
        # Model evaluation gate
        mlflow.log_metric("validation_accuracy", validation_accuracy)
        mlflow.log_metric("validation_precision", validation_precision)
        mlflow.log_metric("validation_recall", validation_recall)
        mlflow.log_metric("validation_f1", validation_f1)

        try:
            check_evaluation_gate(validation_accuracy, validation_f1)
            mlflow.set_tag("evaluation_gate", "passed")
            print("\nEvaluation gate PASSED.")
        except ValueError as exc:
            mlflow.set_tag("evaluation_gate", "failed")
            mlflow.set_tag("evaluation_gate_reason", str(exc))
            print(f"\n{exc}")
            raise

        # --------------------------------------------------
        # 6. Final test evaluation
        # --------------------------------------------------
        test_predictions = model.predict(
            X_test_tfidf
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_precision = precision_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0
        )

        test_f1 = f1_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0
        )

        print("\nTest Results")
        print("----------------")
        print("Accuracy :", test_accuracy)
        print("Precision:", test_precision)
        print("Recall   :", test_recall)
        print("F1-score :", test_f1)

        # --------------------------------------------------
        # 7. Classification report
        # --------------------------------------------------
        print("\nClassification Report")
        print("----------------")
        print(
            classification_report(
                y_test,
                test_predictions,
                zero_division=0
            )
        )

        # --------------------------------------------------
        # 9. MLflow parameters
        # --------------------------------------------------
        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "vectorizer",
            "TF-IDF"
        )

        mlflow.log_param(
            "tfidf_features",
            X_train_tfidf.shape[1]
        )

        mlflow.log_param(
            "training_samples",
            len(train_df)
        )

        mlflow.log_param(
            "validation_samples",
            len(validation_df)
        )

        mlflow.log_param(
            "testing_samples",
            len(test_df)
        )

        mlflow.log_param(
            "number_of_intents",
            df["intent"].nunique()
        )

        # --------------------------------------------------
        # 11. MLflow test metrics
        # --------------------------------------------------
        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "test_precision",
            test_precision
        )

        mlflow.log_metric(
            "test_recall",
            test_recall
        )

        mlflow.log_metric(
            "test_f1",
            test_f1
        )

        # --------------------------------------------------
        # 12. Log model
        # --------------------------------------------------
        mlflow.sklearn.log_model(
            model,
            name="intent_classifier"
        )


        # 13. Save model
        # --------------------------------------------------
        os.makedirs("models", exist_ok=True)

        joblib.dump(
            vectorizer,
            VECTORIZER_PATH
        )

        joblib.dump(
            model,
            MODEL_PATH
        )

        print("\nModel saved:")
        print(VECTORIZER_PATH)
        print(MODEL_PATH)

        # --------------------------------------------------
        print("\nMLflow run completed successfully.")


if __name__ == "__main__":
    main()




