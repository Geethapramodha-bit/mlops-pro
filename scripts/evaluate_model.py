import pandas as pd
import joblib
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


CONVERSATION_PATH = "data/conversations/conversations.csv"

MODEL_PATH = "models/support_intent_classifier.pkl"
VECTORIZER_PATH = "models/support_tfidf_vectorizer.pkl"


def evaluate_intent_model(df, model, vectorizer):

    texts = df["user_query"]
    actual_intents = df["intent"]

    X = vectorizer.transform(texts)
    predicted_intents = model.predict(X)

    accuracy = accuracy_score(
        actual_intents,
        predicted_intents
    )

    precision = precision_score(
        actual_intents,
        predicted_intents,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual_intents,
        predicted_intents,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual_intents,
        predicted_intents,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_intents": predicted_intents
    }


def evaluate_feedback(df):

    feedback_df = df[df["feedback"].notna()].copy()

    if len(feedback_df) == 0:
        return {
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "positive_rate": 0.0
        }

    feedback_df["feedback"] = pd.to_numeric(
        feedback_df["feedback"]
    )

    total_feedback = len(feedback_df)

    positive_feedback = (
        feedback_df["feedback"] == 1
    ).sum()

    negative_feedback = (
        feedback_df["feedback"] == 0
    ).sum()

    positive_rate = (
        positive_feedback / total_feedback
    )

    return {
        "total_feedback": total_feedback,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "positive_rate": positive_rate
    }


def main():

    df = pd.read_csv(CONVERSATION_PATH)

    print("Total conversations:", len(df))

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # --------------------------------
    # MLflow Configuration
    # --------------------------------

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "continuous-evaluation"
    )

    with mlflow.start_run():

        # --------------------------------
        # Intent Model Evaluation
        # --------------------------------

        intent_metrics = evaluate_intent_model(
            df,
            model,
            vectorizer
        )

        print("\nIntent Model Evaluation")
        print("-----------------------")

        print(
            "Accuracy :",
            intent_metrics["accuracy"]
        )

        print(
            "Precision:",
            intent_metrics["precision"]
        )

        print(
            "Recall   :",
            intent_metrics["recall"]
        )

        print(
            "F1-score :",
            intent_metrics["f1_score"]
        )

        # --------------------------------
        # Log Intent Metrics to MLflow
        # --------------------------------

        mlflow.log_metric(
            "evaluation_accuracy",
            intent_metrics["accuracy"]
        )

        mlflow.log_metric(
            "evaluation_precision",
            intent_metrics["precision"]
        )

        mlflow.log_metric(
            "evaluation_recall",
            intent_metrics["recall"]
        )

        mlflow.log_metric(
            "evaluation_f1_score",
            intent_metrics["f1_score"]
        )

        # --------------------------------
        # User Feedback Evaluation
        # --------------------------------

        feedback_metrics = evaluate_feedback(df)

        print("\nUser Feedback Evaluation")
        print("-----------------------")

        print(
            "Total feedback:",
            feedback_metrics["total_feedback"]
        )

        print(
            "Positive feedback:",
            feedback_metrics["positive_feedback"]
        )

        print(
            "Negative feedback:",
            feedback_metrics["negative_feedback"]
        )

        print(
            "Positive feedback rate:",
            feedback_metrics["positive_rate"]
        )

        # --------------------------------
        # Log Feedback Metrics to MLflow
        # --------------------------------

        mlflow.log_metric(
            "total_feedback",
            feedback_metrics["total_feedback"]
        )

        mlflow.log_metric(
            "positive_feedback",
            feedback_metrics["positive_feedback"]
        )

        mlflow.log_metric(
            "negative_feedback",
            feedback_metrics["negative_feedback"]
        )

        mlflow.log_metric(
            "positive_feedback_rate",
            feedback_metrics["positive_rate"]
        )

        # --------------------------------
        # Log Evaluation Parameters
        # --------------------------------

        mlflow.log_param(
            "total_conversations",
            len(df)
        )

        mlflow.log_param(
            "evaluation_type",
            "conversation_based"
        )

        mlflow.log_param(
            "model",
            "support_intent_classifier"
        )

        # --------------------------------
        # Predictions
        # --------------------------------

        print("\nIntent Predictions")
        print("------------------")

        for text, actual, predicted in zip(
            df["user_query"],
            df["intent"],
            intent_metrics["predicted_intents"]
        ):

            print(f"Query: {text}")
            print(f"Recorded intent : {actual}")
            print(f"Predicted intent: {predicted}")
            print()

        print("MLflow evaluation run completed.")


if __name__ == "__main__":
    main()