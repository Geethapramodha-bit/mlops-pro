import mlflow


def main():

    mlflow.set_experiment("squad-rag-evaluation")

    with mlflow.start_run():

        # Parameters
        mlflow.log_param("dataset", "SQuAD 2.0")
        mlflow.log_param("evaluation_samples", 50)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("llm_model", "llama3:latest")
        mlflow.log_param("vector_database", "ChromaDB")

        # Metrics from the completed 50-question RAG evaluation
        mlflow.log_metric(
            "retrieval_accuracy",
            0.92
        )

        mlflow.log_metric(
            "overall_answer_accuracy",
            0.54
        )

        mlflow.log_metric(
            "overall_f1",
            0.5002
        )

        mlflow.log_metric(
            "answerable_f1",
            0.4832
        )

        mlflow.log_metric(
            "abstention_accuracy",
            0.5238
        )

        print("RAG evaluation metrics logged to MLflow.")


if __name__ == "__main__":
    main()