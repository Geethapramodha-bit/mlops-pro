import mlflow

mlflow.set_experiment("wikitext-transformer-evaluation")

with mlflow.start_run(run_name="distilgpt2-wikitext-evaluation"):

    mlflow.log_param("dataset", "WikiText-2")
    mlflow.log_param("model", "distilgpt2")
    mlflow.log_param("evaluation_samples", 500)
    mlflow.log_param("task", "causal language modeling")

    mlflow.log_metric("average_loss", 4.3451)
    mlflow.log_metric("perplexity", 77.1012)

print("WikiText MLflow logging completed.")
print("Average Loss: 4.3451")
print("Perplexity: 77.1012")