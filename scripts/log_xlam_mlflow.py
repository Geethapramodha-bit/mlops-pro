import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

RESULTS_PATH = "reports/xlam_agent_results.csv"

mlflow.set_experiment("xlam-agent-evaluation")

results = pd.read_csv(RESULTS_PATH)

total_expected_calls = results["expected_calls"].apply(
    lambda x: len(eval(x))
).sum()

tool_selection_accuracy = (
    results["correct_tools"].sum() / total_expected_calls
)

required_argument_accuracy = (
    results["correct_required_arguments"].sum() / total_expected_calls
)

exact_argument_accuracy = (
    results["correct_exact_arguments"].sum() / total_expected_calls
)

exact_tool_call_accuracy = (
    results["exact_calls"].sum() / total_expected_calls
)

complete_query_accuracy = (
    results["complete_query_match"].mean()
)

with mlflow.start_run(run_name="xlam-agent-final-evaluation"):

    mlflow.log_param(
        "dataset",
        "xLAM Function Calling 60K"
    )

    mlflow.log_param(
        "evaluation_samples",
        len(results)
    )

    mlflow.log_param(
        "model",
        "llama3:latest"
    )

    mlflow.log_param(
        "evaluation_type",
        "agentic function calling"
    )

    mlflow.log_metric(
        "tool_selection_accuracy",
        tool_selection_accuracy
    )

    mlflow.log_metric(
        "required_argument_accuracy",
        required_argument_accuracy
    )

    mlflow.log_metric(
        "exact_argument_accuracy",
        exact_argument_accuracy
    )

    mlflow.log_metric(
        "exact_tool_call_accuracy",
        exact_tool_call_accuracy
    )

    mlflow.log_metric(
        "complete_query_accuracy",
        complete_query_accuracy
    )

    mlflow.log_artifact(RESULTS_PATH)


print("xLAM MLflow logging completed.")
print(f"Tool selection accuracy: {tool_selection_accuracy:.4f}")
print(f"Required argument accuracy: {required_argument_accuracy:.4f}")
print(f"Exact argument accuracy: {exact_argument_accuracy:.4f}")
print(f"Exact tool call accuracy: {exact_tool_call_accuracy:.4f}")
print(f"Complete query accuracy: {complete_query_accuracy:.4f}")