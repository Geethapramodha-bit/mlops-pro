import json
import re
import pandas as pd
import mlflow
from ollama import chat


DATA_PATH = "data/processed/xlam_agent_eval.csv"
MODEL_NAME = "llama3:latest"
SAMPLE_SIZE = 10


def extract_tool_calls(text):
    """
    Extract tool calls from the LLM response.

    Expected format:

    [
        {
            "tool": "tool_name",
            "arguments": {}
        }
    ]
    """

    # Remove markdown code fences if the model adds them
    text = re.sub(r"```json|```", "", text).strip()

    # Try JSON array first
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            return [parsed]

    except json.JSONDecodeError:
        pass

    # Fallback: extract individual JSON objects
    decoder = json.JSONDecoder()
    objects = []

    index = 0

    while index < len(text):

        start = text.find("{", index)

        if start == -1:
            break

        try:
            obj, end = decoder.raw_decode(text[start:])
            objects.append(obj)
            index = start + end

        except json.JSONDecodeError:
            index = start + 1

    return objects


def predict_tool_calls(query, available_tools):

    tools_text = json.dumps(
        available_tools,
        indent=2
    )

    prompt = f"""
You are an AI agent that selects tools and generates
the arguments required to call those tools.

User request:
{query}

Available tools:
{tools_text}

Your task:

1. Identify every action requested by the user.
2. Select the appropriate tool for each action.
3. Generate the arguments required by each tool.
4. Preserve the order of the requested actions.

Return ONLY a JSON array.

Format:

[
    {{
        "tool": "tool_name",
        "arguments": {{
            "parameter_name": "value"
        }}
    }}
]

Rules:

1. Every selected tool must exist in the available tools.
2. Use the exact tool name from the tool schema.
3. Use the exact parameter names from the selected tool schema.
4. Do not invent parameters.
5. Do not add optional parameters unless the user explicitly requests them.
6. Copy argument values directly from the user request.
7. Do not convert operators such as ">" into "$gt", "gt", or any other form.
8. Do not add, remove, or modify quotation marks inside string values.
9. Preserve numbers as numbers.
10. If the user specifies multiple actions, create one tool call for each action in the same order.
11. Do not guess missing values.
12. Return ONLY valid JSON. No explanations, markdown, or additional text.
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


def normalize_arguments(arguments):

    if arguments is None:
        return {}

    return arguments


def compare_tool_calls(expected_calls, predicted_calls):

    tool_correct = 0
    required_argument_correct = 0
    exact_argument_correct = 0
    exact_correct = 0

    comparison_count = max(
        len(expected_calls),
        len(predicted_calls)
    )

    for i in range(comparison_count):

        # No corresponding prediction
        if i >= len(predicted_calls):
            continue

        # No corresponding expected call
        if i >= len(expected_calls):
            continue

        expected = expected_calls[i]
        predicted = predicted_calls[i]

        expected_tool = expected["name"]

        expected_arguments = normalize_arguments(
            expected["arguments"]
        )

        predicted_tool = predicted.get("tool")

        predicted_arguments = normalize_arguments(
            predicted.get("arguments")
        )

        # ---------------------------------------------
        # 1. Tool selection accuracy
        # ---------------------------------------------

        if predicted_tool == expected_tool:
            tool_correct += 1

        # ---------------------------------------------
        # 2. Required argument accuracy
        #
        # Check only the arguments required by the
        # expected tool call.
        # ---------------------------------------------

        required_arguments_correct = True

        for key, expected_value in expected_arguments.items():

            if key not in predicted_arguments:
                required_arguments_correct = False
                break

            if predicted_arguments[key] != expected_value:
                required_arguments_correct = False
                break

        if required_arguments_correct:
            required_argument_correct += 1

        # ---------------------------------------------
        # 3. Exact argument accuracy
        #
        # Entire argument dictionary must match.
        # Extra arguments therefore cause failure.
        # ---------------------------------------------

        if predicted_arguments == expected_arguments:
            exact_argument_correct += 1

        # ---------------------------------------------
        # 4. Exact complete tool call
        # ---------------------------------------------

        if (
            predicted_tool == expected_tool
            and predicted_arguments == expected_arguments
        ):
            exact_correct += 1

    return (
        tool_correct,
        required_argument_correct,
        exact_argument_correct,
        exact_correct,
        comparison_count
    )


def main():

    print("Loading xLAM evaluation data...")

    df = pd.read_csv(DATA_PATH)

    df = df.head(SAMPLE_SIZE)

    print("Samples to evaluate:", len(df))

    total_tool_correct = 0

    total_required_argument_correct = 0

    total_exact_argument_correct = 0

    total_exact_correct = 0

    total_expected_calls = 0

    exact_query_matches = 0

    results = []

    for index, row in df.iterrows():

        query = row["query"]

        available_tools = json.loads(
            row["available_tools"]
        )

        expected_calls = json.loads(
            row["expected_calls"]
        )

        print("\n" + "=" * 70)
        print(f"Example {index + 1}/{len(df)}")
        print("=" * 70)

        print("\nQuery:")
        print(query)

        print("\nExpected calls:")
        print(
            json.dumps(
                expected_calls,
                indent=2
            )
        )

        raw_prediction = predict_tool_calls(
            query,
            available_tools
        )

        print("\nLLM raw response:")
        print(raw_prediction)

        predicted_calls = extract_tool_calls(
            raw_prediction
        )

        print("\nParsed predictions:")
        print(
            json.dumps(
                predicted_calls,
                indent=2
            )
        )

        (
            tool_correct,
            required_argument_correct,
            exact_argument_correct,
            exact_correct,
            comparison_count
        ) = compare_tool_calls(
            expected_calls,
            predicted_calls
        )

        total_tool_correct += tool_correct

        total_required_argument_correct += (
            required_argument_correct
        )

        total_exact_argument_correct += (
            exact_argument_correct
        )

        total_exact_correct += exact_correct

        total_expected_calls += len(
            expected_calls
        )

        query_exact = (
            expected_calls
            ==
            [
                {
                    "name": p.get("tool"),
                    "arguments": p.get("arguments")
                }
                for p in predicted_calls
            ]
        )

        if query_exact:
            exact_query_matches += 1

        print(
            "\nTool calls expected:",
            len(expected_calls)
        )

        print(
            "Tool calls predicted:",
            len(predicted_calls)
        )

        print(
            "Correct tools:",
            tool_correct
        )

        print(
            "Correct required arguments:",
            required_argument_correct
        )

        print(
            "Correct exact arguments:",
            exact_argument_correct
        )

        print(
            "Exact calls:",
            exact_correct
        )

        print(
            "Complete query match:",
            query_exact
        )

        results.append({
            "id": row["id"],
            "query": query,
            "expected_calls": json.dumps(
                expected_calls
            ),
            "predicted_calls": json.dumps(
                predicted_calls
            ),
            "correct_tools": tool_correct,
            "correct_required_arguments": (
                required_argument_correct
            ),
            "correct_exact_arguments": (
                exact_argument_correct
            ),
            "exact_calls": exact_correct,
            "complete_query_match": query_exact
        })

    # ---------------------------------------------
    # Calculate overall metrics
    # ---------------------------------------------

    tool_accuracy = (
        total_tool_correct / total_expected_calls
        if total_expected_calls
        else 0
    )

    required_argument_accuracy = (
        total_required_argument_correct
        / total_expected_calls
        if total_expected_calls
        else 0
    )

    exact_argument_accuracy = (
        total_exact_argument_correct
        / total_expected_calls
        if total_expected_calls
        else 0
    )

    exact_call_accuracy = (
        total_exact_correct / total_expected_calls
        if total_expected_calls
        else 0
    )

    query_exact_accuracy = (
        exact_query_matches / len(df)
        if len(df)
        else 0
    )

    # ---------------------------------------------
    # Print results
    # ---------------------------------------------

    print("\n")

    print("=" * 70)
    print("xLAM AGENT EVALUATION RESULTS")
    print("=" * 70)

    print(
        "\nQueries evaluated:",
        len(df)
    )

    print(
        "Expected tool calls:",
        total_expected_calls
    )

    print("\nTool Selection")

    print(
        f"Correct: {total_tool_correct} / "
        f"{total_expected_calls}"
    )

    print(
        f"Accuracy: {tool_accuracy:.4f}"
    )

    print("\nRequired Argument Accuracy")

    print(
        f"Correct: "
        f"{total_required_argument_correct} / "
        f"{total_expected_calls}"
    )

    print(
        f"Accuracy: "
        f"{required_argument_accuracy:.4f}"
    )

    print("\nExact Argument Accuracy")

    print(
        f"Correct: "
        f"{total_exact_argument_correct} / "
        f"{total_expected_calls}"
    )

    print(
        f"Accuracy: "
        f"{exact_argument_accuracy:.4f}"
    )

    print("\nExact Tool Calls")

    print(
        f"Correct: {total_exact_correct} / "
        f"{total_expected_calls}"
    )

    print(
        f"Accuracy: {exact_call_accuracy:.4f}"
    )

    print("\nComplete Query Match")

    print(
        f"Correct: {exact_query_matches} / "
        f"{len(df)}"
    )

    print(
        f"Accuracy: {query_exact_accuracy:.4f}"
    )

    # ---------------------------------------------
    # Save results
    # ---------------------------------------------

    os_path = "reports/xlam_agent_results.csv"

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os_path,
        index=False
    )

    print("\nResults saved:")
    print(os_path)

    # ---------------------------------------------
    # MLflow
    # ---------------------------------------------

    mlflow.set_experiment(
        "xlam-agent-evaluation"
    )

    with mlflow.start_run():

        mlflow.log_param(
            "dataset",
            "xLAM Function Calling 60K"
        )

        mlflow.log_param(
            "model",
            MODEL_NAME
        )

        mlflow.log_param(
            "evaluation_samples",
            len(df)
        )

        mlflow.log_param(
            "expected_tool_calls",
            total_expected_calls
        )

        mlflow.log_param(
            "tool_schema",
            "full"
        )

        mlflow.log_metric(
            "tool_selection_accuracy",
            tool_accuracy
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
            exact_call_accuracy
        )

        mlflow.log_metric(
            "complete_query_accuracy",
            query_exact_accuracy
        )

        mlflow.log_artifact(
            os_path
        )

    print("\nMLflow evaluation completed.")


if __name__ == "__main__":
    main()