MIN_VALIDATION_ACCURACY = 0.85
MIN_VALIDATION_F1 = 0.85


def check_evaluation_gate(validation_accuracy, validation_f1):
    """Reject models that do not meet the validation metric thresholds."""
    failures = []

    if validation_accuracy < MIN_VALIDATION_ACCURACY:
        failures.append(
            f"Validation accuracy {validation_accuracy:.4f} is below "
            f"the required {MIN_VALIDATION_ACCURACY:.2f}"
        )

    if validation_f1 < MIN_VALIDATION_F1:
        failures.append(
            f"Validation F1 {validation_f1:.4f} is below "
            f"the required {MIN_VALIDATION_F1:.2f}"
        )

    if failures:
        raise ValueError("Model evaluation gate failed: " + "; ".join(failures))

    return True
