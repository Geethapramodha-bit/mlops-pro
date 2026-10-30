import pytest

from app.mlops.evaluation_gate import check_evaluation_gate


def test_evaluation_gate_passes_when_metrics_meet_thresholds():
    assert check_evaluation_gate(0.85, 0.85) is True
    assert check_evaluation_gate(0.90, 0.88) is True


def test_evaluation_gate_rejects_low_accuracy():
    with pytest.raises(ValueError, match="Validation accuracy"):
        check_evaluation_gate(0.80, 0.90)


def test_evaluation_gate_rejects_low_f1():
    with pytest.raises(ValueError, match="Validation F1"):
        check_evaluation_gate(0.90, 0.80)


def test_evaluation_gate_rejects_when_both_metrics_are_low():
    with pytest.raises(ValueError, match="Validation accuracy.*Validation F1"):
        check_evaluation_gate(0.70, 0.75)
