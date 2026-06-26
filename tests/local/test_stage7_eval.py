"""Local tests for compute_eval1_failure_rate in stage7_evaluation.py."""

from app.pipeline.stage7_evaluation import compute_eval1_failure_rate


def _verdicts(n: int, n_incorrect: int) -> list[dict]:
    verdicts = [{"comment_id": i, "correct": True, "issue": None} for i in range(n - n_incorrect)]
    verdicts += [{"comment_id": n - n_incorrect + i, "correct": False, "issue": "test"} for i in range(n_incorrect)]
    return verdicts


def test_failure_rate_zero():
    result = compute_eval1_failure_rate(_verdicts(20, 0))
    assert result == 0.0


def test_failure_rate_boundary_15():
    result = compute_eval1_failure_rate(_verdicts(20, 3))
    assert result == 0.15


def test_failure_rate_boundary_25():
    result = compute_eval1_failure_rate(_verdicts(20, 5))
    assert result == 0.25
