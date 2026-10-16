import pytest
from colony_detector.metrics import count_metrics


def test_iso_lab_standard_metric():
    # Cases <= 30: allowed error is +- 3
    # Case 1: true=20, pred=22 (error=+2, pass)
    # Case 2: true=10, pred=14 (error=+4, fail)
    # Cases > 30: allowed error is +- 10%
    # Case 3: true=100, pred=108 (error=+8%, pass)
    # Case 4: true=100, pred=115 (error=+15%, fail)
    true_counts = [20, 10, 100, 100]
    pred_counts = [22, 14, 108, 115]

    metrics = count_metrics(true_counts, pred_counts)
    assert "iso_pass_rate" in metrics
    assert "iso_pass_pct" in metrics
    assert metrics["iso_passed"] == 2
    assert metrics["iso_pass_rate"] == 0.50
    assert metrics["iso_pass_pct"] == 50.0
