"""Tests for drift metrics — PSI, its status labels, and the retrain trigger value.

The property that matters: a feature that couldn't be compared (empty
reference or current window) must never read as "stable", and the value fed
to the ml_score condition task must stay a real number even when it happens.
"""

from __future__ import annotations

import math

import pandas as pd

from lib.ml_monitoring import feature_psi_frame, max_monitorable_psi, psi


def test_psi_of_identical_distributions_is_zero():
    reference = pd.Series(range(100))
    current = pd.Series(range(100))
    assert psi(reference, current) == 0.0


def test_psi_is_nan_for_empty_current_window():
    reference = pd.Series(range(100))
    current = pd.Series([], dtype=float)
    assert math.isnan(psi(reference, current))


def test_feature_psi_frame_labels_nan_psi_unmonitorable_not_stable():
    reference = pd.DataFrame({"a": range(100), "b": range(100)})
    current = pd.DataFrame({"a": range(100), "b": [None] * 100})
    frame = feature_psi_frame(reference, current, ["a", "b"])

    row_b = frame.set_index("feature").loc["b"]
    assert row_b["status"] == "unmonitorable"
    assert math.isnan(row_b["psi"])


def test_max_monitorable_psi_ignores_unmonitorable_features():
    psi_pdf = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "psi": [0.3, float("nan")],
            "status": ["shifted", "unmonitorable"],
        }
    )
    assert max_monitorable_psi(psi_pdf) == 0.3


def test_max_monitorable_psi_is_zero_not_nan_when_all_unmonitorable():
    psi_pdf = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "psi": [float("nan"), float("nan")],
            "status": ["unmonitorable", "unmonitorable"],
        }
    )
    result = max_monitorable_psi(psi_pdf)
    assert result == 0.0
    assert not math.isnan(result)
