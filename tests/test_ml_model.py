"""Tests for the model pipeline, evaluation, and the promotion gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.ml_features import FEATURE_COLUMNS, TARGET
from lib.ml_model import (
    PROMOTION_MARGIN,
    challenger_wins,
    evaluate,
    fit_model,
    naive_baseline_mae,
)
from lib.ml_monitoring import drift_columns, feature_psi_frame, psi


def synthetic_frame(n=400, seed=0):
    """Rent that actually depends on the features, so the model can learn it."""
    rng = np.random.default_rng(seed)
    lag_1 = rng.uniform(300, 900, n)
    pdf = pd.DataFrame(
        {
            "postcode": rng.choice(["2000", "2327", "2650"], n),
            "property_type": rng.choice(["house", "unit"], n),
            "bedroom_band": rng.choice(["1", "2", "3"], n),
            "rent_lag_1": lag_1,
            "rent_lag_3": lag_1 * rng.uniform(0.95, 1.0, n),
            "rent_lag_12": lag_1 * rng.uniform(0.9, 0.98, n),
            "rent_trailing_3m": lag_1 * rng.uniform(0.97, 1.01, n),
            "volume_trailing_3m": rng.uniform(5, 50, n),
            "sale_price_lag_1": lag_1 * 1000 + rng.normal(0, 20000, n),
            "month_of_year": rng.integers(1, 13, n),
        }
    )
    pdf[TARGET] = lag_1 * 1.02 + rng.normal(0, 5, n)
    return pdf


def test_model_fits_and_beats_naive_on_learnable_data():
    pdf = synthetic_frame()
    train, test = pdf.iloc[:300], pdf.iloc[300:]
    model = fit_model(train)
    metrics = evaluate(model, test)
    assert set(metrics) == {"mae", "r2"}
    assert metrics["mae"] < naive_baseline_mae(test)
    assert metrics["r2"] > 0.9


def test_model_survives_missing_values_and_unseen_categories():
    pdf = synthetic_frame()
    model = fit_model(pdf.iloc[:300])
    holed = pdf.iloc[300:].copy()
    holed.loc[:, "rent_lag_12"] = np.nan
    holed.loc[:, "sale_price_lag_1"] = np.nan
    holed.loc[:, "postcode"] = "9999"  # never seen in training
    pred = model.predict(holed[FEATURE_COLUMNS])
    assert np.isfinite(pred).all()


def test_promotion_gate():
    assert challenger_wins(100.0, None)  # first model ever → promote
    assert challenger_wins(90.0, 100.0)  # clear win
    assert not challenger_wins(99.5, 100.0)  # inside the noise margin → hold
    assert not challenger_wins(101.0, 100.0)  # worse → hold
    exact_margin = 100.0 * (1 - PROMOTION_MARGIN)
    assert not challenger_wins(exact_margin, 100.0)  # must beat, not meet


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(500, 50, 2000))
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-6)
    shifted = pd.Series(rng.normal(700, 50, 2000))
    assert psi(ref, shifted) > 0.25


def test_drift_columns_exclude_cyclical_indicators():
    from lib.ml_features import NUMERIC_FEATURES

    kept = drift_columns(NUMERIC_FEATURES)
    assert "month_of_year" not in kept  # always "shifted" vs a full-year reference
    assert "rent_lag_1" in kept


def test_feature_psi_frame_labels_status():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(3, 1, 1000)})
    out = feature_psi_frame(ref, cur, ["a", "b"]).set_index("feature")
    assert out.loc["a", "status"] == "stable"
    assert out.loc["b", "status"] == "shifted"
