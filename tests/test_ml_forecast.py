"""Tests for the per-series forecasters used in the AI_FORECAST comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.ml_forecast import (
    MIN_HISTORY_MONTHS,
    seasonal_naive_forecast,
    trend_seasonal_forecast,
)


def synthetic_series(n_months=60, trend=0.5, seasonal_amp=10.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    y = (
        100
        + trend * np.arange(n_months)
        + seasonal_amp * np.sin(2 * np.pi * (months.month - 1) / 12)
        + rng.normal(0, noise, n_months)
    )
    return pd.DataFrame({"month": months, "y": y})


def test_trend_seasonal_recovers_a_clean_signal():
    history = synthetic_series()
    forecast = trend_seasonal_forecast(history, horizon=6)
    assert len(forecast) == 6
    assert forecast["month"].iloc[0] == pd.Timestamp("2025-01-01")
    # On a noiseless linear+seasonal series the fit should be near-exact.
    truth = synthetic_series(n_months=66)["y"].iloc[60:].to_numpy()
    assert np.abs(forecast["yhat"].to_numpy() - truth).max() < 1.0


def test_trend_seasonal_beats_naive_on_trending_series():
    history = synthetic_series(noise=1.0)
    truth = synthetic_series(n_months=66, noise=0.0)["y"].iloc[60:].to_numpy()
    trained = trend_seasonal_forecast(history, horizon=6)["yhat"].to_numpy()
    naive = seasonal_naive_forecast(history, horizon=6)["yhat"].to_numpy()
    assert np.abs(trained - truth).mean() < np.abs(naive - truth).mean()


def test_short_series_yields_no_forecast_rather_than_a_bad_one():
    history = synthetic_series(n_months=MIN_HISTORY_MONTHS - 1)
    assert trend_seasonal_forecast(history, horizon=6).empty


def test_seasonal_naive_is_last_years_value():
    history = synthetic_series(noise=0.0, trend=0.0)
    forecast = seasonal_naive_forecast(history, horizon=3)
    for _, row in forecast.iterrows():
        past = history.loc[history["month"] == row["month"] - pd.DateOffset(months=12), "y"].iloc[0]
        assert row["yhat"] == pytest.approx(past)
