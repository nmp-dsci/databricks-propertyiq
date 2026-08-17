"""Per-series forecasting for the AI_FORECAST comparison — pure pandas.

The comparison needs a *trained* counterpart to AI_FORECAST that runs on Free
Edition serverless, which rules out the usual suspects: prophet and
statsforecast are not pre-installed and runtime pip from an arbitrary index is
forbidden, and Spark MLlib does not exist under Spark Connect. So the trained
model is deliberately modest — per-series linear trend + month-of-year
seasonality via sklearn (pre-installed) — because the deliverable is the
**backtesting discipline and the tool-selection judgment**, not forecast
accuracy. The honest baseline both must beat is seasonal-naive.

Each function takes one series as a pandas frame with columns (month, y),
sorted by month — the shape `applyInPandas` hands over per group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# A series needs enough history to estimate 12 seasonal offsets and a trend
# without reading noise as signal.
MIN_HISTORY_MONTHS = 36


def _future_months(last_month: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_month, periods=horizon + 1, freq="MS")[1:]


def _design(months: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Trend index + one-hot month-of-year, anchored at a fixed origin."""
    months = pd.DatetimeIndex(months)
    t = ((months.year - origin.year) * 12 + (months.month - origin.month)).to_numpy()
    seasonal = np.zeros((len(months), 12))
    seasonal[np.arange(len(months)), months.month - 1] = 1.0
    return np.column_stack([t, seasonal])


def trend_seasonal_forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Linear trend + monthly seasonality, fit per series, forecast `horizon` months."""
    history = history.sort_values("month")
    if len(history) < MIN_HISTORY_MONTHS:
        return pd.DataFrame(columns=["month", "yhat"])

    months = pd.DatetimeIndex(history["month"])
    origin = months[0]
    model = LinearRegression().fit(_design(months, origin), history["y"].to_numpy())

    future = _future_months(months[-1], horizon)
    return pd.DataFrame({"month": future, "yhat": model.predict(_design(future, origin))})


def seasonal_naive_forecast(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """The honest baseline: next July = last July. Beat this or train nothing."""
    history = history.sort_values("month")
    if len(history) < 12:
        return pd.DataFrame(columns=["month", "yhat"])

    by_month = history.set_index(pd.DatetimeIndex(history["month"]))["y"]
    future = _future_months(pd.DatetimeIndex(history["month"])[-1], horizon)
    yhat = [by_month.get(m - pd.DateOffset(months=12), np.nan) for m in future]
    return pd.DataFrame({"month": future, "yhat": yhat})
