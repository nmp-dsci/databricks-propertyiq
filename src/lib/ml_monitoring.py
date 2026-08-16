"""Drift metrics — the hand-rolled substitute for Lakehouse Monitoring.

Lakehouse Monitoring's availability on Free Edition is undocumented, so the
monitoring story here is built from parts that work everywhere: this module
computes the numbers, a notebook writes them to a metrics table, and a SQL
alert on the warehouse watches the table. Same loop, no preview features.

Two families of metric:

- **PSI** (population stability index) per feature: has the *input*
  distribution moved since training? Conventional reading: < 0.1 stable,
  0.1–0.25 drifting, > 0.25 shifted. Catches drift *before* actuals arrive.
- **Rolling MAE** once actuals land: is the *output* still right? The rent
  mart publishes monthly, so predictions can be scored against reality with a
  one-month delay — a luxury most ML systems don't get.

Pandas-in, dict/DataFrame-out, unit-tested like everything else in lib/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_BINS = 10
# Conventional thresholds; the SQL alert fires on `psi > 0.25`.
PSI_DRIFTING = 0.1
PSI_SHIFTED = 0.25

# Features that must not be drift-checked. month_of_year is cyclical: a single
# scored month always occupies exactly one of twelve reference bins, so its PSI
# is enormous by construction and would fire the retrain trigger on every run —
# a treadmill, not a monitor. First observed live: PSI 8.64 on an otherwise
# healthy run.
DRIFT_EXCLUDE = frozenset({"month_of_year"})


def drift_columns(columns: list[str]) -> list[str]:
    """The subset of feature columns that PSI is meaningful for."""
    return [c for c in columns if c not in DRIFT_EXCLUDE]


def psi(reference: pd.Series, current: pd.Series, bins: int = PSI_BINS) -> float:
    """PSI between a training-time reference and a current window.

    Bin edges come from the *reference* quantiles, so the question is always
    "how far has now moved from what the model trained on", never a moving
    target. Empty bins get a floor count to keep the log finite.
    """
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # a near-constant feature has nothing to drift
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur_pct = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_psi_frame(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """One row per numeric feature: its PSI and a status label for the alert."""
    rows = []
    for col in columns:
        value = psi(reference[col], current[col])
        status = (
            "shifted" if value > PSI_SHIFTED else "drifting" if value > PSI_DRIFTING else "stable"
        )
        rows.append({"feature": col, "psi": value, "status": status})
    return pd.DataFrame(rows)
