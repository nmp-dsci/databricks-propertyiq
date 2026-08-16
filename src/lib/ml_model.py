"""The rent model: sklearn pipeline, evaluation, and the promotion rule.

Deliberately boring model choice: HistGradientBoostingRegressor is
pre-installed on serverless (no runtime pip, which Free Edition egress rules
forbid), handles missing values natively, and takes ordinal-encoded categories
without a 600-column one-hot blowup for postcode. The model is a vehicle — the
MLOps loop around it is the deliverable.

Everything here is pandas-in, so it runs identically in a local pytest and in
the training notebook. The promotion rule (`challenger_wins`) is plain code
with a test, because "when does the alias flip" is exactly the kind of logic
that should not live inline in a notebook.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from lib.ml_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET

# A challenger must beat the champion's MAE by at least this margin (relative).
# Zero would flip the alias on noise; 2% is small enough to let real
# improvements through and large enough that a retrain on near-identical data
# leaves the champion alone.
PROMOTION_MARGIN = 0.02


def make_model(random_state: int = 42) -> Pipeline:
    """Ordinal-encode the categoricals, pass numerics through, boost."""
    encoder = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("encode", encoder),
            ("model", HistGradientBoostingRegressor(random_state=random_state)),
        ]
    )


def fit_model(train_pdf: pd.DataFrame, random_state: int = 42) -> Pipeline:
    model = make_model(random_state)
    model.fit(train_pdf[FEATURE_COLUMNS], train_pdf[TARGET])
    return model


def evaluate(model: Pipeline, test_pdf: pd.DataFrame) -> dict[str, float]:
    """MAE is the headline (dollars-per-week, explainable out loud); R² for shape."""
    pred = model.predict(test_pdf[FEATURE_COLUMNS])
    return {
        "mae": float(mean_absolute_error(test_pdf[TARGET], pred)),
        "r2": float(r2_score(test_pdf[TARGET], pred)),
    }


def naive_baseline_mae(test_pdf: pd.DataFrame) -> float:
    """MAE of "next month's rent is last month's rent".

    The honesty check: a model that cannot beat rent_lag_1 has learned nothing
    worth deploying, whatever its absolute MAE looks like.
    """
    anchored = test_pdf.dropna(subset=["rent_lag_1"])
    return float(mean_absolute_error(anchored[TARGET], anchored["rent_lag_1"]))


def challenger_wins(challenger_mae: float, champion_mae: float | None) -> bool:
    """The promotion gate. No champion yet → the first model wins by default."""
    if champion_mae is None:
        return True
    return challenger_mae < champion_mae * (1 - PROMOTION_MARGIN)
