"""Feature engineering for the rent model — pure DataFrame transforms.

Same contract as transforms.py: DataFrame in, DataFrame out, no I/O, no
globals, unit-tested locally. The notebook that materialises the feature table
and the notebook that scores both call `build_rent_features`, so training and
inference cannot drift apart — the function *is* the no-skew guarantee (rubric
R4).

Grain: one row per postcode × property_type × bedroom_band × month, the same
grain as gold_property_rent. The target is that month's median_weekly_rent;
every feature is built only from *earlier* months (lags and trailing windows),
so a row never sees its own answer.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# What the model eats. Kept as module constants so the trainer, the scorer and
# the tests all agree by construction rather than by copy-paste.
TARGET = "median_weekly_rent"
CATEGORICAL_FEATURES = ["postcode", "property_type", "bedroom_band"]
NUMERIC_FEATURES = [
    "rent_lag_1",
    "rent_lag_3",
    "rent_lag_12",
    "rent_trailing_3m",
    "volume_trailing_3m",
    "sale_price_lag_1",
    "month_of_year",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
KEY_COLUMNS = ["postcode", "property_type", "bedroom_band", "month"]


def _postcode_sales_signal(gold_sales: DataFrame) -> DataFrame:
    """Average sale price per postcode × property_type × month.

    gold_property_sales carries extra dimensions (suburb, area_band, zoning),
    so re-aggregate by summing the additive legs and dividing — never by
    averaging the per-row averages, per the marts' own documentation.
    """
    return gold_sales.groupBy("postcode", "property_type", "month").agg(
        (F.sum("total_sale_value") / F.sum("n_sold")).alias("avg_sale_price_pc"),
    )


def build_rent_features(gold_rent: DataFrame, gold_sales: DataFrame) -> DataFrame:
    """Lagged rent features + a sales-side signal, at the rent-mart grain.

    Lags join on the actual calendar month (shift the frame forward with
    add_months, join back on the key), not on row offsets: thin postcodes skip
    months, and a row-offset `F.lag` would smuggle a 4-month-old value in as
    "last month". A missing month yields NULL, which the model treats as
    missing — the trailing window below is the row-offset exception, and it is
    an average, where an occasional gap blurs rather than lies.
    """
    series = Window.partitionBy(*CATEGORICAL_FEATURES).orderBy("month")
    trailing_3 = series.rowsBetween(-3, -1)

    base = gold_rent.select(*KEY_COLUMNS, TARGET, "n_rented", "avg_weekly_rent")

    lag_1 = base.select(
        *CATEGORICAL_FEATURES,
        F.add_months(F.col("month"), 1).alias("month"),
        F.col(TARGET).alias("rent_lag_1"),
    )
    lag_3 = base.select(
        *CATEGORICAL_FEATURES,
        F.add_months(F.col("month"), 3).alias("month"),
        F.col(TARGET).alias("rent_lag_3"),
    )
    lag_12 = base.select(
        *CATEGORICAL_FEATURES,
        F.add_months(F.col("month"), 12).alias("month"),
        F.col(TARGET).alias("rent_lag_12"),
    )

    sales = _postcode_sales_signal(gold_sales).select(
        "postcode",
        "property_type",
        F.add_months(F.col("month"), 1).alias("month"),
        F.col("avg_sale_price_pc").alias("sale_price_lag_1"),
    )

    return (
        base.withColumn("rent_trailing_3m", F.avg("avg_weekly_rent").over(trailing_3))
        .withColumn("volume_trailing_3m", F.avg("n_rented").over(trailing_3))
        .join(lag_1, on=KEY_COLUMNS, how="left")
        .join(lag_3, on=KEY_COLUMNS, how="left")
        .join(lag_12, on=KEY_COLUMNS, how="left")
        .join(sales, on=["postcode", "property_type", "month"], how="left")
        .withColumn("month_of_year", F.month("month"))
        # A row with no last-month rent has nothing to anchor a prediction to,
        # and in practice means the series just started. Everything else may be
        # NULL — HistGradientBoosting handles missing values natively.
        .filter(F.col("rent_lag_1").isNotNull())
        .select(*KEY_COLUMNS, *NUMERIC_FEATURES, TARGET)
    )


def split_by_month(df: DataFrame, holdout_months: int) -> tuple[DataFrame, DataFrame]:
    """Time-based split: the last `holdout_months` months are the test set.

    Never a random split — shuffled rows would let the model see the future of
    the very series it is tested on, and the resulting metric would be fiction.
    """
    cutoff = df.select(F.add_months(F.max("month"), -holdout_months).alias("c")).collect()[0]["c"]
    return df.filter(F.col("month") <= cutoff), df.filter(F.col("month") > cutoff)
