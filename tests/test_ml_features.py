"""Tests for the rent-model feature builder.

The properties that matter: lags come from the actual calendar month (a gap in
a thin series must NOT smuggle an older value in as "last month"), no feature
row leaks its own target, and the time split never lets test months into train.
"""

from __future__ import annotations

import datetime as dt

import pytest

from lib.ml_features import (
    FEATURE_COLUMNS,
    build_rent_features,
    split_by_month,
)

RENT_COLUMNS = [
    "postcode",
    "property_type",
    "bedroom_band",
    "month",
    "total_weekly_rent",
    "n_rented",
    "avg_weekly_rent",
    "median_weekly_rent",
    "min_weekly_rent",
    "max_weekly_rent",
]

SALES_COLUMNS = [
    "postcode",
    "suburb",
    "property_type",
    "area_band",
    "zoning",
    "month",
    "total_sale_value",
    "n_sold",
    "avg_sale_price",
    "median_sale_price",
    "min_sale_price",
    "max_sale_price",
]


def rent_row(month, median=500.0, **overrides):
    base = {
        "postcode": "2327",
        "property_type": "house",
        "bedroom_band": "3",
        "month": dt.date.fromisoformat(month),
        "total_weekly_rent": median * 10,
        "n_rented": 10,
        "avg_weekly_rent": median,
        "median_weekly_rent": median,
        "min_weekly_rent": median - 100,
        "max_weekly_rent": median + 100,
    }
    base.update(overrides)
    return tuple(base[c] for c in RENT_COLUMNS)


def sales_row(month, avg=650000.0, n=4, **overrides):
    base = {
        "postcode": "2327",
        "suburb": "KURRI KURRI",
        "property_type": "house",
        "area_band": "unknown",
        "zoning": "R2",
        "month": dt.date.fromisoformat(month),
        "total_sale_value": avg * n,
        "n_sold": n,
        "avg_sale_price": avg,
        "median_sale_price": avg,
        "min_sale_price": avg,
        "max_sale_price": avg,
    }
    base.update(overrides)
    return tuple(base[c] for c in SALES_COLUMNS)


RENT_SCHEMA = (
    "postcode string, property_type string, bedroom_band string, month date, "
    "total_weekly_rent double, n_rented long, avg_weekly_rent double, "
    "median_weekly_rent double, min_weekly_rent double, max_weekly_rent double"
)

SALES_SCHEMA = (
    "postcode string, suburb string, property_type string, area_band string, "
    "zoning string, month date, total_sale_value double, n_sold long, "
    "avg_sale_price double, median_sale_price double, min_sale_price double, "
    "max_sale_price double"
)


@pytest.fixture
def gold_rent(spark):
    def make(rows):
        return spark.createDataFrame(rows, RENT_SCHEMA)

    return make


@pytest.fixture
def gold_sales(spark):
    def make(rows):
        return spark.createDataFrame(rows, SALES_SCHEMA)

    return make


def test_lag_1_is_previous_calendar_month(gold_rent, gold_sales):
    rows = [rent_row("2026-01-01", 500.0), rent_row("2026-02-01", 520.0)]
    out = build_rent_features(gold_rent(rows), gold_sales([])).collect()
    assert len(out) == 1  # January has no lag_1 and is dropped
    feb = out[0]
    assert feb["month"] == dt.date(2026, 2, 1)
    assert feb["rent_lag_1"] == 500.0
    assert feb["median_weekly_rent"] == 520.0


def test_series_gap_does_not_fake_a_lag(gold_rent, gold_sales):
    # Thin postcode skips February entirely; March's "last month" must be NULL,
    # not January's value wearing a lag_1 badge.
    rows = [rent_row("2026-01-01", 500.0), rent_row("2026-03-01", 540.0)]
    out = build_rent_features(gold_rent(rows), gold_sales([]))
    assert out.count() == 0  # March has no lag_1 → dropped, January is a series start


def test_lag_12_reaches_a_year_back(gold_rent, gold_sales):
    rows = [
        rent_row("2025-02-01", 480.0),
        rent_row("2026-01-01", 500.0),
        rent_row("2026-02-01", 520.0),
    ]
    out = {r["month"]: r for r in build_rent_features(gold_rent(rows), gold_sales([])).collect()}
    assert out[dt.date(2026, 2, 1)]["rent_lag_12"] == 480.0
    assert out[dt.date(2026, 2, 1)]["rent_lag_1"] == 500.0


def test_sales_signal_reaggregates_additive_legs(gold_rent, gold_sales):
    # Two suburb cells in the same postcode/month: the signal must be the
    # legs-weighted average, not the average of the two cell averages.
    sales = [
        sales_row("2026-01-01", avg=1000000.0, n=1),
        sales_row("2026-01-01", avg=500000.0, n=9, suburb="WESTON"),
    ]
    rows = [rent_row("2026-01-01", 500.0), rent_row("2026-02-01", 520.0)]
    feb = build_rent_features(gold_rent(rows), gold_sales(sales)).collect()[0]
    assert feb["sale_price_lag_1"] == pytest.approx(550000.0)  # (1M + 4.5M) / 10


def test_no_target_leakage_in_features(gold_rent, gold_sales):
    # Same-month target must never appear among the features: perturb the
    # target of the scored month and confirm no feature moves.
    base = [rent_row("2026-01-01", 500.0), rent_row("2026-02-01", 520.0)]
    bumped = [rent_row("2026-01-01", 500.0), rent_row("2026-02-01", 999.0)]
    row_a = build_rent_features(gold_rent(base), gold_sales([])).collect()[0]
    row_b = build_rent_features(gold_rent(bumped), gold_sales([])).collect()[0]
    for col in FEATURE_COLUMNS:
        assert row_a[col] == row_b[col], f"feature {col} leaked the target"


def test_split_by_month_holds_out_the_tail(gold_rent, gold_sales, spark):
    rows = [rent_row(f"2026-0{m}-01", 500.0 + m) for m in range(1, 8)]
    features = build_rent_features(gold_rent(rows), gold_sales([]))
    train, test = split_by_month(features, holdout_months=2)
    max_train = train.agg({"month": "max"}).collect()[0][0]
    min_test = test.agg({"month": "min"}).collect()[0][0]
    assert max_train < min_test
    assert test.count() == 2  # June + July
