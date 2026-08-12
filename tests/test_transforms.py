"""Unit tests for the silver/gold transform rules.

These run in a few seconds on a local Spark session — no workspace, no warehouse,
no network. The point is that a change to a cleaning rule gets a red test
immediately, instead of after a job run and a squint at a dashboard.
"""

from __future__ import annotations

import pytest

from lib.transforms import (
    add_revenue,
    daily_revenue,
    deduplicate_orders,
    flag_quality,
    normalise_country,
    quality_summary,
    to_silver,
)

BRONZE_COLUMNS = [
    "order_id",
    "order_ts",
    "customer_id",
    "country",
    "channel",
    "category",
    "quantity",
    "unit_price",
    "_ingested_at",
]


def bronze_rows():
    """A handful of rows covering every rule, including the awkward ones.

    Kept as a literal table — a fixture factory would hide exactly the detail
    that makes each case interesting.
    """
    # fmt: off
    return [
        # id       order_ts            cust  country  channel    category    qty  price  ingested
        # clean baseline
        ("ORD-1", "2026-08-01 10:00", 1,    "AU",    "web",     "grocery",   2, 10.0, "2026-08-02"),
        # lower-case and padded country codes — formatting noise, repairable
        ("ORD-2", "2026-08-01 11:00", 2,    "au",    "app",     "home",      1, 20.0, "2026-08-02"),
        ("ORD-3", "2026-08-01 12:00", 3,    "  US ", "store",   "sport",     3,  5.0, "2026-08-02"),
        # country we don't operate in — becomes NULL, not a guess
        ("ORD-4", "2026-08-01 13:00", 4,    "ZZ",    "web",     "apparel",   1, 15.0, "2026-08-02"),
        # negative quantity: a return mis-keyed as an order
        ("ORD-5", "2026-08-01 14:00", 5,    "NZ",    "web",     "grocery",  -2,  8.0, "2026-08-02"),
        # missing customer id
        ("ORD-6", "2026-08-01 15:00", None, "GB",    "partner", "home",      1, 30.0, "2026-08-02"),
        # duplicate of ORD-1, ingested a day later, with a corrected quantity
        ("ORD-1", "2026-08-01 10:00", 1,    "AU",    "web",     "grocery",   4, 10.0, "2026-08-03"),
    ]
    # fmt: on


@pytest.fixture
def bronze(spark):
    return spark.createDataFrame(bronze_rows(), BRONZE_COLUMNS)


class TestNormaliseCountry:
    def test_repairs_case_and_whitespace(self, bronze):
        got = {r.order_id: r.country for r in normalise_country(bronze).collect()}
        assert got["ORD-2"] == "AU"
        assert got["ORD-3"] == "US"

    def test_unknown_code_becomes_null_not_a_guess(self, bronze):
        got = {r.order_id: r.country for r in normalise_country(bronze).collect()}
        assert got["ORD-4"] is None


class TestDeduplicate:
    def test_one_row_per_order(self, bronze):
        result = deduplicate_orders(bronze)
        assert result.count() == 6
        assert result.select("order_id").distinct().count() == 6

    def test_keeps_the_latest_ingested_version(self, bronze):
        """ORD-1 arrives twice; the later row corrected quantity 2 -> 4."""
        row = deduplicate_orders(bronze).filter("order_id = 'ORD-1'").collect()[0]
        assert row.quantity == 4


class TestRevenue:
    def test_rounds_to_cents(self, spark):
        df = spark.createDataFrame([(3, 9.99)], ["quantity", "unit_price"])
        assert add_revenue(df).collect()[0].revenue == pytest.approx(29.97)

    def test_negative_quantity_yields_negative_revenue(self, spark):
        df = spark.createDataFrame([(-2, 8.0)], ["quantity", "unit_price"])
        assert add_revenue(df).collect()[0].revenue == pytest.approx(-16.0)


class TestQualityFlags:
    @pytest.fixture
    def flagged(self, bronze):
        df = flag_quality(normalise_country(bronze))
        return {r.order_id: r for r in df.collect()}

    def test_clean_row_has_no_reasons(self, flagged):
        assert flagged["ORD-1"]._quality == []
        assert flagged["ORD-1"]._is_valid is True

    @pytest.mark.parametrize(
        ("order_id", "reason"),
        [
            ("ORD-4", "unknown_country"),
            ("ORD-5", "non_positive_quantity"),
            ("ORD-6", "missing_customer"),
        ],
    )
    def test_each_rule_names_itself(self, flagged, order_id, reason):
        assert reason in flagged[order_id]._quality
        assert flagged[order_id]._is_valid is False

    def test_reasons_accumulate(self, spark):
        """A row can fail more than one check, and should report all of them."""
        df = spark.createDataFrame(
            [("ORD-X", None, None, -1, 0.0)],
            ["order_id", "customer_id", "country", "quantity", "unit_price"],
        )
        reasons = set(flag_quality(df).collect()[0]._quality)
        assert reasons == {
            "non_positive_quantity",
            "non_positive_price",
            "missing_customer",
            "unknown_country",
        }


class TestSilverEndToEnd:
    @pytest.fixture
    def silver(self, bronze):
        return to_silver(bronze)

    def test_deduplicates_and_keeps_invalid_rows(self, silver):
        """Silver keeps everything — the verdict is a column, not a filter."""
        assert silver.count() == 6
        assert silver.filter("NOT _is_valid").count() == 3

    def test_derives_order_date(self, silver):
        dates = {str(r.order_date) for r in silver.collect()}
        assert dates == {"2026-08-01"}


class TestGoldAggregates:
    def test_daily_revenue_excludes_invalid_rows(self, bronze):
        gold = daily_revenue(to_silver(bronze))
        # 3 of the 6 deduped rows are valid: ORD-1, ORD-2, ORD-3.
        assert gold.agg({"orders": "sum"}).collect()[0][0] == 3
        # ORD-1 (4 x 10) + ORD-2 (1 x 20) + ORD-3 (3 x 5) = 75
        assert sum(r.revenue for r in gold.collect()) == pytest.approx(75.0)

    def test_quality_summary_accounts_for_every_row(self, bronze):
        silver = to_silver(bronze)
        summary = quality_summary(silver)
        ok = sum(r.rows for r in summary.collect() if r.reason == "ok")
        assert ok == silver.filter("_is_valid").count()
