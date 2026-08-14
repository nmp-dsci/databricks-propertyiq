"""Unit tests for the PropertyIQ silver/gold transform rules.

These run in seconds on a local Spark session — no workspace, no warehouse, no
network. Each rule ported from the dbt project gets a fixture row that proves
it, and the sales fixtures deliberately cover all three encoding eras of the
source data: raw DAT strings, pandas float round-trips, and the legacy
monolith's scientific notation.
"""

from __future__ import annotations

import pytest

from lib.transforms import (
    clean_rent,
    clean_sales,
    gold_rent_monthly,
    gold_sales_monthly,
    gold_yield_monthly,
    quality_summary,
)

SALES_COLUMNS = [
    "file",
    "index",
    "contract_dt",
    "locality",
    "postcode",
    "prop_purpose",
    "property_id",
    "sale_price",
    "strata_no",
    "area_sqm",
    "area_type",
    "zoning",
]


def sales_row(**overrides):
    """One clean era-2 (raw DAT string) sale; override fields per test case."""
    base = {
        "file": "001_SALES_DATA_NNME_29062026.DAT",
        "index": "1",
        "contract_dt": "20260315",
        "locality": "KURRI KURRI",
        "postcode": "2327",
        "prop_purpose": "RESIDENCE",
        "property_id": "2848404",
        "sale_price": "650000",
        "strata_no": "",
        "area_sqm": "668",
        "area_type": "M",
        "zoning": "R2",
    }
    base.update(overrides)
    return tuple(base[c] for c in SALES_COLUMNS)


@pytest.fixture
def sales(spark):
    rows = [
        # era 2: raw DAT strings — the baseline clean row
        sales_row(),
        # era 1: pandas float round-trip ("20150416.0", "2327.0")
        sales_row(
            index="2",
            contract_dt="20150416.0",
            postcode="2327.0",
            property_id="16521.0",
            sale_price="198000.0",
            area_sqm="809.43",
        ),
        # era 0: the legacy monolith's scientific notation
        sales_row(index="3", contract_dt="2.0150416E7", sale_price="200000.0"),
        # strata present -> unit; area in hectares
        sales_row(index="4", strata_no="SP1234", area_sqm="0.0668", area_type="H"),
        # $1 family transfer — kept, flagged
        sales_row(index="5", sale_price="1"),
        # commercial purpose + blank locality — two reasons on one row
        sales_row(index="6", prop_purpose="COMMERCIAL", locality=" "),
        # junk date
        sales_row(index="7", contract_dt="not-a-date"),
        # 6-digit date — really occurs; crashed the first serverless run
        sales_row(index="10", contract_dt="170823.0"),
        # 8 digits that are not a date — month 99
        sales_row(index="11", contract_dt="20159999"),
        # pre-2010 sale
        sales_row(index="8", contract_dt="20051103"),
        # exact duplicate of the baseline: same business key, distinct sale_id
        sales_row(index="9"),
    ]
    return clean_sales(spark.createDataFrame(rows, SALES_COLUMNS))


class TestSalesEncodingEras:
    def test_all_three_eras_parse_to_the_same_shapes(self, sales):
        got = {r.sale_id: r for r in sales.collect()}
        dates = {str(r.sale_date) for r in got.values() if r.sale_date and r.sale_date.year == 2015}
        assert dates == {"2015-04-16"}  # era 1 and era 0 land identically

    def test_postcode_float_damage_is_repaired(self, sales):
        assert {r.postcode for r in sales.select("postcode").distinct().collect()} == {"2327"}

    def test_junk_dates_become_null_not_an_error(self, sales):
        """'not-a-date', 6-digit '170823.0' and impossible '20159999' all flag."""
        rows = sales.filter("array_contains(_quality, 'unparseable_contract_date')").collect()
        assert len(rows) == 3 and all(r.sale_date is None for r in rows)


class TestSalesRules:
    def test_strata_derives_unit(self, sales):
        types = sales.groupBy("property_type").count().collect()
        assert {r.property_type: r["count"] for r in types} == {"house": 10, "unit": 1}

    def test_hectares_standardised_to_sqm(self, sales):
        unit = sales.filter("property_type = 'unit'").collect()[0]
        assert unit.area_sqm == pytest.approx(668.0)
        assert unit.area_band == "400-700"

    def test_suburb_title_cased(self, sales):
        assert sales.filter("suburb = 'Kurri Kurri'").count() > 0

    def test_nominal_transfer_kept_and_flagged(self, sales):
        """The $10k floor flags rather than drops — the row stays queryable."""
        row = sales.filter("sale_price = 1").collect()[0]
        assert "non_arms_length_price" in row._quality
        assert row._is_valid is False

    def test_reasons_accumulate(self, sales):
        row = sales.filter("prop_purpose = 'COMMERCIAL'").collect()[0]
        assert {"not_residence", "blank_locality"} <= set(row._quality)

    def test_pre_2010_flagged(self, sales):
        assert sales.filter("array_contains(_quality, 'pre_2010_sale')").count() == 1

    def test_clean_rows_are_valid(self, sales):
        assert sales.filter("_is_valid").count() == 5  # eras 0/1/2, the unit, the dup

    def test_sale_ids_unique_even_for_identical_business_keys(self, sales):
        """Split settlements share property_id+date+price; ids must not collide."""
        assert sales.select("sale_id").distinct().count() == sales.count()


RENT_COLUMNS = ["lodgement_dt", "postcode", "property_type", "bedrooms", "weekly_rent"]


@pytest.fixture
def rent(spark):
    rows = [
        ("2026-06-19", "2000", "F", "0", "800"),
        ("2026-06-19", "2000", "H", "3", "950"),
        ("2026-06-19", "2000", "T", "3", "900"),  # townhouse counts as house
        ("2026-06-19", "2327", "U", "6", "700"),  # 5+ band
        ("2026-06-19", "2327", "F", "", "550"),  # unknown bedrooms
        ("junk", "2000", "F", "1", "500"),  # unparseable date
        ("2026-06-19", "2000", "F", "1", "0"),  # zero rent
    ]
    return clean_rent(spark.createDataFrame(rows, RENT_COLUMNS))


class TestRentRules:
    def test_house_vocabulary_mirrors_sales(self, rent):
        houses = rent.filter("property_type = 'house'")
        assert {r.property_type_code for r in houses.collect()} == {"H", "T"}

    def test_bedroom_bands(self, rent):
        bands = {r.bedroom_band for r in rent.collect()}
        assert {"0", "3", "5+", "unknown"} <= bands

    def test_bad_rows_flagged_not_dropped(self, rent):
        assert rent.count() == 7
        flagged = rent.filter("NOT _is_valid")
        assert flagged.count() == 2
        reasons = {reason for r in flagged.collect() for reason in r._quality}
        assert reasons == {"unparseable_lodgement_date", "non_positive_rent"}

    def test_rent_ids_unique(self, rent):
        assert rent.select("rent_id").distinct().count() == 7


def _yield_inputs(spark):
    """Six sales and six bonds in one cell (2000/house/June), two in another."""
    sales_rows = [
        sales_row(index=str(i), postcode="2000", contract_dt="20260605", sale_price="1000000")
        for i in range(6)
    ] + [
        sales_row(index=str(10 + i), postcode="2327", contract_dt="20260605", sale_price="500000")
        for i in range(2)
    ]
    rent_rows = [("2026-06-10", "2000", "H", "3", "1000")] * 6 + [
        ("2026-06-10", "2327", "H", "3", "500")
    ] * 2
    return (
        clean_sales(spark.createDataFrame(sales_rows, SALES_COLUMNS)),
        clean_rent(spark.createDataFrame(rent_rows, RENT_COLUMNS)),
    )


class TestGold:
    def test_yield_is_ratio_of_averages(self, spark):
        silver_sales, silver_rent = _yield_inputs(spark)
        got = gold_yield_monthly(silver_sales, silver_rent).collect()
        assert len(got) == 1  # the n>=5 floor removes the thin 2327 cell
        row = got[0]
        assert row.postcode == "2000"
        assert row.gross_yield_pct == pytest.approx(52 * 1000 / 1_000_000 * 100, abs=0.01)
        # additive legs survive so any rollup can recompute the ratio
        assert row.total_sale_value == 6_000_000 and row.n_sold == 6

    def test_thin_cells_excluded_but_present_in_marts(self, spark):
        silver_sales, silver_rent = _yield_inputs(spark)
        marts = gold_sales_monthly(silver_sales)
        assert marts.filter("postcode = '2327'").collect()[0].n_sold == 2
        assert gold_rent_monthly(silver_rent).filter("postcode = '2327'").count() == 1

    def test_gold_excludes_invalid_rows(self, spark):
        silver_sales, _ = _yield_inputs(spark)
        flagged = clean_sales(
            silver_sales.sparkSession.createDataFrame(
                [sales_row(index="99", sale_price="1", postcode="2000")], SALES_COLUMNS
            )
        )
        combined = silver_sales.unionByName(flagged)
        total = gold_sales_monthly(combined).agg({"n_sold": "sum"}).collect()[0][0]
        assert total == 8  # the $1 transfer is not counted

    def test_quality_summary_accounts_for_every_row(self, spark):
        _, silver_rent = _yield_inputs(spark)
        summary = quality_summary(silver_rent, "rent_month")
        assert summary.agg({"rows": "sum"}).collect()[0][0] == 8
