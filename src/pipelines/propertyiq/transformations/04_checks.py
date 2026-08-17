"""Checks — the part expectations do NOT fully replace.

This file is the honest bit of the comparison, and the most useful thing to be
able to say about declarative pipelines in an interview.

Expectations are **row-level**: each one is a SQL boolean evaluated per row. The
yield plausibility band is row-level, so it moved cleanly onto the table it
guards (see 03_gold.py) and this file does not repeat it.

But four of the job's checks are **aggregate** assertions:

  - three grain-uniqueness tests  (GROUP BY ... HAVING count(*) > 1)
  - three coverage floors         (HAVING count(DISTINCT postcode) < 10)

No expectation can express those, because there is no single row to evaluate.
The pattern below is the idiomatic workaround: compute the violations as a
dataset, then attach a row-level expectation asserting the count is zero. It
works, and it is still less code than the job's 04_checks notebook — but it is
a workaround, not a feature, and claiming "expectations replaced my tests" would
be overselling it.

`private=True` keeps these out of Unity Catalog: they are pipeline-internal
assertions, not something anyone should query.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

_GRAINS = {
    "gold_property_sales": ["postcode", "suburb", "property_type", "area_band", "zoning", "month"],
    "gold_property_rent": ["postcode", "property_type", "bedroom_band", "month"],
    "gold_property_yield": ["postcode", "property_type", "month"],
}

# Same floors as the job's 04_checks notebook: a build that produces almost
# nothing is a broken build, even when every individual row in it looks fine.
_COVERAGE_FLOORS = {
    "gold_property_sales": 10,
    "gold_property_rent": 10,
    "gold_property_yield": 5,
}


@dp.materialized_view(
    name="_check_grain_uniqueness",
    private=True,
    comment="One row per mart: how many grain keys appear more than once. Must be 0.",
)
@dp.expect_or_fail("grain_is_unique", "duplicate_keys = 0")
def _check_grain_uniqueness():
    frames = []
    for table, keys in _GRAINS.items():
        dupes = (
            spark.read.table(table)  # noqa: F821
            .groupBy(*keys)
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        frames.append((table, dupes))
    return spark.createDataFrame(frames, "mart STRING, duplicate_keys BIGINT")  # noqa: F821


@dp.materialized_view(
    name="_check_coverage",
    private=True,
    comment="One row per mart: distinct postcodes against the floor. shortfall must be 0.",
)
@dp.expect_or_fail("coverage_above_floor", "shortfall = 0")
def _check_coverage():
    frames = []
    for table, floor in _COVERAGE_FLOORS.items():
        n = spark.read.table(table).select("postcode").distinct().count()  # noqa: F821
        frames.append((table, n, floor, max(0, floor - n)))
    schema = "mart STRING, distinct_postcodes BIGINT, floor BIGINT, shortfall BIGINT"
    return spark.createDataFrame(frames, schema)  # noqa: F821
