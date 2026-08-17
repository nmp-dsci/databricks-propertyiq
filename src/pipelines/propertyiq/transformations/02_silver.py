"""Silver — materialized views, and the same tested transform functions.

The important structural point: silver is a **materialized view**, not a
streaming table. `resolve_versions()` is a window function ranking every landed
file per partition — a full scan over all of bronze, not an append-only
operation — and a streaming table cannot express that. A materialized view over
a batch read (`spark.read.table`, no STREAM) is the declarative equivalent of
the job's `.mode("overwrite")` full rebuild.

That mapping is worth noticing: the "full recompute" choice made in the job was
not a limitation to be apologised for. It is exactly what an MV is.

The cleaning rules are IDENTICAL to the job's — literally the same functions
from `src/lib/transforms.py`, unit-tested by `make test` with no workspace. Only
the wiring differs, which is what makes this a fair comparison.

Expectations here are `@dp.expect` (warn), not `expect_or_drop`. That preserves
the deliberate design decision behind the job: invalid rows stay in silver with
a `_quality` array naming what they failed, and gold applies the filter — so the
quality dashboard and the price dashboard read the same table. The difference is
that the violation counts now appear in the pipeline UI for free, instead of
being hand-rolled into `gold_quality_summary`.
"""

from pyspark import pipelines as dp

from lib.transforms import clean_rent, clean_sales, resolve_versions

_SALES_RULES = {
    "has_sale_month": "sale_month IS NOT NULL",
    "has_postcode": "postcode IS NOT NULL",
    "price_within_band": "sale_price IS NULL OR sale_price BETWEEN 10000 AND 8000000",
}

_RENT_RULES = {
    "has_rent_month": "rent_month IS NOT NULL",
    "has_postcode": "postcode IS NOT NULL",
    "rent_positive": "weekly_rent IS NULL OR weekly_rent > 0",
}


@dp.materialized_view(
    name="silver_sales",
    cluster_by=["postcode", "sale_month"],
    comment="Typed, repaired and quality-flagged sales. Invalid rows kept, flagged in _quality.",
)
@dp.expect_all(_SALES_RULES)
def silver_sales():
    return clean_sales(resolve_versions(spark.read.table("bronze_sales")))  # noqa: F821


@dp.materialized_view(
    name="silver_rent",
    cluster_by=["postcode", "rent_month"],
    comment="Typed, repaired and quality-flagged rental lodgements. Invalid rows kept.",
)
@dp.expect_all(_RENT_RULES)
def silver_rent():
    return clean_rent(resolve_versions(spark.read.table("bronze_rent")))  # noqa: F821
