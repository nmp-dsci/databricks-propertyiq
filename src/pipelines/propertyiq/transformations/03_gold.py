"""Gold — four materialized views, comments carried across verbatim.

Same four marts as `src/notebooks/03_gold_aggregate.py`, same functions, same
`COMMENT` text. The comments matter as much here as there: Genie and the
Assistant read them when turning a question into SQL, and the wording was
already proven as NL2SQL grounding in the sibling dbt project.

Two things the job had to do by hand and this does not:

  - `save()` — a helper wrapping `.write.mode("overwrite").saveAsTable()`,
    repeated per mart. Here the decorator is the write.
  - `depends_on:` in the job YAML. The DAG is inferred from the fact that these
    functions read `silver_sales` / `silver_rent`.

The plausibility check that was a separate task in the job is now an expectation
attached to the table it guards — `expect_or_fail`, matching the job's behaviour
of failing the run rather than serving a wrong number.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

from lib.transforms import (
    gold_rent_monthly,
    gold_sales_monthly,
    gold_yield_monthly,
    quality_summary,
)


@dp.materialized_view(
    name="gold_property_sales",
    comment=(
        "NSW residential property sales, aggregated monthly. One row per postcode / "
        "suburb / property_type (house|unit) / area_band / zoning / month. "
        "total_sale_value and n_sold are additive — sum them before dividing when "
        "rolling up. median_sale_price is NOT additive across buckets. Thin cells "
        "are kept: filter on n_sold in the query, not by excluding rows here. "
        "Excludes rows failing quality checks (non-arms-length transfers under "
        "$10k, unparseable dates, non-residential)."
    ),
)
def gold_property_sales():
    return gold_sales_monthly(spark.read.table("silver_sales"))  # noqa: F821


@dp.materialized_view(
    name="gold_property_rent",
    comment=(
        "NSW rental bond lodgements, aggregated monthly. One row per postcode / "
        "property_type (house|unit) / bedroom_band (unknown,0-4,5+) / month. "
        "total_weekly_rent and n_rented are additive legs; medians are not. "
        "Bedroom bands only exist here — sales carry area_band instead."
    ),
)
def gold_property_rent():
    return gold_rent_monthly(spark.read.table("silver_rent"))  # noqa: F821


@dp.materialized_view(
    name="gold_property_yield",
    comment=(
        "Gross rental yield by postcode / property_type / month, joined on exactly "
        "those three keys — never join sales to rent on suburb. gross_yield_pct = "
        "52 * avg weekly rent / avg sale price * 100, a ratio of averages. Cells "
        "with fewer than 5 sales or 5 bonds in the month are excluded as too thin. "
        "To roll up, sum the four additive legs and recompute the ratio."
    ),
)
# The job ran this as a post-load check in task 04 and raised RuntimeError.
# Here it guards the table directly: Australian residential gross yields sit
# roughly 1-12%, so outside 0.3-25% is a units bug, not a market.
@dp.expect_or_fail("yield_plausible", "gross_yield_pct BETWEEN 0.3 AND 25")
def gold_property_yield():
    return gold_yield_monthly(
        spark.read.table("silver_sales"),  # noqa: F821
        spark.read.table("silver_rent"),  # noqa: F821
    )


@dp.materialized_view(
    name="gold_quality_summary",
    comment=(
        "Data quality rollup: rows per month per named reason per dataset "
        "(nsw_sales | nsw_rent). Valid rows count under reason = ok. Reads the same "
        "silver tables the marts are built from, so the quality story and the "
        "revenue story can never disagree."
    ),
)
def gold_quality_summary():
    sales = spark.read.table("silver_sales")  # noqa: F821
    rent = spark.read.table("silver_rent")  # noqa: F821
    return (
        quality_summary(sales, "sale_month")
        .withColumn("dataset", F.lit("nsw_sales"))
        .unionByName(quality_summary(rent, "rent_month").withColumn("dataset", F.lit("nsw_rent")))
    )
