"""Pure DataFrame transforms, kept out of the notebooks on purpose.

Notebooks are for orchestration and narrative; the logic that decides what
"clean" means lives here so it can be unit-tested locally (see tests/) without a
workspace, a warehouse, or a network. In an interview this is the difference
between "I wrote some cells" and "I shipped a pipeline someone else can trust".

Every function takes a DataFrame and returns a DataFrame. No I/O, no globals.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# ISO-3166 alpha-2 codes this business actually operates in. Anything else is
# quarantined rather than guessed at.
VALID_COUNTRIES = ("AU", "NZ", "US", "GB", "SG")


def normalise_country(df: DataFrame, column: str = "country") -> DataFrame:
    """Trim and upper-case country codes; unknown codes become NULL.

    The landing data contains `"au"`, `"  US "` and `"ZZ"`. The first two are
    formatting noise and are safe to repair. `ZZ` is not a country we sell in,
    and silently mapping it to something plausible would be worse than losing it.
    """
    cleaned = F.upper(F.trim(F.col(column)))
    return df.withColumn(
        column,
        F.when(cleaned.isin(*VALID_COUNTRIES), cleaned).otherwise(F.lit(None)),
    )


def deduplicate_orders(
    df: DataFrame, key: str = "order_id", order_by: str = "_ingested_at"
) -> DataFrame:
    """Keep one row per order id — the most recently ingested one.

    The upstream delivers at-least-once, so exact duplicates are expected. Using
    a window rather than `dropDuplicates` means that when the duplicate rows are
    *not* identical (a late correction, say) we deterministically keep the
    latest instead of an arbitrary one.
    """
    window = Window.partitionBy(key).orderBy(F.col(order_by).desc(), F.col(key))
    return df.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")


def add_revenue(df: DataFrame) -> DataFrame:
    """Derive line revenue. Rounded to cents so downstream sums are stable."""
    return df.withColumn("revenue", F.round(F.col("quantity") * F.col("unit_price"), 2))


def flag_quality(df: DataFrame) -> DataFrame:
    """Attach a `_quality` array describing why a row is suspect.

    An array of reasons, not a boolean: when someone asks "why did 8% of rows
    drop out?" the answer should be in the table, not in a notebook someone has
    to re-run.
    """
    reasons = F.array_compact(
        F.array(
            F.when(F.col("quantity") <= 0, F.lit("non_positive_quantity")),
            F.when(F.col("unit_price") <= 0, F.lit("non_positive_price")),
            F.when(F.col("customer_id").isNull(), F.lit("missing_customer")),
            F.when(F.col("country").isNull(), F.lit("unknown_country")),
            F.when(F.col("order_id").isNull(), F.lit("missing_order_id")),
        )
    )
    return df.withColumn("_quality", reasons).withColumn(
        "_is_valid", F.size(F.col("_quality")) == 0
    )


def to_silver(df: DataFrame) -> DataFrame:
    """Full bronze -> silver pipeline: typed, normalised, deduped, flagged.

    Note what this does *not* do: it does not drop the invalid rows. Silver keeps
    everything with a verdict attached, and gold decides what to count. That way
    a data quality dashboard and the revenue dashboard read from the same table.
    """
    return (
        df.withColumn("order_ts", F.to_timestamp("order_ts"))
        .withColumn("order_date", F.to_date("order_ts"))
        .transform(normalise_country)
        .transform(add_revenue)
        .transform(deduplicate_orders)
        .transform(flag_quality)
    )


def daily_revenue(df: DataFrame) -> DataFrame:
    """Gold aggregate: one row per date / country / channel / category."""
    return (
        df.filter(F.col("_is_valid"))
        .groupBy("order_date", "country", "channel", "category")
        .agg(
            F.countDistinct("order_id").alias("orders"),
            F.sum("quantity").alias("units"),
            F.round(F.sum("revenue"), 2).alias("revenue"),
            F.countDistinct("customer_id").alias("customers"),
        )
    )


def quality_summary(df: DataFrame) -> DataFrame:
    """Gold aggregate: how many rows failed, and for which reason, per day."""
    return (
        df.select("order_date", F.explode_outer("_quality").alias("reason"))
        .withColumn("reason", F.coalesce(F.col("reason"), F.lit("ok")))
        .groupBy("order_date", "reason")
        .agg(F.count("*").alias("rows"))
    )
