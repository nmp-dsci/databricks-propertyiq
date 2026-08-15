"""Pure DataFrame transforms for the PropertyIQ pipeline.

Notebooks are for orchestration and narrative; the logic that decides what
"clean" means lives here so it can be unit-tested locally (see tests/) without a
workspace, a warehouse, or a network.

Every rule below is a port of tested SQL from the dbt project in
data-qa-agent/services/data-pipeline/dbt (stg_sales, stg_rent, the three marts),
with one deliberate divergence: dbt *drops* failing rows in a WHERE clause,
while silver here *keeps* them with a `_quality` array of named reasons and gold
applies the filter. Same numbers out of the marts; the difference is that "why
did 12% of rows drop?" becomes a query instead of an archaeology project.

Every function takes a DataFrame and returns a DataFrame. No I/O, no globals.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# dbt stg_sales: "$10k floor removes non-arms-length transfers (family /
# deceased-estate transfers recorded at nominal consideration)". The $8M cap
# matches the original pandas analysis this was ported from.
MIN_SALE_PRICE = 10_000
MAX_SALE_PRICE = 8_000_000
MIN_SALE_YEAR = 2010

# mart_property_yield: cells with fewer sales or bonds than this in a month are
# too thin to yield a meaningful ratio and are excluded from the join.
MIN_MONTHLY_VOLUME = 5


# Anything a pandas export can do to a number: "668", "809.43", "2.0150416E7".
_NUMERIC = r"^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$"


def _safe_double(col: Column) -> Column:
    """Cast to double with a regex guard, so junk becomes NULL, never an error.

    The guard matters twice over: local Spark runs with ANSI off (bad casts
    return NULL) but serverless runs ANSI on (bad casts *throw*), and
    Column.try_cast is not available in the local PySpark. A regex-guarded
    plain cast behaves identically in both worlds.
    """
    trimmed = F.trim(col)
    return F.when(trimmed.rlike(_NUMERIC), trimmed.cast("double"))


def _digits_from_floatish(col: Column) -> Column:
    """Recover an integer string from any of the source's numeric encodings.

    The sales data carries three encodings of the same value, depending on
    which era wrote it: raw DAT strings ("20150416"), pandas float round-trips
    ("20150416.0"), and scientific notation from the legacy monolith export
    ("2.0150416E7"). Casting through double normalises all three; doubles hold
    9-digit integers exactly, so nothing is lost. Junk becomes NULL.
    """
    return _safe_double(col).cast("bigint").cast("string")


def resolve_versions(df: DataFrame) -> DataFrame:
    """Keep only the newest landed file per partition.

    The landing area is append-only, so a partition that gets rewritten
    upstream arrives as a second file rather than replacing the first:

        landing/lodgements/month=2026-06_f720a07c.parquet   <- first run
        landing/lodgements/month=2026-06_9c4e11ab.parquet   <- month rewritten

    Both are in bronze (correctly — bronze is lossless). Silver has to pick one,
    or every row of a revised month is counted twice. The partition label is
    everything before the `_<sha8>.parquet` suffix, and the winner is the file
    with the latest `_ingested_at`; ties break on the filename so the result is
    deterministic when a bootstrap lands many files in the same second.

    Immutable partitions (all of sales) have exactly one file per label, so this
    is a no-op for them — it is applied uniformly anyway, because "sales never
    gets restated" is an upstream assumption rather than a guarantee.

    Rows that did not come from a landing file are dropped. Bronze is history,
    so it still holds everything ingested from the pre-Parquet feed — the two
    monolith CSVs at the volume root — and those rows carry the same sales and
    bonds as the partitions do. Keeping them would double every figure in gold.
    Membership is decided by the filename contract (`<partition>_<sha8>.parquet`)
    rather than by a date cutoff, so it stays correct however often the feed is
    replayed or bronze is rebuilt.
    """
    labelled = df.withColumn(
        "_partition",
        F.regexp_extract(F.col("_source_file"), r"([^/]+)_[0-9a-f]{8}\.parquet$", 1),
    ).filter(F.col("_partition") != "")
    newest = Window.partitionBy("_partition").orderBy(
        F.col("_ingested_at").desc(), F.col("_source_file").desc()
    )
    return (
        labelled.withColumn("_version_rank", F.dense_rank().over(newest))
        .filter(F.col("_version_rank") == 1)
        .drop("_version_rank", "_partition")
    )


def clean_sales(df: DataFrame) -> DataFrame:
    """Bronze sales (all-string) -> typed, repaired, quality-flagged silver.

    Ports stg_sales.sql. Column selection mirrors dbt's: ymd/settle_dt/
    dealing_no and the other dropped columns stay behind in bronze, which is
    lossless — nothing is thrown away, just not carried forward.
    """
    contract_ymd = _digits_from_floatish(F.col("contract_dt"))
    price = _safe_double(F.col("sale_price"))
    area_raw = _safe_double(F.col("area_sqm"))
    area_type = F.upper(F.nullif(F.trim(F.col("area_type")), F.lit("")))

    typed = (
        df.withColumn("contract_ymd", contract_ymd)
        # try_to_timestamp, not to_date: the data really does contain 6-digit
        # dates ('170823'), and under serverless's ANSI mode to_date *throws*
        # on those where local Spark quietly returns NULL. try_* is the only
        # parse that behaves identically in both.
        .withColumn(
            "sale_date",
            F.when(
                F.col("contract_ymd").rlike(r"^\d{8}$"),
                F.try_to_timestamp(F.col("contract_ymd"), F.lit("yyyyMMdd")).cast("date"),
            ),
        )
        .withColumn("sale_price", price)
        # "2327.0" -> "2327"; stays a string — postcodes are labels, not numbers.
        .withColumn("postcode", _digits_from_floatish(F.col("postcode")))
        .withColumn("property_id", _digits_from_floatish(F.col("property_id")))
        # dbt: strata_no empty means freestanding house, present means a unit in
        # a strata plan. Empty *string* check, not NULL — blank CSV cells load
        # as ''.
        .withColumn(
            "property_type",
            F.when(F.coalesce(F.col("strata_no"), F.lit("")) == "", "house").otherwise("unit"),
        )
        .withColumn("suburb", F.initcap(F.trim(F.col("locality"))))
        # Area arrives in hectares (H) or square metres (M); anything else is
        # unusable. Standardise to sqm so bands mean one thing.
        .withColumn(
            "area_sqm",
            F.when(area_type == "H", F.round(area_raw * 10_000))
            .when(area_type == "M", F.round(area_raw))
            .otherwise(F.lit(None).cast("double")),
        )
        .withColumn("zoning", F.nullif(F.trim(F.col("zoning")), F.lit("")))
    )

    quality = F.array_compact(
        F.array(
            F.when(F.col("sale_date").isNull(), F.lit("unparseable_contract_date")),
            F.when(F.year("sale_date") < MIN_SALE_YEAR, F.lit("pre_2010_sale")),
            F.when(F.col("sale_price").isNull(), F.lit("unparseable_price")),
            F.when(F.col("sale_price") < MIN_SALE_PRICE, F.lit("non_arms_length_price")),
            F.when(F.col("sale_price") > MAX_SALE_PRICE, F.lit("price_above_cap")),
            F.when(F.col("prop_purpose") != "RESIDENCE", F.lit("not_residence")),
            F.when(F.coalesce(F.trim(F.col("locality")), F.lit("")) == "", F.lit("blank_locality")),
            F.when(F.col("postcode").isNull(), F.lit("bad_postcode")),
        )
    )

    # dbt learned the hard way that property_id + date + price is not unique
    # (split/fractional-interest settlements share all three), so a plain hash
    # collides. Hash the business key, then number the rows *within* each hash
    # deterministically — unique unconditionally, and stable across rebuilds,
    # unlike dbt's global row_number().
    #
    # Deliberately no dedup or aggregation here, matching dbt: split
    # settlements stay as separate rows, each with its own sale_id. Collapsing
    # part-payments to one row per property was considered and rejected — it
    # would skew avg/median away from what the dbt marts report, since the
    # real aggregation to postcode/suburb/type/area_band/zoning/month happens
    # in gold, not here.
    key_hash = F.sha2(
        F.concat_ws(
            "|",
            F.coalesce(F.col("property_id"), F.lit("")),
            F.coalesce(F.col("contract_ymd"), F.lit("")),
            F.coalesce(F.col("sale_price").cast("string"), F.lit("")),
            F.col("property_type"),
        ),
        256,
    )
    dedup_window = Window.partitionBy("_sale_key").orderBy(
        F.col("file"), F.col("index"), F.col("_source_row")
    )

    flagged = (
        typed.withColumn("_quality", quality)
        .withColumn("_is_valid", F.size(F.col("_quality")) == 0)
        .withColumn("sale_month", F.trunc("sale_date", "month"))
        .withColumn("_sale_key", key_hash)
        .withColumn("_source_row", F.monotonically_increasing_id())
        .withColumn(
            "sale_id",
            F.concat_ws("-", F.substring("_sale_key", 1, 16), F.row_number().over(dedup_window)),
        )
        .drop("_source_row")
    )

    area_band = (
        F.when(F.col("area_sqm").isNull(), F.lit(None).cast("string"))
        .when(F.col("area_sqm") < 400, "<400")
        .when(F.col("area_sqm") < 700, "400-700")
        .when(F.col("area_sqm") < 1000, "700-1000")
        .when(F.col("area_sqm") < 5000, "1000-5000")
        .otherwise("5000+")
    )

    return flagged.withColumn("area_band", area_band).select(
        "sale_id",
        "property_id",
        "suburb",
        "postcode",
        "property_type",
        "sale_date",
        "sale_month",
        "sale_price",
        "area_sqm",
        "area_band",
        "zoning",
        "_quality",
        "_is_valid",
    )


def clean_rent(df: DataFrame) -> DataFrame:
    """Bronze rent bonds (all-string) -> typed, quality-flagged silver.

    Ports stg_rent.sql. The one subtlety worth saying out loud: property_type
    is mapped onto the *sales* vocabulary (H/T -> house, everything else ->
    unit) so the yield join compares like with like.
    """
    bedrooms = _digits_from_floatish(F.col("bedrooms")).cast("int")
    rent = _safe_double(F.col("weekly_rent"))

    typed = (
        df.withColumn(
            "rent_date",
            F.when(
                F.col("lodgement_dt").rlike(r"^\d{4}-\d{2}-\d{2}"),
                # try_*: a '9999-99-99' must become NULL under serverless ANSI
                # mode, not a job failure.
                F.try_to_timestamp(F.substring("lodgement_dt", 1, 10)).cast("date"),
            ),
        )
        .withColumn("postcode", _digits_from_floatish(F.col("postcode")))
        .withColumn("property_type_code", F.upper(F.trim(F.col("property_type"))))
        .withColumn(
            "property_type",
            F.when(F.col("property_type_code").isin("H", "T"), "house").otherwise("unit"),
        )
        .withColumn("bedrooms", bedrooms)
        .withColumn(
            "bedroom_band",
            F.when(bedrooms.isNull(), "unknown")
            .when(bedrooms >= 5, "5+")
            .otherwise(bedrooms.cast("string")),
        )
        .withColumn("weekly_rent", rent)
    )

    quality = F.array_compact(
        F.array(
            F.when(F.col("rent_date").isNull(), F.lit("unparseable_lodgement_date")),
            F.when(F.year("rent_date") < MIN_SALE_YEAR, F.lit("pre_2010_lodgement")),
            F.when(F.col("weekly_rent").isNull(), F.lit("unparseable_rent")),
            F.when(F.col("weekly_rent") <= 0, F.lit("non_positive_rent")),
            F.when(F.col("postcode").isNull(), F.lit("bad_postcode")),
        )
    )

    # No id-like column exists at all upstream, and two genuinely distinct
    # bonds can share every column — same hash-then-number trick as sales.
    key_hash = F.sha2(
        F.concat_ws(
            "|",
            F.coalesce(F.col("rent_date").cast("string"), F.lit("")),
            F.coalesce(F.col("postcode"), F.lit("")),
            F.col("property_type"),
            F.col("bedroom_band"),
            F.coalesce(F.col("weekly_rent").cast("string"), F.lit("")),
        ),
        256,
    )
    dedup_window = Window.partitionBy("_rent_key").orderBy(F.col("_source_row"))

    return (
        typed.withColumn("_quality", quality)
        .withColumn("_is_valid", F.size(F.col("_quality")) == 0)
        .withColumn("rent_month", F.trunc("rent_date", "month"))
        .withColumn("_rent_key", key_hash)
        .withColumn("_source_row", F.monotonically_increasing_id())
        .withColumn(
            "rent_id",
            F.concat_ws("-", F.substring("_rent_key", 1, 16), F.row_number().over(dedup_window)),
        )
        .select(
            "rent_id",
            "rent_date",
            "rent_month",
            "postcode",
            "property_type",
            "property_type_code",
            "bedrooms",
            "bedroom_band",
            "weekly_rent",
            "_quality",
            "_is_valid",
        )
    )


def gold_sales_monthly(df: DataFrame) -> DataFrame:
    """mart_property_sales: additive legs + medians at the dbt grain.

    Additive columns (n_sold, total_sale_value) come first-class so any rollup
    — quarter, SA3, state — recomputes averages correctly by summing legs.
    Medians are convenience columns and are NOT additive; the dbt docs say so
    and the UC column comments will too.
    """
    return (
        df.filter(F.col("_is_valid"))
        .groupBy(
            "postcode",
            "suburb",
            "property_type",
            F.coalesce(F.col("area_band"), F.lit("unknown")).alias("area_band"),
            F.coalesce(F.col("zoning"), F.lit("unknown")).alias("zoning"),
            F.col("sale_month").alias("month"),
        )
        .agg(
            F.sum("sale_price").alias("total_sale_value"),
            F.count("*").alias("n_sold"),
            F.round(F.avg("sale_price")).alias("avg_sale_price"),
            F.round(F.expr("percentile(sale_price, 0.5)")).alias("median_sale_price"),
            F.min("sale_price").alias("min_sale_price"),
            F.max("sale_price").alias("max_sale_price"),
        )
    )


def gold_rent_monthly(df: DataFrame) -> DataFrame:
    """mart_property_rent: same shape over weekly rents."""
    return (
        df.filter(F.col("_is_valid"))
        .groupBy(
            "postcode",
            "property_type",
            "bedroom_band",
            F.col("rent_month").alias("month"),
        )
        .agg(
            F.sum("weekly_rent").alias("total_weekly_rent"),
            F.count("*").alias("n_rented"),
            F.round(F.avg("weekly_rent")).alias("avg_weekly_rent"),
            F.round(F.expr("percentile(weekly_rent, 0.5)")).alias("median_weekly_rent"),
            F.min("weekly_rent").alias("min_weekly_rent"),
            F.max("weekly_rent").alias("max_weekly_rent"),
        )
    )


def gold_yield_monthly(silver_sales: DataFrame, silver_rent: DataFrame) -> DataFrame:
    """mart_property_yield: gross yield where both sides have real volume.

    Ported verbatim from the dbt model, including its two load-bearing choices:
    the join re-aggregates from *silver* (not the other marts) so each side is
    at exactly postcode+type+month, and the yield is a ratio of averages —
    never an average of per-row yields, which over-weights cheap properties.
    Cells with fewer than MIN_MONTHLY_VOLUME on either side are excluded: a
    two-sale month produces a headline, not a statistic.
    """
    sales = (
        silver_sales.filter(F.col("_is_valid"))
        .groupBy("postcode", "property_type", F.col("sale_month").alias("month"))
        .agg(F.sum("sale_price").alias("total_sale_value"), F.count("*").alias("n_sold"))
    )
    rent = (
        silver_rent.filter(F.col("_is_valid"))
        .groupBy("postcode", "property_type", F.col("rent_month").alias("month"))
        .agg(F.sum("weekly_rent").alias("total_weekly_rent"), F.count("*").alias("n_rented"))
    )
    avg_price = F.col("total_sale_value") / F.col("n_sold")
    avg_rent = F.col("total_weekly_rent") / F.col("n_rented")
    return (
        sales.join(rent, ["postcode", "property_type", "month"])
        .filter((F.col("n_sold") >= MIN_MONTHLY_VOLUME) & (F.col("n_rented") >= MIN_MONTHLY_VOLUME))
        .select(
            "postcode",
            "property_type",
            "month",
            "total_sale_value",
            "n_sold",
            "total_weekly_rent",
            "n_rented",
            F.round(avg_price).alias("avg_sale_price"),
            F.round(avg_rent).alias("avg_weekly_rent"),
            F.round(52 * avg_rent / avg_price * 100, 2).alias("gross_yield_pct"),
        )
    )


def quality_summary(df: DataFrame, month_col: str) -> DataFrame:
    """Rows per month per quality reason; valid rows count under 'ok'.

    One shape for both datasets — the notebook adds a `dataset` literal and
    unions them, so the quality dashboard is a single table scan.
    """
    return (
        df.select(F.col(month_col).alias("month"), F.explode_outer("_quality").alias("reason"))
        .withColumn("reason", F.coalesce(F.col("reason"), F.lit("ok")))
        .groupBy("month", "reason")
        .agg(F.count("*").alias("rows"))
    )
