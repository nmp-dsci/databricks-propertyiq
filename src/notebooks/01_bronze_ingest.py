# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — incremental ingest with Auto Loader
# MAGIC
# MAGIC Two sources land here from the `propertyiq` volume: NSW Valuer General
# MAGIC sales (28 columns) and Rental Bond Board lodgements (5 columns).
# MAGIC
# MAGIC Bronze is **append-only and lossless**: every column as a string, exactly
# MAGIC as the file carried it, plus ingest metadata. No typing, no repair — the
# MAGIC source data has three different numeric encodings depending on which era
# MAGIC wrote it, and absorbing that mess is silver's job, in code that has unit
# MAGIC tests. If a silver rule turns out wrong we rebuild from here.
# MAGIC
# MAGIC Auto Loader (`cloudFiles`) with `Trigger.AvailableNow` gives streaming
# MAGIC semantics — exactly-once file tracking via the checkpoint — on a batch
# MAGIC cadence. Today it reads the two uploaded CSVs; when the updates-only
# MAGIC Parquet feed lands in `landing/`, only the options cell changes.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "propertyiq")
dbutils.widgets.text("volume", "propertyiq")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
state_root = f"{volume_root}/_pipeline"

# COMMAND ----------

from pyspark.sql import functions as F

# Everything is a STRING on purpose — this is the landing contract, pinned by
# the source repo's own tests (propertyiq_getdata FINAL_COLUMNS). Declaring it
# beats inferring it: no surprise type flips between runs, and anything
# unexpected lands in _rescued_data instead of failing the pipeline.
SALES_SCHEMA = """
    file STRING, fn_src STRING, ymd STRING, index STRING, area_sqm STRING,
    area_type STRING, component_cd STRING, contract_dt STRING, create_dt STRING,
    dealing_no STRING, district_code STRING, house_no STRING, locality STRING,
    postcode STRING, prop_name STRING, prop_nature STRING, prop_purpose STRING,
    property_id STRING, record_type STRING, sale_cd STRING, sale_counter STRING,
    sale_interest STRING, sale_price STRING, settle_dt STRING, strata_no STRING,
    street_name STRING, unit_no STRING, zoning STRING
"""

RENT_SCHEMA = """
    lodgement_dt STRING, postcode STRING, property_type STRING,
    bedrooms STRING, weekly_rent STRING
"""


def ingest(name: str, ddl: str, glob: str) -> None:
    """One Auto Loader pass: volume file(s) -> bronze Delta table."""
    target = f"{catalog}.{schema}.bronze_{name}"
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", f"{state_root}/schemas/bronze_{name}")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("pathGlobFilter", glob)
        .option("header", "true")
        .schema(ddl)
        .load(volume_root)
        .select(
            "*",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_ingested_at"),
        )
    )
    (
        stream.writeStream.format("delta")
        .option("checkpointLocation", f"{state_root}/checkpoints/bronze_{name}")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(target)
    ).awaitTermination()
    print(f"{target}: {spark.table(target).count():,} rows")


# COMMAND ----------

ingest("sales", SALES_SCHEMA, "nswgov_df.csv")

# COMMAND ----------

ingest("rent", RENT_SCHEMA, "rentboard_df.csv")

# COMMAND ----------

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.bronze_sales IS
    'NSW Valuer General property sales as landed — every column a raw string,
     append-only, no repair. One row per dealing. Rebuild source for silver.'
""")
spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.bronze_rent IS
    'NSW Rental Bond Board lodgements as landed — raw strings, append-only.
     One row per bond. Rebuild source for silver.'
""")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {catalog}.{schema}.bronze_sales LIMIT 10"))
