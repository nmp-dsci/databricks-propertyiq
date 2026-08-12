# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — incremental ingest with Auto Loader
# MAGIC
# MAGIC Bronze is **append-only and lossless**: raw payload, plus ingest metadata.
# MAGIC No filtering, no dedup, no type coercion beyond the declared schema. If the
# MAGIC silver logic turns out to be wrong we can always rebuild from here.
# MAGIC
# MAGIC Uses Auto Loader (`cloudFiles`) with `Trigger.AvailableNow` — streaming
# MAGIC semantics (exactly-once file tracking via the checkpoint, schema evolution
# MAGIC via `_rescued_data`) on a batch cadence and a batch bill.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_spike")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

landing_path = f"/Volumes/{catalog}/{schema}/landing/orders"
checkpoint_path = f"/Volumes/{catalog}/{schema}/landing/_checkpoints/bronze_orders"
schema_path = f"/Volumes/{catalog}/{schema}/landing/_schemas/bronze_orders"
target = f"{catalog}.{schema}.bronze_orders"

# COMMAND ----------

from pyspark.sql import functions as F

# Declaring the schema beats inferring it: no surprise type flips between runs,
# and anything unexpected in the file lands in _rescued_data instead of failing
# the pipeline or silently vanishing.
ORDER_SCHEMA = """
    order_id     STRING,
    order_ts     STRING,
    customer_id  BIGINT,
    country      STRING,
    channel      STRING,
    category     STRING,
    quantity     BIGINT,
    unit_price   DOUBLE
"""

stream = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", schema_path)
    .option("cloudFiles.schemaEvolutionMode", "rescue")
    .schema(ORDER_SCHEMA)
    .load(landing_path)
    .select(
        "*",
        F.col("_metadata.file_path").alias("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
    )
)

# COMMAND ----------

query = (
    stream.writeStream.format("delta")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(target)
)
query.awaitTermination()

print(f"bronze rows: {spark.table(target).count():,}")

# COMMAND ----------

# Liquid clustering rather than partitioning: cardinality here is low and the
# table is small, so hive-style partitions would just make small files.
spark.sql(f"ALTER TABLE {target} CLUSTER BY (country, category)")

spark.sql(f"""
    COMMENT ON TABLE {target} IS
    'Raw retail orders as landed from the JSON landing zone. Append-only, no
     deduplication or validation applied. Rebuild source for all downstream layers.'
""")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {target} LIMIT 20"))
