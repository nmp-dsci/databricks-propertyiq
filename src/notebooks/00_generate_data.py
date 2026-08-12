# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Generate synthetic retail orders
# MAGIC
# MAGIC Free Edition has **restricted outbound internet**, so we never download a
# MAGIC dataset — we synthesise one with Spark and land it as JSON files in a Unity
# MAGIC Catalog Volume. That volume is the "landing zone" the bronze layer ingests
# MAGIC from, which keeps the medallion story honest (files in, tables out) instead
# MAGIC of a DataFrame that magically appears.
# MAGIC
# MAGIC Dirt is injected on purpose — nulls, duplicates, negative quantities, junk
# MAGIC country codes — so the silver layer has real work to do.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_spike")
dbutils.widgets.text("rows", "500000")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
rows = int(dbutils.widgets.get("rows"))

volume = "landing"
landing_path = f"/Volumes/{catalog}/{schema}/{volume}/orders"

print(f"target: {catalog}.{schema}  rows={rows:,}  landing={landing_path}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

# Idempotent: every run starts from a clean landing zone.
dbutils.fs.rm(landing_path, recurse=True)

# COMMAND ----------

from pyspark.sql import functions as F

# Deterministic seeds keep the demo reproducible across runs.
SEED = 42

countries = ["AU", "NZ", "US", "GB", "SG", "au", "  US ", "ZZ"]  # last 3 are dirt
channels = ["web", "store", "app", "partner"]
categories = ["electronics", "apparel", "grocery", "home", "sport"]

df = (
    spark.range(rows)
    .withColumn("order_id", F.concat(F.lit("ORD-"), F.lpad(F.col("id").cast("string"), 10, "0")))
    .withColumn("rnd", F.rand(SEED))
    # Orders spread over the last 180 days, weighted so recent days are busier.
    .withColumn(
        "order_ts",
        F.expr("timestamp(current_timestamp() - make_interval(0, 0, 0, cast(pow(rand(7), 1.6) * 180 as int)))"),
    )
    .withColumn("customer_id", (F.rand(SEED + 1) * 25000).cast("int") + 1)
    .withColumn("country", F.element_at(F.array(*[F.lit(c) for c in countries]),
                                        (F.rand(SEED + 2) * len(countries)).cast("int") + 1))
    .withColumn("channel", F.element_at(F.array(*[F.lit(c) for c in channels]),
                                        (F.rand(SEED + 3) * len(channels)).cast("int") + 1))
    .withColumn("category", F.element_at(F.array(*[F.lit(c) for c in categories]),
                                         (F.rand(SEED + 4) * len(categories)).cast("int") + 1))
    .withColumn("quantity", (F.rand(SEED + 5) * 5).cast("int") + 1)
    .withColumn("unit_price", F.round(F.rand(SEED + 6) * 240 + 5, 2))
    # ~1.5% of rows get a negative quantity (returns mis-keyed as orders).
    .withColumn("quantity", F.when(F.rand(SEED + 7) < 0.015, -F.col("quantity")).otherwise(F.col("quantity")))
    # ~2% lose their customer id.
    .withColumn("customer_id", F.when(F.rand(SEED + 8) < 0.02, F.lit(None)).otherwise(F.col("customer_id")))
    .select("order_id", "order_ts", "customer_id", "country", "channel", "category",
            "quantity", "unit_price")
)

# ~1% exact duplicates, simulating an at-least-once delivery upstream.
dupes = df.sample(fraction=0.01, seed=SEED)
orders = df.unionByName(dupes)

# COMMAND ----------

# Land as JSON files. Multiple files (not one big one) so Auto Loader in the
# bronze notebook has something realistic to incrementally discover.
(
    orders.repartition(8)
    .write.mode("overwrite")
    .json(landing_path)
)

landed = spark.read.json(landing_path).count()
print(f"landed {landed:,} rows across {len(dbutils.fs.ls(landing_path))} files")

# COMMAND ----------

display(spark.read.json(landing_path).limit(20))
