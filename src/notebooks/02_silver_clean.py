# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — typed, repaired, quality-flagged
# MAGIC
# MAGIC The cleaning rules live in `src/lib/transforms.py`, not in this notebook —
# MAGIC unit-tested locally (`make test`) with no workspace, so a rule change gets
# MAGIC a red test in seconds rather than a job run. The rules themselves are a
# MAGIC port of the dbt models in data-qa-agent (stg_sales, stg_rent), which
# MAGIC cleaned this exact data for Postgres.
# MAGIC
# MAGIC Silver **keeps every row** and attaches a verdict (`_is_valid`,
# MAGIC `_quality`). The dbt originals drop failures in a WHERE clause; here gold
# MAGIC applies the filter instead, so "what happened to the missing rows?" is a
# MAGIC query, not an archaeology project.

# COMMAND ----------

import os
import sys

# Make `src/` importable so `from lib.transforms import ...` works both here and
# in the local test run. Bundles sync the whole tree, so the sibling is present.
sys.path.insert(0, os.path.abspath(".."))

from lib.transforms import clean_rent, clean_sales, resolve_versions  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "propertyiq")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

# resolve_versions first: landing is append-only, so a partition rewritten
# upstream (rentboard rewrites its trailing month every run) sits in bronze as
# two files. Keeping both would double-count that month. See transforms.py.
silver_sales = clean_sales(resolve_versions(spark.table(f"{catalog}.{schema}.bronze_sales")))
(
    silver_sales.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .clusterBy("postcode", "sale_month")
    .saveAsTable(f"{catalog}.{schema}.silver_sales")
)

silver_rent = clean_rent(resolve_versions(spark.table(f"{catalog}.{schema}.bronze_rent")))
(
    silver_rent.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .clusterBy("postcode", "rent_month")
    .saveAsTable(f"{catalog}.{schema}.silver_rent")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constraints, not just conventions
# MAGIC
# MAGIC Delta constraints make the contract enforceable — a future write that
# MAGIC violates them fails loudly instead of quietly poisoning gold.

# COMMAND ----------

for table, key in (("silver_sales", "sale_id"), ("silver_rent", "rent_id")):
    spark.sql(f"ALTER TABLE {catalog}.{schema}.{table} ALTER COLUMN {key} SET NOT NULL")

# COMMAND ----------

from pyspark.sql import functions as F

for table in ("silver_sales", "silver_rent"):
    df = spark.table(f"{catalog}.{schema}.{table}")
    total = df.count()
    print(f"-- {table}: {total:,} rows --")
    display(
        df.groupBy("_is_valid")
        .agg(F.count("*").alias("rows"))
        .withColumn("pct", F.round(F.col("rows") * 100.0 / total, 2))
    )

# COMMAND ----------

# Why did rows fail? This is the payoff of flag-don't-drop: the answer is a
# one-line query over the same table the marts are built from.
display(
    spark.table(f"{catalog}.{schema}.silver_sales")
    .select(F.explode("_quality").alias("reason"))
    .groupBy("reason")
    .count()
    .orderBy(F.desc("count"))
)
