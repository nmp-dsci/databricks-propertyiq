# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — conformed, deduplicated, quality-flagged
# MAGIC
# MAGIC The actual cleaning rules live in `src/lib/transforms.py`, not in this
# MAGIC notebook. That file is unit-tested locally (`make test`) with no workspace
# MAGIC and no cluster, so a rule change gets a red test in seconds rather than a
# MAGIC 4-minute job run.
# MAGIC
# MAGIC Silver **keeps every row** and attaches a verdict (`_is_valid`,
# MAGIC `_quality`). Dropping bad rows here would mean the revenue dashboard and
# MAGIC the data-quality dashboard read from different tables, and the first
# MAGIC question a stakeholder asks — "what happened to the missing 8%?" — would
# MAGIC have no answer in the data.

# COMMAND ----------

import os
import sys

# Make `src/` importable so `from lib.transforms import ...` works both here and
# in the local test run. Bundles sync the whole tree, so the sibling is present.
sys.path.insert(0, os.path.abspath(".."))

from lib.transforms import to_silver  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_spike")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

source = f"{catalog}.{schema}.bronze_orders"
target = f"{catalog}.{schema}.silver_orders"

# COMMAND ----------

silver = to_silver(spark.table(source))

(
    silver.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .clusterBy("order_date", "country")
    .saveAsTable(target)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constraints, not just conventions
# MAGIC
# MAGIC Delta constraints make the contract enforceable rather than aspirational —
# MAGIC a future write that violates them fails loudly instead of quietly poisoning
# MAGIC the gold layer.

# COMMAND ----------

spark.sql(f"ALTER TABLE {target} ALTER COLUMN order_id SET NOT NULL")
spark.sql(f"""
    ALTER TABLE {target}
    ADD CONSTRAINT revenue_matches_line
    CHECK (revenue IS NULL OR abs(revenue - quantity * unit_price) < 0.011)
""")

# COMMAND ----------

from pyspark.sql import functions as F

display(
    spark.table(target)
    .groupBy("_is_valid")
    .agg(F.count("*").alias("rows"))
    .withColumn("pct", F.round(F.col("rows") * 100.0 / spark.table(target).count(), 2))
)

# COMMAND ----------

display(
    spark.table(target)
    .select(F.explode("_quality").alias("reason"))
    .groupBy("reason")
    .count()
    .orderBy(F.desc("count"))
)
