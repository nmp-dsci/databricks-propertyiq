# Databricks notebook source
# MAGIC %md
# MAGIC # ML 01 · Feature table
# MAGIC
# MAGIC Materialises `features_rent` in the ML schema from the gold marts. The
# MAGIC feature logic lives in `src/lib/ml_features.py` (unit-tested locally) and is
# MAGIC the **same function** the scoring notebook calls — training/serving skew is
# MAGIC prevented by construction, not by discipline.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from lib.ml_features import build_rent_features  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "propertyiq")
dbutils.widgets.text("ml_schema", "propertyiq_ml")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
ml_schema = dbutils.widgets.get("ml_schema")

# Own schema, same pattern as the declarative pipeline's propertyiq_dp: ML
# tables can never collide with the marts the medallion job owns.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{ml_schema}")

# COMMAND ----------

features = build_rent_features(
    spark.table(f"{catalog}.{schema}.gold_property_rent"),
    spark.table(f"{catalog}.{schema}.gold_property_sales"),
)

target = f"{catalog}.{ml_schema}.features_rent"
features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target)
print(f"features_rent: {spark.table(target).count():,} rows")

# COMMAND ----------

# An informational primary key (UC does not enforce it) declares the feature
# grain — postcode × property_type × bedroom_band × month — which is what makes
# this a *feature table* rather than just a table with features in it, and what
# point-in-time consumers key on.
for col in ("postcode", "property_type", "bedroom_band", "month"):
    spark.sql(f"ALTER TABLE {target} ALTER COLUMN {col} SET NOT NULL")
spark.sql(f"ALTER TABLE {target} DROP CONSTRAINT IF EXISTS features_rent_pk")
spark.sql(
    f"ALTER TABLE {target} ADD CONSTRAINT features_rent_pk "
    "PRIMARY KEY (postcode, property_type, bedroom_band, month)"
)

spark.sql(f"""
    COMMENT ON TABLE {target} IS
    'Model features for the rent estimator, one row per postcode / property_type
     / bedroom_band / month. Every feature is built from strictly earlier months
     (calendar-month lag joins), so a row never contains its own answer. Built by
     the same tested function at training and scoring time.'
""")

# COMMAND ----------

display(spark.table(target).orderBy("postcode", "month").limit(20))
