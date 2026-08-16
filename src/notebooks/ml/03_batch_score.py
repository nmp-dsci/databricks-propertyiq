# Databricks notebook source
# MAGIC %md
# MAGIC # ML 03 · Batch score
# MAGIC
# MAGIC Scores the latest feature month with whatever `@champion` currently points
# MAGIC at, via `mlflow.pyfunc.spark_udf`. The features are rebuilt by the **same
# MAGIC tested function** the training path used, so there is no second feature
# MAGIC implementation to drift.
# MAGIC
# MAGIC Every scored row is stamped with the model version and timestamp — any
# MAGIC number in `predictions_rent` is traceable to the exact model, code and data
# MAGIC that produced it.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from lib.ml_features import FEATURE_COLUMNS  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("ml_schema", "propertyiq_ml")

catalog = dbutils.widgets.get("catalog")
ml_schema = dbutils.widgets.get("ml_schema")

MODEL_NAME = f"{catalog}.{ml_schema}.rent_estimator"
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

features = spark.table(f"{catalog}.{ml_schema}.features_rent")
score_month = features.agg(F.max("month")).first()[0]
to_score = features.filter(F.col("month") == F.lit(score_month))
print(f"scoring {to_score.count():,} rows for {score_month}")

# COMMAND ----------

# Driver-side pandas scoring, deliberately. One month is a few thousand rows —
# the right tool is a single in-process predict, not a distributed UDF. The
# scale-out pattern (`mlflow.pyfunc.spark_udf`) is what you would reach for at
# millions of rows, but it is also currently broken on serverless: the
# environment's mlflow fails parsing the serverless runtime version string
# ("Invalid version: '18.x-aarch64-photon-scala2'") before it ever loads the
# model. A real Free Edition finding — documented here rather than papered over.
champion_version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

scored_pdf = to_score.toPandas()
scored_pdf["predicted_weekly_rent"] = champion.predict(scored_pdf[FEATURE_COLUMNS]).round(2)

predictions = (
    spark.createDataFrame(
        scored_pdf[
            [
                "postcode",
                "property_type",
                "bedroom_band",
                "month",
                "predicted_weekly_rent",
                "median_weekly_rent",  # the published actual, for error tracking
            ]
        ]
    )
    .withColumn("model_version", F.lit(int(champion_version)))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

target = f"{catalog}.{ml_schema}.predictions_rent"

# Idempotent per month: re-running the job replaces this month's predictions
# instead of duplicating them, and history for earlier months is untouched.
if spark.catalog.tableExists(target):
    (
        predictions.write.mode("overwrite")
        .option("replaceWhere", f"month = '{score_month}'")
        .saveAsTable(target)
    )
else:
    predictions.write.saveAsTable(target)
    spark.sql(f"""
        COMMENT ON TABLE {target} IS
        'Champion-model weekly rent predictions per postcode / property_type /
         bedroom_band / month, stamped with the Unity Catalog model_version and
         scored_at that produced each row. median_weekly_rent is the published
         actual for the same cell, so prediction error is a simple difference.'
    """)

print(f"wrote predictions for {score_month} with model v{champion_version}")
display(spark.table(target).orderBy(F.desc("month")).limit(10))
