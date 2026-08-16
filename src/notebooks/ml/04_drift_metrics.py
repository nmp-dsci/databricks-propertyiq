# Databricks notebook source
# MAGIC %md
# MAGIC # ML 04 · Drift + accuracy metrics
# MAGIC
# MAGIC The hand-rolled monitoring loop (Lakehouse Monitoring's Free Edition
# MAGIC availability is undocumented, so this uses parts that work everywhere):
# MAGIC
# MAGIC - **PSI per feature** — latest month vs the training reference window.
# MAGIC   Catches input drift *before* actuals arrive.
# MAGIC - **MAE per month** — predictions vs the published medians. Catches output
# MAGIC   rot as soon as actuals land.
# MAGIC
# MAGIC Both land in tables a SQL alert can watch; the job's condition task reads
# MAGIC the max PSI task value and triggers retraining when the input has shifted.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from pyspark.sql import functions as F  # noqa: E402

from lib.ml_features import NUMERIC_FEATURES  # noqa: E402
from lib.ml_monitoring import drift_columns, feature_psi_frame  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("ml_schema", "propertyiq_ml")
dbutils.widgets.text("reference_months", "24")

catalog = dbutils.widgets.get("catalog")
ml_schema = dbutils.widgets.get("ml_schema")
reference_months = int(dbutils.widgets.get("reference_months"))

features = spark.table(f"{catalog}.{ml_schema}.features_rent")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Input drift: PSI, latest month vs a trailing reference window
# MAGIC
# MAGIC Two things are deliberately *not* compared here, both learned from live
# MAGIC runs rather than theory:
# MAGIC
# MAGIC - **month_of_year** is excluded — a cyclical indicator always "drifts"
# MAGIC   against a full-year reference (first run: PSI 8.64 on a healthy month).
# MAGIC - The reference is the **trailing 2 years**, not all history. Rents trend,
# MAGIC   so the latest month sits above the 10-year distribution *permanently* —
# MAGIC   an all-history reference turns the retrain trigger into a treadmill. The
# MAGIC   trailing window asks the actionable question: has the input moved
# MAGIC   *recently*, in a way the champion hasn't seen?

# COMMAND ----------

latest_month = features.agg(F.max("month")).first()[0]
ref_start = features.select(F.add_months(F.max("month"), -reference_months)).first()[0]

psi_features = drift_columns(NUMERIC_FEATURES)

reference = (
    features.filter((F.col("month") >= ref_start) & (F.col("month") < F.lit(latest_month)))
    .select(*psi_features)
    .toPandas()
)
current = features.filter(F.col("month") == F.lit(latest_month)).select(*psi_features).toPandas()

psi_pdf = feature_psi_frame(reference, current, psi_features)
psi_pdf["month"] = latest_month

drift = (
    spark.createDataFrame(psi_pdf)
    .withColumn("computed_at", F.current_timestamp())
    .select("month", "feature", "psi", "status", "computed_at")
)

drift_target = f"{catalog}.{ml_schema}.drift_metrics"
if spark.catalog.tableExists(drift_target):
    (
        drift.write.mode("overwrite")
        .option("replaceWhere", f"month = '{latest_month}'")
        .saveAsTable(drift_target)
    )
else:
    drift.write.saveAsTable(drift_target)
    spark.sql(f"""
        COMMENT ON TABLE {drift_target} IS
        'Feature drift per month: PSI of each model feature in that month vs the
         training reference window. Read status as stable (<0.1), drifting
         (0.1-0.25), shifted (>0.25). A SQL alert on this table is the monitoring
         hook; the scoring job retrains automatically on shifted input.'
    """)

display(drift)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output accuracy: MAE per month, champion vs naive
# MAGIC
# MAGIC `predictions_rent` carries the published actual next to every prediction,
# MAGIC so accuracy is one aggregation — no join to maintain.

# COMMAND ----------

predictions = spark.table(f"{catalog}.{ml_schema}.predictions_rent")

accuracy = (
    predictions.groupBy("month", "model_version")
    .agg(
        F.count("*").alias("n_scored"),
        F.round(F.avg(F.abs(F.col("predicted_weekly_rent") - F.col("median_weekly_rent"))), 2).alias(
            "mae"
        ),
    )
    .withColumn("computed_at", F.current_timestamp())
)

acc_target = f"{catalog}.{ml_schema}.prediction_accuracy"
accuracy.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(acc_target)
spark.sql(f"""
    COMMENT ON TABLE {acc_target} IS
    'Rolling model accuracy: MAE of predicted vs published median weekly rent per
     month per model version. Rising MAE with stable PSI means the world changed
     in a way the features do not capture; rising PSI predicts this table before
     it happens.'
""")

display(spark.table(acc_target).orderBy(F.desc("month")))

# COMMAND ----------

# The condition task in ml_score reads this: max PSI over the latest month.
# > 0.25 ("shifted") routes the run into the retrain task.
max_psi = float(psi_pdf["psi"].max())
dbutils.jobs.taskValues.set("max_psi", max_psi)
print(f"max PSI this month: {max_psi:.4f}")
