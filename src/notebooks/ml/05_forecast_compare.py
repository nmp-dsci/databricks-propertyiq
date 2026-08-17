# Databricks notebook source
# MAGIC %md
# MAGIC # ML 05 · Forecast comparison — AI_FORECAST vs a trained model vs naive
# MAGIC
# MAGIC Same task, three tools, one honest backtest. Forecast `gross_yield_pct`
# MAGIC six months ahead per postcode × property_type:
# MAGIC
# MAGIC 1. **ai_forecast** — the platform-native SQL table function (verified live
# MAGIC    on the Free Edition warehouse). AI functions are a *warehouse* feature,
# MAGIC    so this notebook drives it over the Statement Execution API rather than
# MAGIC    pretending it runs on job compute.
# MAGIC 2. **trained** — per-series linear trend + monthly seasonality (sklearn,
# MAGIC    pre-installed; prophet/statsforecast would need runtime pip, which Free
# MAGIC    Edition egress forbids), trained in parallel with `applyInPandas`.
# MAGIC 3. **seasonal_naive** — next July = last July. The bar both must clear.
# MAGIC
# MAGIC The backtest holds out the last 6 published months: every method forecasts
# MAGIC them from the same truncated history, and `forecast_comparison` reports MAE
# MAGIC per method per horizon. The *comparison table* is the deliverable — the
# MAGIC tool-selection judgment, not the forecast.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from pyspark.sql import functions as F  # noqa: E402

from lib.ml_forecast import (  # noqa: E402
    MIN_HISTORY_MONTHS,
    seasonal_naive_forecast,
    trend_seasonal_forecast,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "propertyiq")
dbutils.widgets.text("ml_schema", "propertyiq_ml")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("backtest_months", "6")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
ml_schema = dbutils.widgets.get("ml_schema")
warehouse_id = dbutils.widgets.get("warehouse_id")
backtest_months = int(dbutils.widgets.get("backtest_months"))

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{ml_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The shared input: truncated history, one series per postcode × type

# COMMAND ----------

yield_mart = spark.table(f"{catalog}.{schema}.gold_property_yield")
latest = yield_mart.agg(F.max("month")).first()[0]
cutoff = yield_mart.select(F.add_months(F.lit(latest), -backtest_months)).first()[0]

# Only series long enough to fit 12 seasonal offsets + trend without reading
# noise as signal; the same floor applies to every method, so the comparison
# stays apples-to-apples.
eligible = (
    yield_mart.filter(F.col("month") <= cutoff)
    .groupBy("postcode", "property_type")
    .agg(F.count("*").alias("n"))
    .filter(F.col("n") >= MIN_HISTORY_MONTHS)
    .select("postcode", "property_type")
)

history = (
    yield_mart.filter(F.col("month") <= cutoff)
    .join(eligible, ["postcode", "property_type"])
    .select("postcode", "property_type", "month", F.col("gross_yield_pct").alias("y"))
)

input_table = f"{catalog}.{ml_schema}.forecast_input"
history.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(input_table)
n_series = eligible.count()
print(f"backtest: history to {cutoff}, actuals to {latest}, {n_series} series")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trained + naive, in parallel per series

# COMMAND ----------

horizon = backtest_months
out_schema = "postcode string, property_type string, month date, yhat double, method string"


def _forecast_group(pdf):
    import pandas as pd  # local import: runs inside the grouped-map worker

    key = pdf[["postcode", "property_type"]].iloc[0]
    frames = []
    for method, fn in (("trained", trend_seasonal_forecast), ("seasonal_naive", seasonal_naive_forecast)):
        f = fn(pdf[["month", "y"]], horizon)
        if not f.empty:
            f = f.assign(postcode=key["postcode"], property_type=key["property_type"], method=method)
            frames.append(f[["postcode", "property_type", "month", "yhat", "method"]])
    if not frames:
        return pd.DataFrame(columns=["postcode", "property_type", "month", "yhat", "method"])
    return pd.concat(frames)


forecasts = (
    spark.table(input_table)
    .groupBy("postcode", "property_type")
    .applyInPandas(_forecast_group, schema=out_schema)
)

forecast_table = f"{catalog}.{ml_schema}.forecast_yield"
forecasts.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(forecast_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AI_FORECAST, on the warehouse it lives on

# COMMAND ----------

from databricks.sdk import WorkspaceClient  # noqa: E402

statement = f"""
    CREATE OR REPLACE TABLE {catalog}.{ml_schema}.forecast_ai AS
    SELECT
      split_part(series, '|', 1) AS postcode,
      split_part(series, '|', 2) AS property_type,
      CAST(ds AS DATE)           AS month,
      y_forecast                 AS yhat,
      'ai_forecast'              AS method
    FROM AI_FORECAST(
      TABLE(
        SELECT concat(postcode, '|', property_type) AS series, month AS ds, y
        FROM {input_table}
      ),
      horizon    => '{latest}',
      time_col   => 'ds',
      value_col  => 'y',
      group_col  => 'series'
    )
"""

import time  # noqa: E402

w = WorkspaceClient()
result = w.statement_execution.execute_statement(
    warehouse_id=warehouse_id, statement=statement, wait_timeout="50s"
)

# 50s is the API's maximum synchronous wait; forecasting hundreds of series
# takes longer, so the statement continues async and we poll it to a terminal
# state. ~15 min ceiling, though live runs finish well inside it.
deadline = time.time() + 900
while result.status.state.value in ("PENDING", "RUNNING") and time.time() < deadline:
    time.sleep(15)
    result = w.statement_execution.get_statement(result.statement_id)

assert result.status.state.value == "SUCCEEDED", result.status
print(f"forecast_ai: {spark.table(f'{catalog}.{ml_schema}.forecast_ai').count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The verdict table

# COMMAND ----------

all_forecasts = spark.table(forecast_table).unionByName(
    spark.table(f"{catalog}.{ml_schema}.forecast_ai")
)

actuals = yield_mart.filter(F.col("month") > cutoff).select(
    "postcode", "property_type", "month", F.col("gross_yield_pct").alias("actual")
)

comparison = (
    all_forecasts.join(actuals, ["postcode", "property_type", "month"])
    .withColumn("horizon", F.months_between("month", F.lit(cutoff)).cast("int"))
    .groupBy("method", "horizon")
    .agg(
        F.count("*").alias("n"),
        F.round(F.avg(F.abs(F.col("yhat") - F.col("actual"))), 4).alias("mae"),
    )
)

comparison_table = f"{catalog}.{ml_schema}.forecast_comparison"
comparison.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(comparison_table)
spark.sql(f"""
    COMMENT ON TABLE {comparison_table} IS
    'Backtest verdict for yield forecasting: MAE (percentage points of gross
     yield) per method per horizon month, all methods forecasting the same
     held-out months from the same truncated history. Methods: ai_forecast
     (platform SQL function), trained (per-series trend+seasonality), and
     seasonal_naive — the baseline any method must beat to justify itself.'
""")

display(spark.table(comparison_table).orderBy("horizon", "method"))

# COMMAND ----------

summary = (
    spark.table(comparison_table)
    .groupBy("method")
    .agg(F.round(F.avg("mae"), 4).alias("avg_mae"))
    .orderBy("avg_mae")
)
display(summary)
best = summary.first()
print(f"best overall: {best['method']} (avg MAE {best['avg_mae']} yield points)")
