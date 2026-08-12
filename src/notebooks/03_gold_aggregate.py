# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — serving tables for the dashboard
# MAGIC
# MAGIC Two narrow, pre-aggregated tables sized for a 2X-Small warehouse to scan in
# MAGIC well under a second. The dashboard never touches silver: BI queries hitting
# MAGIC a 500k-row row-level table is how you end up explaining a slow dashboard to
# MAGIC a customer.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from lib.transforms import daily_revenue, quality_summary  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "retail_spike")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

silver = spark.table(f"{catalog}.{schema}.silver_orders")

# COMMAND ----------

(
    daily_revenue(silver)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.gold_daily_revenue")
)

(
    quality_summary(silver)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{catalog}.{schema}.gold_quality_summary")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Document the serving layer
# MAGIC
# MAGIC Comments are not decoration here: Genie and the Databricks Assistant read
# MAGIC table and column comments when translating a natural-language question into
# MAGIC SQL. A documented gold layer is what makes "ask your data a question"
# MAGIC actually work — worth saying out loud in a demo.

# COMMAND ----------

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.gold_daily_revenue IS
    'Daily retail revenue by country, channel and product category. One row per
     date/country/channel/category. Excludes rows failing data quality checks —
     see gold_quality_summary for what was excluded and why.'
""")

for col, comment in [
    ("order_date", "Calendar date of the order, in workspace timezone."),
    ("country", "ISO-3166 alpha-2 country code. One of AU, NZ, US, GB, SG."),
    ("channel", "Sales channel: web, store, app or partner."),
    ("category", "Product category: electronics, apparel, grocery, home or sport."),
    ("orders", "Distinct count of orders."),
    ("units", "Total units sold."),
    ("revenue", "Gross revenue in AUD, quantity * unit price."),
    ("customers", "Distinct count of customers who ordered."),
]:
    spark.sql(
        f"ALTER TABLE {catalog}.{schema}.gold_daily_revenue "
        f"ALTER COLUMN {col} COMMENT '{comment}'"
    )

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.gold_quality_summary IS
    'Daily count of rows by data quality verdict. reason = "ok" means the row
     passed every check; other values name the specific failed rule.'
""")

# COMMAND ----------

display(spark.sql(f"""
    SELECT country,
           round(sum(revenue)) AS revenue,
           sum(orders)         AS orders
    FROM {catalog}.{schema}.gold_daily_revenue
    GROUP BY country
    ORDER BY revenue DESC
"""))
