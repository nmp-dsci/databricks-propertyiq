# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — serving marts for dashboards, Genie and agents
# MAGIC
# MAGIC Three pre-aggregated marts plus a quality rollup, sized for a 2X-Small
# MAGIC warehouse to scan in well under a second. Grain and metric shape are ported
# MAGIC from the dbt marts: **additive legs first** (`total_*`, `n_*`) so any
# MAGIC consumer rolling up to quarter or region recomputes ratios correctly by
# MAGIC summing legs — never by averaging averages. Medians are convenience
# MAGIC columns and are NOT additive.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from lib.transforms import (  # noqa: E402
    gold_rent_monthly,
    gold_sales_monthly,
    gold_yield_monthly,
    quality_summary,
)

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "propertyiq")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

silver_sales = spark.table(f"{catalog}.{schema}.silver_sales")
silver_rent = spark.table(f"{catalog}.{schema}.silver_rent")


def save(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{catalog}.{schema}.{name}"
    )
    print(f"{name}: {spark.table(f'{catalog}.{schema}.{name}').count():,} rows")


# COMMAND ----------

save(gold_sales_monthly(silver_sales), "gold_property_sales")
save(gold_rent_monthly(silver_rent), "gold_property_rent")
save(gold_yield_monthly(silver_sales, silver_rent), "gold_property_yield")

# COMMAND ----------

from pyspark.sql import functions as F  # noqa: E402

save(
    quality_summary(silver_sales, "sale_month")
    .withColumn("dataset", F.lit("nsw_sales"))
    .unionByName(
        quality_summary(silver_rent, "rent_month").withColumn("dataset", F.lit("nsw_rent"))
    ),
    "gold_quality_summary",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Document the serving layer
# MAGIC
# MAGIC These comments are not decoration: Genie and the Assistant read them when
# MAGIC turning a question into SQL. The wording is lifted from the dbt project's
# MAGIC `_marts.yml` — descriptions that already spent a year grounding an NL2SQL
# MAGIC agent over this exact data.

# COMMAND ----------

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.gold_property_sales IS
    'NSW residential property sales, aggregated monthly. One row per postcode /
     suburb / property_type (house|unit) / area_band / zoning / month.
     total_sale_value and n_sold are additive — sum them before dividing when
     rolling up. median_sale_price is NOT additive across buckets. Thin cells
     are kept: filter on n_sold in the query, not by excluding rows here.
     Excludes rows failing quality checks (non-arms-length transfers under
     $10k, unparseable dates, non-residential).'
""")
spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.gold_property_rent IS
    'NSW rental bond lodgements, aggregated monthly. One row per postcode /
     property_type (house|unit) / bedroom_band (unknown,0-4,5+) / month.
     total_weekly_rent and n_rented are additive legs; medians are not.
     Bedroom bands only exist here — sales carry area_band instead.'
""")
spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.gold_property_yield IS
    'Gross rental yield by postcode / property_type / month, joined on exactly
     those three keys — never join sales to rent on suburb. gross_yield_pct =
     52 * avg weekly rent / avg sale price * 100, a ratio of averages. Cells
     with fewer than 5 sales or 5 bonds in the month are excluded as too thin.
     To roll up, sum the four additive legs and recompute the ratio.'
""")
spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.gold_quality_summary IS
    'Data quality rollup: rows per month per named reason per dataset
     (nsw_sales | nsw_rent). Valid rows count under reason = ok. Reads the same
     silver tables the marts are built from, so the quality story and the
     revenue story can never disagree.'
""")

# COMMAND ----------

display(
    spark.table(f"{catalog}.{schema}.gold_property_yield")
    .orderBy(F.desc("month"), F.desc("n_sold"))
    .limit(20)
)
