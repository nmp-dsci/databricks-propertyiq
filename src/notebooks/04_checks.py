# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Checks — fail the run loudly, not the dashboard quietly
# MAGIC
# MAGIC A port of the dbt project's six singular tests, run as the job's last task
# MAGIC so a bad build stops here instead of serving wrong numbers. Three kinds:
# MAGIC grain uniqueness (the marts must have exactly one row per key), coverage
# MAGIC floors ("did the build silently produce almost nothing"), and a
# MAGIC plausibility band on yield — Australian residential gross yields sit
# MAGIC roughly 1–12%, so anything outside 0.3–25% signals a units bug or a
# MAGIC thin-cell artefact, not a market.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "propertyiq")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
prefix = f"{catalog}.{schema}"

failures: list[str] = []


def check(name: str, violation_sql: str) -> None:
    n = spark.sql(violation_sql).count()
    status = "ok" if n == 0 else f"FAIL ({n:,} violations)"
    print(f"{name}: {status}")
    if n:
        failures.append(f"{name}: {n:,} violations")


# COMMAND ----------

# Grain uniqueness — group-by-having, exactly as the dbt singular tests do.
check(
    "sales mart grain unique",
    f"""SELECT postcode, suburb, property_type, area_band, zoning, month
        FROM {prefix}.gold_property_sales
        GROUP BY ALL HAVING count(*) > 1""",
)
check(
    "rent mart grain unique",
    f"""SELECT postcode, property_type, bedroom_band, month
        FROM {prefix}.gold_property_rent
        GROUP BY ALL HAVING count(*) > 1""",
)
check(
    "yield mart grain unique",
    f"""SELECT postcode, property_type, month
        FROM {prefix}.gold_property_yield
        GROUP BY ALL HAVING count(*) > 1""",
)

# COMMAND ----------

# Coverage floors — a build that produces almost nothing is a broken build,
# even though every individual row in it looks fine.
check(
    "sales covers >= 10 postcodes",
    f"""SELECT 1 FROM {prefix}.gold_property_sales
        HAVING count(DISTINCT postcode) < 10""",
)
check(
    "rent covers >= 10 postcodes",
    f"""SELECT 1 FROM {prefix}.gold_property_rent
        HAVING count(DISTINCT postcode) < 10""",
)
check(
    "yield covers >= 5 postcodes",
    f"""SELECT 1 FROM {prefix}.gold_property_yield
        HAVING count(DISTINCT postcode) < 5""",
)

# COMMAND ----------

# Plausibility — catches weekly rent read as annual, nominal transfers that
# slipped the price floor, or a join gone wrong. One range check, many bugs.
check(
    "yield within 0.3% and 25%",
    f"""SELECT postcode, property_type, month, gross_yield_pct
        FROM {prefix}.gold_property_yield
        WHERE gross_yield_pct < 0.3 OR gross_yield_pct > 25""",
)

# COMMAND ----------

if failures:
    raise RuntimeError("post-load checks failed: " + "; ".join(failures))
print("all checks passed")
