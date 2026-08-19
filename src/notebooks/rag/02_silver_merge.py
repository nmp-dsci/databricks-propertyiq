# Databricks notebook source
# MAGIC %md
# MAGIC # rag 02 · Silver — reconcile each entity against its newest snapshot
# MAGIC
# MAGIC Every landed file is a **full snapshot** of one entity, so silver is a
# MAGIC reconciliation rather than an append: take the newest snapshot in bronze,
# MAGIC `MERGE` it onto silver by natural key, and mark anything silver still holds
# MAGIC that the snapshot no longer contains as `is_current = false`.
# MAGIC
# MAGIC Soft-delete, not hard-delete, because the local pipeline recreates chunk
# MAGIC ids on every re-index (`replace_chunks()` deletes and rewrites a video's
# MAGIC chunks), so a "missing" row is usually a re-chunk, not a retraction —
# MAGIC and the history is worth keeping when a retrieval regression needs
# MAGIC explaining.
# MAGIC
# MAGIC The merge key is `(video_id, chunk_index)` plus a `text_sha`, never the
# MAGIC chunk id, for the same reason: ids churn, content does not.
# MAGIC
# MAGIC `silver_chunks` is created with Change Data Feed on — it is the
# MAGIC prerequisite for a Vector Search delta-sync index, and it lets the index
# MAGIC follow only what changed instead of re-embedding the whole corpus.

# COMMAND ----------

import os
import sys

# One level deeper than the propertyiq notebooks (src/notebooks/rag/), so climb
# two to reach src/ and make `lib` importable.
sys.path.insert(0, os.path.abspath("../.."))

from lib.rag_export import ENTITY_KEYS  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "rag")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql import Window

# Change Data Feed is only required where an index syncs from, but turning it on
# for every silver table costs nothing here (these are small) and keeps the
# option open.
CDF = "delta.enableChangeDataFeed = true"


def newest_snapshot(entity: str):
    """The most recently landed full snapshot for one entity.

    Bronze holds every snapshot ever landed. The winner is the newest
    `_source_file`; ties break on the ingest timestamp so a re-ingest of the
    same file never splits a snapshot in half.
    """
    bronze = spark.table(f"{catalog}.{schema}.bronze_{entity}")
    ranked = bronze.withColumn(
        "_rank",
        F.dense_rank().over(
            Window.orderBy(F.col("_ingested_at").desc(), F.col("_source_file").desc())
        ),
    )
    return ranked.filter(F.col("_rank") == 1).drop("_rank")


def reconcile(entity: str) -> dict:
    """MERGE the newest snapshot into silver_<entity>, soft-deleting the rest."""
    target_name = f"{catalog}.{schema}.silver_{entity}"
    keys = ENTITY_KEYS.get(entity)
    if not keys:
        raise ValueError(f"no natural key declared for {entity}")

    snapshot = (
        newest_snapshot(entity)
        .drop("_rescued_data")
        .withColumn("is_current", F.lit(True))
        .withColumn("_synced_at", F.current_timestamp())
    )
    # A snapshot that repeats a key is a bug in the exporter, not something to
    # merge — MERGE would fail anyway, but failing here names the entity.
    total = snapshot.count()
    distinct = snapshot.select(*keys).distinct().count()
    if total != distinct:
        raise ValueError(f"{entity}: snapshot has {total - distinct} duplicate key(s) on {keys}")

    if not spark.catalog.tableExists(target_name):
        snapshot.write.format("delta").saveAsTable(target_name)
        spark.sql(f"ALTER TABLE {target_name} SET TBLPROPERTIES ({CDF})")
        return {"entity": entity, "rows": total, "action": "created"}

    target = DeltaTable.forName(spark, target_name)
    condition = " AND ".join(f"t.{key} <=> s.{key}" for key in keys)
    (
        target.alias("t")
        .merge(snapshot.alias("s"), condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        # Present in silver, absent from the newest snapshot: retired upstream.
        .whenNotMatchedBySourceUpdate(set={"is_current": F.lit(False)})
        .execute()
    )
    current = spark.table(target_name).filter("is_current").count()
    retired = spark.table(target_name).filter("NOT is_current").count()
    return {"entity": entity, "rows": current, "action": f"merged ({retired} retired)"}


# COMMAND ----------

results = []
for entity in ENTITY_KEYS:
    if not spark.catalog.tableExists(f"{catalog}.{schema}.bronze_{entity}"):
        print(f"bronze_{entity}: not landed yet, skipping")
        continue
    results.append(reconcile(entity))
    print(results[-1])

# COMMAND ----------

# CDF has to be on for the delta-sync index to follow changes incrementally.
# Setting it after creation covers tables created by an earlier run.
for entity in ENTITY_KEYS:
    name = f"{catalog}.{schema}.silver_{entity}"
    if spark.catalog.tableExists(name):
        spark.sql(f"ALTER TABLE {name} SET TBLPROPERTIES ({CDF})")

# COMMAND ----------

# Vector Search addresses rows by a NOT NULL primary key. `chunk_key` comes from
# the exporter (see lib.rag_export.chunk_rows) rather than being minted here, so
# the same key exists in the Parquet, in bronze, and in silver.
spark.sql(f"""
    ALTER TABLE {catalog}.{schema}.silver_chunks
    ALTER COLUMN chunk_key SET NOT NULL
""")

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.silver_chunks IS
    'transcript-lab chunks, newest snapshot reconciled. One row per
     (video_id, chunk_index); is_current=false marks chunks the local pipeline
     no longer produces. embedding is the local 384-dim MiniLM vector;
     embedding_text is what both the local and the Databricks index embed.
     Source of truth for the Vector Search indexes.'
""")

# COMMAND ----------

display(spark.createDataFrame(results)) if results else print("nothing to reconcile")
