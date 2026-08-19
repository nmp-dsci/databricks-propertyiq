# Databricks notebook source
# MAGIC %md
# MAGIC # rag 01 · Bronze — land the transcript-lab snapshots
# MAGIC
# MAGIC `scripts/rag_export.py` publishes one Parquet per entity into
# MAGIC `landing/<entity>/`, named `<entity>_<sha8>.parquet` where the hash is of
# MAGIC the entity's whole content. Unchanged content is never uploaded, so a
# MAGIC second file only ever appears when the local corpus actually moved.
# MAGIC
# MAGIC Bronze is append-only: every landed snapshot is kept, and silver picks the
# MAGIC newest one per entity. That is the same contract the property feed uses —
# MAGIC Auto Loader tracks files exactly-once and never re-reads a changed file,
# MAGIC so "new version" has to mean "new file".
# MAGIC
# MAGIC Unlike the property feed, the schema is **inferred** rather than declared:
# MAGIC the publisher here is our own exporter writing typed Parquet (including
# MAGIC `array<float>` embeddings and `array<string>` tag lists), so the file's own
# MAGIC types are authoritative. `rescue` mode still catches anything unexpected.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "rag")
dbutils.widgets.text("volume", "landing")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")

volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
state_root = f"{volume_root}/_pipeline"

# COMMAND ----------

from pyspark.sql import functions as F

# Every entity the exporter can land. Keeping this list explicit (rather than
# globbing the volume) means a typo in an upload path shows up as a missing
# table instead of a silently-created one.
ENTITIES = [
    "chunks",
    "chunks_contextual",
    "transcripts",
    "segments",
    "summaries",
    "golden_qa",
    "eval_runs",
    "themes",
    "graph_entities",
    "graph_relations",
    "graph_claims",
]

# Bumped when the landing contract changes: a checkpoint carries the file list
# and schema of the feed it was built against, so a new suffix forces a clean
# re-ingest rather than silently skipping the new shape.
FEED_VERSION = "v1"

# COMMAND ----------


def land(entity: str) -> int:
    """One Auto Loader pass: landing/<entity>/*.parquet -> bronze_<entity>."""
    source = f"{volume_root}/{entity}"
    target = f"{catalog}.{schema}.bronze_{entity}"
    try:
        dbutils.fs.ls(source)
    except Exception:  # noqa: BLE001 — entity never exported yet; not an error
        print(f"{target}: no landed files yet, skipping")
        return 0

    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaLocation", f"{state_root}/schemas/{entity}_{FEED_VERSION}")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .load(source)
        .select(
            "*",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_ingested_at"),
        )
    )
    (
        stream.writeStream.format("delta")
        .option("checkpointLocation", f"{state_root}/checkpoints/{entity}_{FEED_VERSION}")
        .option("mergeSchema", "true")
        .trigger(availableNow=True)
        .toTable(target)
    ).awaitTermination()

    rows = spark.table(target).count()
    files = spark.table(target).select("_source_file").distinct().count()
    print(f"{target}: {rows:,} rows across {files:,} landed snapshot(s)")
    return rows


# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

landed = {entity: land(entity) for entity in ENTITIES}

# COMMAND ----------

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.bronze_chunks IS
    'transcript-lab chunks as landed — append-only history of full snapshots.
     One row per chunk per export. Carries the local MiniLM embedding and the
     embedding_text the local index was built from. Silver keeps the newest
     snapshot; this table is the rebuild source.'
""")

# COMMAND ----------

display(spark.createDataFrame([(k, v) for k, v in landed.items()], "entity string, rows long"))
