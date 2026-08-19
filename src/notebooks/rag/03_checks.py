# Databricks notebook source
# MAGIC %md
# MAGIC # rag 03 · Checks — the corpus has to be intact before it is indexed
# MAGIC
# MAGIC Runs last, so a bad export stops here rather than quietly re-embedding a
# MAGIC half-loaded corpus. Four kinds of check:
# MAGIC
# MAGIC * **grain** — one row per natural key, or the MERGE key is wrong;
# MAGIC * **referential** — every chunk belongs to a transcript we actually landed;
# MAGIC * **embedding shape** — 384 dims on every current chunk, since a ragged
# MAGIC   vector column breaks a self-managed index at sync time rather than here;
# MAGIC * **coverage floors** — "did this export silently produce almost nothing",
# MAGIC   the failure mode a full-snapshot feed is most exposed to.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "rag")
# Floors are deliberately well below the corpus as it stands (2,966 chunks /
# 107 transcripts) so ordinary growth never trips them, but a truncated export
# does.
dbutils.widgets.text("min_chunks", "1000")
dbutils.widgets.text("min_transcripts", "50")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
min_chunks = int(dbutils.widgets.get("min_chunks"))
min_transcripts = int(dbutils.widgets.get("min_transcripts"))
prefix = f"{catalog}.{schema}"

failures: list[str] = []


def check(name: str, violation_sql: str) -> None:
    n = spark.sql(violation_sql).count()
    print(f"{name}: {'ok' if n == 0 else f'FAIL ({n:,} violations)'}")
    if n:
        failures.append(f"{name}: {n:,} violations")


def at_least(name: str, table: str, floor: int, where: str = "is_current") -> None:
    n = spark.table(f"{prefix}.{table}").filter(where).count()
    print(f"{name}: {n:,} rows (floor {floor:,}) {'ok' if n >= floor else 'FAIL'}")
    if n < floor:
        failures.append(f"{name}: {n:,} < {floor:,}")


# COMMAND ----------

check(
    "chunk grain unique",
    f"""SELECT video_id, chunk_index FROM {prefix}.silver_chunks
        WHERE is_current GROUP BY ALL HAVING count(*) > 1""",
)
check(
    "chunk_key unique",
    f"""SELECT chunk_key FROM {prefix}.silver_chunks
        WHERE is_current GROUP BY ALL HAVING count(*) > 1""",
)
check(
    "transcript grain unique",
    f"""SELECT video_id FROM {prefix}.silver_transcripts
        WHERE is_current GROUP BY ALL HAVING count(*) > 1""",
)
check(
    "segment grain unique",
    f"""SELECT video_id, segment_index FROM {prefix}.silver_segments
        WHERE is_current GROUP BY ALL HAVING count(*) > 1""",
)

# COMMAND ----------

# Every current chunk must trace back to a transcript we hold. A chunk without
# one means the export raced a local re-index and landed half a corpus.
check(
    "chunks reference a landed transcript",
    f"""SELECT c.video_id FROM {prefix}.silver_chunks c
        LEFT JOIN {prefix}.silver_transcripts t
          ON c.video_id = t.video_id AND t.is_current
        WHERE c.is_current AND t.video_id IS NULL""",
)

# A ragged embedding column fails at index-sync time with a far worse error
# message than this one.
check(
    "embeddings are 384-dim",
    f"""SELECT chunk_key FROM {prefix}.silver_chunks
        WHERE is_current AND (embedding IS NULL OR size(embedding) != 384)""",
)

check(
    "embedding_text is populated",
    f"""SELECT chunk_key FROM {prefix}.silver_chunks
        WHERE is_current AND (embedding_text IS NULL OR length(trim(embedding_text)) = 0)""",
)

# COMMAND ----------

at_least("chunk coverage", "silver_chunks", min_chunks)
at_least("transcript coverage", "silver_transcripts", min_transcripts)

# COMMAND ----------

if failures:
    raise AssertionError("data quality checks failed:\n  - " + "\n  - ".join(failures))
print("all checks passed")
