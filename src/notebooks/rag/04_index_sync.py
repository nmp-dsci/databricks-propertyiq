# Databricks notebook source
# MAGIC %md
# MAGIC # rag 04 · Embed and push the chunks into Vector Search
# MAGIC
# MAGIC Two indexes over the same `silver_chunks`, so the eval can tell an
# MAGIC *engine* difference from an *embedding model* difference:
# MAGIC
# MAGIC | index | vectors | dim | queried by |
# MAGIC |---|---|---|---|
# MAGIC | `rag_chunks_gte` | `databricks-gte-large-en`, computed here with `ai_query` | 1024 | the deployed agent |
# MAGIC | `rag_chunks_minilm` | the local MiniLM vectors carried through the export | 384 | the local eval harness (parity control) |
# MAGIC
# MAGIC **Why DIRECT_ACCESS rather than DELTA_SYNC.** A delta-sync index would do
# MAGIC the embedding and syncing for us, and it was the plan's first choice — but
# MAGIC on Free Edition its backing pipeline never leaves "pending setup of
# MAGIC pipeline resources" (probed for an hour; the endpoint itself is ONLINE and
# MAGIC direct-access indexes on it work fine). So this notebook does by hand what
# MAGIC delta-sync would have done: embed what changed, upsert it, and retire keys
# MAGIC silver no longer holds as current.
# MAGIC
# MAGIC Embedding is **incremental on `text_sha`**. Re-embedding 2,966 chunks on
# MAGIC every ingest would be slow and pointless when a typical export changes a
# MAGIC handful of chunks, so `chunk_vectors_gte` is a cache keyed by the content
# MAGIC hash — exactly the "only what changed" behaviour CDF gives delta-sync.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "rag")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")
dbutils.widgets.text("index_endpoint", "rag-search")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
embedding_endpoint = dbutils.widgets.get("embedding_endpoint")
index_endpoint = dbutils.widgets.get("index_endpoint")
prefix = f"{catalog}.{schema}"

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# The columns the index carries alongside the vector. Everything the agent needs
# to cite an answer (title, url, timestamp) has to live here — a retrieval hit
# that needs a second lookup to become a citation is a slow answer.
PAYLOAD = [
    "chunk_key",
    "video_id",
    "chunk_index",
    "title",
    "channel_name",
    "source_url",
    "start_seconds",
    "text",
    "embedding_text",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 · Embed new or changed chunks with `ai_query`
# MAGIC
# MAGIC One SQL statement embeds everything missing from the cache — no Python
# MAGIC loop, no client-side batching, and it parallelises across the cluster.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {prefix}.chunk_vectors_gte (
      chunk_key STRING NOT NULL,
      text_sha STRING NOT NULL,
      embedding ARRAY<FLOAT>,
      embedded_at TIMESTAMP
    )
    COMMENT 'gte-large-en embeddings for silver_chunks, keyed by content hash so
             an unchanged chunk is never re-embedded. Feeds rag_chunks_gte.'
""")

pending = spark.sql(f"""
    SELECT c.chunk_key, c.text_sha, c.embedding_text
    FROM {prefix}.silver_chunks c
    LEFT JOIN {prefix}.chunk_vectors_gte v
      ON c.chunk_key = v.chunk_key AND c.text_sha = v.text_sha
    WHERE c.is_current AND v.chunk_key IS NULL
""")
pending_count = pending.count()
print(f"chunks needing a gte embedding: {pending_count:,}")

# COMMAND ----------

if pending_count:
    pending.createOrReplaceTempView("pending_chunks")
    spark.sql(f"""
        INSERT INTO {prefix}.chunk_vectors_gte
        SELECT
          chunk_key,
          text_sha,
          ai_query('{embedding_endpoint}', embedding_text) AS embedding,
          current_timestamp() AS embedded_at
        FROM pending_chunks
    """)
    print(f"embedded {pending_count:,} chunk(s)")
else:
    print("embedding cache is current")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 · Upsert into the indexes
# MAGIC
# MAGIC Only rows whose content changed since the last sync are pushed. The
# MAGIC watermark table is what makes a no-op ingest cost nothing.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {prefix}.index_sync_state (
      index_name STRING NOT NULL,
      chunk_key STRING NOT NULL,
      text_sha STRING NOT NULL,
      synced_at TIMESTAMP
    )
    COMMENT 'What each Vector Search index currently holds, so a sync only
             pushes rows whose text_sha moved.'
""")


_PAYLOAD_SQL = ", ".join(f"c.{column}" for column in PAYLOAD)

# Rows this index has not seen at their current content hash. Anti-joining the
# sync state (rather than diffing counts) means a re-run after a partial failure
# picks up exactly the stragglers.
_UNSYNCED = f"""
    LEFT JOIN {prefix}.index_sync_state s
      ON s.index_name = '{{index_name}}' AND s.chunk_key = c.chunk_key
         AND s.text_sha = c.text_sha
    WHERE c.is_current AND s.chunk_key IS NULL
"""


def gte_rows_to_sync():
    """Chunks whose gte vector is embedded but not yet in the index."""
    return spark.sql(f"""
        SELECT {_PAYLOAD_SQL}, g.embedding AS embedding, c.text_sha
        FROM {prefix}.silver_chunks c
        JOIN {prefix}.chunk_vectors_gte g
          ON c.chunk_key = g.chunk_key AND c.text_sha = g.text_sha
        {_UNSYNCED.format(index_name="rag_chunks_gte")}
    """)


def minilm_rows_to_sync():
    """Same, using the MiniLM vector the export already carries."""
    return spark.sql(f"""
        SELECT {_PAYLOAD_SQL}, c.embedding AS embedding, c.text_sha
        FROM {prefix}.silver_chunks c
        {_UNSYNCED.format(index_name="rag_chunks_minilm")}
    """)


def push(index_name: str, frame, batch_size: int = 200) -> int:
    """Upsert in batches — one call per 2,966 rows would exceed the request size."""
    full_name = f"{prefix}.{index_name}"
    records = frame.collect()
    if not records:
        print(f"{index_name}: already current")
        return 0

    pushed = 0
    batch: list[dict] = []
    synced: list[tuple] = []
    for record in records:
        row = {column: record[column] for column in PAYLOAD}
        row["embedding"] = [float(value) for value in record["embedding"]]
        batch.append(row)
        synced.append((index_name, record["chunk_key"], record["text_sha"]))
        if len(batch) >= batch_size:
            pushed += _upsert(full_name, batch)
            batch = []
    if batch:
        pushed += _upsert(full_name, batch)

    state = spark.createDataFrame(synced, "index_name string, chunk_key string, text_sha string")
    from pyspark.sql import functions as F

    state.withColumn("synced_at", F.current_timestamp()).write.mode("append").saveAsTable(
        f"{prefix}.index_sync_state"
    )
    print(f"{index_name}: upserted {pushed:,} row(s)")
    return pushed


def _upsert(full_name: str, batch: list[dict]) -> int:
    response = w.api_client.do(
        "POST",
        f"/api/2.0/vector-search/indexes/{full_name}/upsert-data",
        body={"inputs_json": json.dumps(batch)},
    )
    result = response.get("result") or {}
    failed = result.get("failed_primary_keys") or []
    if failed:
        raise RuntimeError(f"{full_name}: {len(failed)} row(s) rejected, e.g. {failed[:3]}")
    return int(result.get("success_row_count") or 0)


def stale_keys(index_name: str):
    """Keys this index has synced that silver no longer holds as current.

    A soft-deleted chunk (transcript-lab's replace_chunks() re-chunking a video)
    just stops showing up in _UNSYNCED — it was already upserted in a prior run
    and would otherwise sit in the index forever, retrievable and citable even
    though the corpus no longer has that text.
    """
    return spark.sql(f"""
        SELECT s.chunk_key
        FROM {prefix}.index_sync_state s
        LEFT JOIN {prefix}.silver_chunks c
          ON c.chunk_key = s.chunk_key AND c.is_current
        WHERE s.index_name = '{index_name}' AND c.chunk_key IS NULL
    """)


def _delete(full_name: str, keys: list[str]) -> int:
    response = w.api_client.do(
        "POST",
        f"/api/2.0/vector-search/indexes/{full_name}/delete-data",
        body={"primary_keys": keys},
    )
    result = response.get("result") or {}
    failed = result.get("failed_primary_keys") or []
    if failed:
        raise RuntimeError(f"{full_name}: {len(failed)} row(s) rejected, e.g. {failed[:3]}")
    return len(keys)


def retire(index_name: str, batch_size: int = 200) -> int:
    """Delete keys silver no longer holds as current, and drop them from sync state."""
    full_name = f"{prefix}.{index_name}"
    keys = [row["chunk_key"] for row in stale_keys(index_name).collect()]
    if not keys:
        print(f"{index_name}: nothing to retire")
        return 0

    retired = 0
    for start in range(0, len(keys), batch_size):
        batch = keys[start : start + batch_size]
        retired += _delete(full_name, batch)

    keys_literal = ", ".join(f"'{key}'" for key in keys)
    spark.sql(f"""
        DELETE FROM {prefix}.index_sync_state
        WHERE index_name = '{index_name}' AND chunk_key IN ({keys_literal})
    """)
    print(f"{index_name}: retired {retired:,} row(s)")
    return retired


# COMMAND ----------

gte_pushed = push("rag_chunks_gte", gte_rows_to_sync())

# COMMAND ----------

# The parity control keeps the local MiniLM vectors exactly as transcript-lab
# produced them, so a retrieval difference against Chroma is the engine's, not
# the model's.
minilm_pushed = push("rag_chunks_minilm", minilm_rows_to_sync())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 · Retire chunks silver no longer holds as current
# MAGIC
# MAGIC A re-chunk that produces fewer chunks for a video leaves stale keys behind
# MAGIC in the index — left there, the agent can retrieve and cite text the corpus
# MAGIC no longer contains, which is worse than a miss because it looks authoritative.

# COMMAND ----------

gte_retired = retire("rag_chunks_gte")

# COMMAND ----------

minilm_retired = retire("rag_chunks_minilm")

# COMMAND ----------

for index_name in ("rag_chunks_gte", "rag_chunks_minilm"):
    status = w.api_client.do("GET", f"/api/2.0/vector-search/indexes/{prefix}.{index_name}")
    print(index_name, "->", status.get("status", {}).get("indexed_row_count"), "rows indexed")
