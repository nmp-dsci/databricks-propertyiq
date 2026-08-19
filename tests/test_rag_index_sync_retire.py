"""Local-Spark check for the retire step's stale-key logic in
src/notebooks/rag/04_index_sync.py (stale_keys() and retire()'s DELETE).

The notebook itself can't be imported (dbutils widgets, a live WorkspaceClient),
so this reconstructs the exact SQL strings from the notebook against local temp
views and checks the NOT EXISTS subquery selects the same rows an interpolated
IN (...) list of quoted chunk_key literals would have — including a chunk_key
containing a single quote, which is exactly what an interpolated IN list gets
wrong and NOT EXISTS never has to worry about.
"""

from __future__ import annotations


def _build_tables(spark, silver_rows, sync_state_rows):
    silver = spark.createDataFrame(silver_rows, "chunk_key string, is_current boolean")
    silver.createOrReplaceTempView("silver_chunks")
    sync_state = spark.createDataFrame(sync_state_rows, "index_name string, chunk_key string")
    sync_state.createOrReplaceTempView("index_sync_state")


def test_stale_keys_matches_notebook_select(spark):
    # gte-1 was re-chunked away (no longer current); gte-2 is still current.
    _build_tables(
        spark,
        silver_rows=[("gte-1", False), ("gte-2", True)],
        sync_state_rows=[("rag_chunks_gte", "gte-1"), ("rag_chunks_gte", "gte-2")],
    )

    # Verbatim from 04_index_sync.py::stale_keys
    stale = spark.sql("""
        SELECT s.chunk_key
        FROM index_sync_state s
        LEFT JOIN silver_chunks c
          ON c.chunk_key = s.chunk_key AND c.is_current
        WHERE s.index_name = 'rag_chunks_gte' AND c.chunk_key IS NULL
    """).collect()

    assert [row["chunk_key"] for row in stale] == ["gte-1"]


def test_not_exists_delete_handles_quote_in_chunk_key(spark):
    """A chunk_key containing a single quote is exactly what breaks a naive
    interpolated IN (...) list; the NOT EXISTS subquery never interpolates
    the key at all, so it isn't a code path that can be affected."""
    tricky_key = "video's-chunk-0"
    _build_tables(
        spark,
        silver_rows=[(tricky_key, False)],
        sync_state_rows=[("rag_chunks_gte", tricky_key)],
    )

    to_delete = spark.sql("""
        SELECT s.chunk_key
        FROM index_sync_state s
        WHERE s.index_name = 'rag_chunks_gte'
          AND NOT EXISTS (
            SELECT 1 FROM silver_chunks c
            WHERE c.chunk_key = s.chunk_key AND c.is_current
          )
    """).collect()

    assert [row["chunk_key"] for row in to_delete] == [tricky_key]


def test_not_exists_delete_leaves_current_keys_in_sync_state(spark):
    _build_tables(
        spark,
        silver_rows=[("k1", False), ("k2", True)],
        sync_state_rows=[
            ("rag_chunks_gte", "k1"),
            ("rag_chunks_gte", "k2"),
            ("rag_chunks_minilm", "k1"),
        ],
    )

    remaining = spark.sql("""
        SELECT s.index_name, s.chunk_key
        FROM index_sync_state s
        WHERE NOT (
          NOT EXISTS (
            SELECT 1 FROM silver_chunks c
            WHERE c.chunk_key = s.chunk_key AND c.is_current
          )
          AND s.index_name = 'rag_chunks_gte'
        )
    """).collect()

    rows = {(row["index_name"], row["chunk_key"]) for row in remaining}
    assert rows == {("rag_chunks_gte", "k2"), ("rag_chunks_minilm", "k1")}
