"""Bronze — the same Auto Loader ingest, declared instead of orchestrated.

Compare with `src/notebooks/01_bronze_ingest.py`. The reader options are almost
identical; everything *around* them is gone:

  - No `writeStream` / `checkpointLocation` / `toTable` / `awaitTermination`.
    You return a streaming DataFrame and the framework owns the write.
  - No `cloudFiles.schemaLocation`. The pipeline manages schema and checkpoint
    state itself, which is why the whole `FEED_VERSION` problem — a checkpoint
    outliving the feed it was built against and silently skipping every new
    file — does not arise here in the same form.
  - No `trigger(availableNow=True)`. Triggered-vs-continuous is a property of
    the pipeline (`continuous: false`), not of this code.

What does NOT change: Auto Loader still tracks files exactly-once by path, so
landing still has to be append-only. That constraint belongs to the source, not
to the framework wrapped around it.
"""

# Databricks resolves sibling files in the same pipeline, so this import works
# once the bundle is deployed.
from importlib import import_module

from pyspark import pipelines as dp
from pyspark.sql import functions as F

_boot = import_module("00_bootstrap")
LANDING_ROOT, SALES_SCHEMA, RENT_SCHEMA = (
    _boot.LANDING_ROOT,
    _boot.SALES_SCHEMA,
    _boot.RENT_SCHEMA,
)


def _landed(subdir: str, ddl: str):
    """One Auto Loader read. Returns a streaming DataFrame, writes nothing."""
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821 — `spark` is injected
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .schema(ddl)
        .load(f"{LANDING_ROOT}/{subdir}")
        .select(
            "*",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_ingested_at"),
        )
    )


@dp.table(
    name="bronze_sales",
    comment=(
        "NSW Valuer General property sales as landed — every column a raw string, "
        "append-only, no repair. One row per dealing. Rebuild source for silver."
    ),
)
def bronze_sales():
    return _landed("sales", SALES_SCHEMA)


@dp.table(
    name="bronze_rent",
    comment=(
        "NSW Rental Bond Board lodgements as landed — raw strings, append-only. "
        "One row per bond. Rebuild source for silver."
    ),
)
def bronze_rent():
    return _landed("lodgements", RENT_SCHEMA)
