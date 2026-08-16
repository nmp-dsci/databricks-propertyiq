# The same medallion, as a Declarative Pipeline

This directory is a **second implementation** of what
`resources/jobs/medallion_job.yml` already does, written as a Lakeflow
Declarative Pipeline so the two can be read against each other.

It is not a migration and not a replacement. The job is the working pipeline.

## It does not run

- No `trigger`, no `schedule`, `continuous: false`.
- It publishes to its own schema (`propertyiq_dp`, the `pipeline_schema`
  bundle variable), so it cannot write over the tables the job owns. Same table
  names, different schema.
- `databricks bundle deploy` **creates** it. It only ever runs if someone types
  `databricks bundle run medallion_pipeline`.

## The point: the logic is identical

Both implementations import the same functions from `src/lib/transforms.py` —
`resolve_versions`, `clean_sales`, `clean_rent`, `gold_*`, `quality_summary` —
which are unit-tested by `make test` in ~5 seconds with no workspace. Nothing
was rewritten for the pipeline.

That is deliberate. It makes every difference below a difference of
*framework*, not of logic.

## What actually differs

| | Job (`src/notebooks/`) | Pipeline (here) |
|---|---|---|
| Write | `.writeStream…toTable()` / `.write.mode("overwrite").saveAsTable()` | The decorator is the write |
| Checkpoints | `checkpointLocation` + `cloudFiles.schemaLocation`, versioned by hand with `FEED_VERSION` | Managed by the framework; **do not set `schemaLocation`** |
| Trigger | `.trigger(availableNow=True)` in code | `continuous: false` in config |
| DAG | `depends_on:` written by hand in YAML | Inferred from which tables each function reads |
| Parameters | `dbutils.widgets` + job `parameters:` | `configuration:` + `spark.conf.get()` |
| Library import | relative `sys.path.insert(0, "..")` | explicit `lib_root` passed as configuration |
| Quality | `_quality` array + a 4th `checks` task raising `RuntimeError` | `@dp.expect*` on the datasets |
| Silver/gold | full `.mode("overwrite")` rebuild | materialized views — *the same semantics*, declared |

## Two honest findings

**1. Silver had to be a materialized view, not a streaming table.**
`resolve_versions()` ranks every landed file per partition — a full scan, not an
append-only operation. Streaming tables cannot express it. An MV over a batch
read is the declarative equivalent of the job's full overwrite, which means the
"full recompute" decision made for the job was never a limitation; it is exactly
what an MV *is*.

**2. Expectations did not replace all the tests.**
Expectations are row-level. The yield plausibility band moved cleanly onto
`gold_property_yield`. But grain uniqueness (`GROUP BY … HAVING count(*) > 1`)
and the coverage floors are *aggregate* assertions with no single row to
evaluate. `04_checks.py` works around that by computing violations as a private
dataset and asserting the count is zero — less code than the job's checks
notebook, but a workaround rather than a feature. "Expectations replaced my
tests" would be overselling it.

## Not yet verified

This has been validated (`databricks bundle validate`), linted and formatted,
but **never deployed or run**. The parts most likely to need a fix on first
update:

- the `lib_root` import bootstrap in `00_bootstrap.py`, and the sibling-file
  `import_module("00_bootstrap")` in `01_bronze.py`
- the exact expectation expressions in `02_silver.py`, which assume column names
  (`sale_month`, `weekly_rent`) that `make test` covers but that were not
  re-checked against a live silver table here
