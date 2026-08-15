# databricks-propertyiq

Author Databricks notebooks, SQL and dashboards **here** — in a normal git repo,
with a normal editor and a coding agent — and have them render and run **there**,
in a Databricks workspace. Nothing is authored in the browser and copy-pasted
back.

Built against **Databricks Free Edition**, which is serverless-only, allows one
2X-Small SQL warehouse, caps concurrent job tasks at 5, and restricts outbound
internet. Every choice below falls out of those limits.

---

## What's in here

```
databricks.yml                 bundle definition — targets, variables
resources/
  jobs/medallion_job.yml       4-task serverless job (bronze → silver → gold → checks)
  dashboards/property_overview.yml
genie/propertyiq.genie.json    Genie space definition (checked in for reproducibility, not a bundle resource)
src/
  notebooks/*.py               Databricks source-format notebooks (render as notebooks up there)
  lib/transforms.py            the actual logic — pure, unit-tested, no I/O
  sql/*.sql                    warehouse queries, runnable from the terminal
dashboards/*.lvdash.json       AI/BI dashboard, deployed as code
tests/                         local Spark tests, no workspace required
scripts/                       setup, auth, SQL runner, gold/dbt parity check
```

### The one trick that makes this work

`src/notebooks/*.py` are **Databricks source-format notebooks**: plain Python
files that start with `# Databricks notebook source` and separate cells with
`# COMMAND ----------`. Markdown cells are `# MAGIC %md` comments.

Databricks renders these as real notebooks — cells, `display()` output, the
Assistant, everything. Git sees clean Python: readable diffs, no base64 image
blobs, no merge conflicts in JSON metadata. Edits made in the Databricks UI
serialise straight back to the same format.

That's the whole local↔cloud story. No conversion step.

---

## Setup

```bash
make install     # Databricks CLI, python deps, checks for a JRE
make auth        # OAuth login (falls back to a PAT if Free Edition refuses OAuth)
```

Then find your SQL warehouse id (SQL Warehouses → your warehouse → Connection
details) and either put it in `databricks.yml` under `variables.warehouse_id.default`,
or pass it per-command:

```bash
make deploy TARGET=dev DB_VARS='--var=warehouse_id=abc123'
```

---

## The loop

```bash
make test        # ~5s, local Spark, no workspace
make deploy      # push notebooks + job + dashboard to the workspace
make run         # run the job, stream logs into this terminal
make ship        # all three
```

Two ways to get code up there, and they're for different moments:

| | `make deploy` (Asset Bundles) | `make sync` (file watcher) |
|---|---|---|
| What it does | Deploys notebooks **and** job/dashboard definitions | Mirrors files only, on every save |
| Speed | ~10s | instant |
| Use when | You want the job graph and dashboard updated | You're iterating on one notebook, open in the UI |

Leave `make sync` running in a second terminal during a live session: edit here,
hit **Run** in the browser, output appears. That is the fastest possible feedback
loop and it's the one to use in the Interview 3 pair-programming round.

### SQL without leaving the terminal

```bash
make sql FILE=src/sql/01_explore.sql
```

Runs each statement on the serverless warehouse and prints the result sets here.

### Dashboards

Dashboards are miserable to hand-edit as JSON and excellent to edit in the UI, so
the workflow is a round-trip:

```bash
make deploy           # 1. push dashboards/property_overview.lvdash.json
                      # 2. rearrange / restyle in the Databricks UI
make pull-dashboard   # 3. write the UI version back into the repo
git commit            # 4. it's code again
```

`embed_credentials: false` in the resource means a deploy **errors** rather than
silently overwriting UI edits you forgot to pull.

> The dashboard SQL hardcodes `workspace.propertyiq.*`. The `.lvdash.json` is
> deployed verbatim — bundle variables are not substituted inside it. If you
> change catalog or schema, change it there too.

---

## The pipeline

A medallion pipeline over real NSW property data: Valuer General property
sales (~3.1M rows) and Rental Bond Board rental lodgements (~3.3M rows),
landed as an updates-only Parquet feed under `landing/sales/` and
`landing/lodgements/` in the `workspace.propertyiq` Unity Catalog volume by
the sibling `propertyiq_getdata` project (`propertyiq-getdata publish
databricks`), which runs outside this workspace. The rules are a deliberate
port of tested dbt models from a sibling project (`data-qa-agent`), with one
intentional divergence explained below.

The job runs on a **file arrival trigger** watching the `landing/` directory,
so publishing upstream starts a run about a minute later rather than waiting
for someone to type `make run` (which still forces one on demand). A clock
schedule would have fired on the many days with no new data, and still sat
idle for hours after a publish that did happen. The trigger debounces a
multi-file publish and is capped at one run per five minutes — see
`resources/jobs/medallion_job.yml`.

**01 · bronze** — Auto Loader (`cloudFiles`) with `Trigger.AvailableNow`:
streaming file-tracking semantics on a batch cadence, reading Parquet files
from `landing/`. Append-only, every column declared as `STRING` (the landing
contract, not inferred — Parquet carries its own types, but silver's parsers
are string-based), unexpected fields land in `_rescued_data` instead of
failing the run. Checkpoints and schema locations are suffixed with a feed
version (`FEED_VERSION` in the notebook) so a future feed-format change gets a
clean re-ingest instead of Auto Loader silently skipping every new file.
Bronze is the rebuild point when silver logic turns out wrong.

**02 · silver** — Typed, repaired and quality-flagged, ported from the dbt
`stg_sales` / `stg_rent` models that cleaned this exact data for Postgres.
Before typing, `resolve_versions` collapses bronze to one row set per landed
partition: landing is append-only, so a partition rewritten upstream (rentboard
rewrites its trailing month every run) arrives as a second file rather than
replacing the first, and the newest `_ingested_at` per partition wins. It also
drops any bronze rows that didn't come from a landing file, so history kept
from the pre-Parquet monolith CSVs can't double-count gold once the feed has
moved on. **Invalid rows are kept**, with a `_quality` array naming every rule
they failed, rather than dropped in a `WHERE` clause as the dbt originals do —
gold applies the filter instead, so the quality dashboard and the price
dashboard read from the same table. Delta `NOT NULL` constraints on the
surrogate keys make the contract enforceable rather than aspirational.

**03 · gold** — Three monthly marts (`gold_property_sales`,
`gold_property_rent`, `gold_property_yield`) plus a `gold_quality_summary`
rollup, sized for the 2X-Small warehouse to scan in well under a second.
Additive legs (`total_*`, `n_*`) come first so any rollup to quarter or region
recomputes ratios correctly by summing legs, never by averaging averages.
Tables and columns carry `COMMENT`s lifted from the dbt project's docs, which
is what lets Genie answer questions against them in plain English.

**04 · checks** — a port of the dbt project's singular tests, run last so a
bad build fails the job instead of quietly reaching the dashboard: grain
uniqueness on each mart, coverage floors (did the build silently produce
almost nothing), and a plausibility band on yield (0.3–25% — outside that is a
units bug, not a market).

Keeping bad rows in silver rather than filtering them means the price dashboard
and the data-quality dashboard read the same table — so "what happened to the
missing rows?" has an answer in the data instead of in someone's memory.

---

## Why the logic isn't in the notebooks

`src/lib/transforms.py` holds every cleaning rule as a pure DataFrame→DataFrame
function. The notebooks import it and do orchestration only.

That buys a 5-second local test cycle (`make test`) instead of a 4-minute job
run, and it makes the rules reviewable as code. The tests cover each rule
individually, plus the cases that are easy to get wrong: all three numeric
encoding eras in the source data landing on the same value, rows failing
several checks at once, and the guarantee that gold's totals exclude exactly
what quality-summary reports.

One real bug surfaced this way: Databricks serverless SQL runs in ANSI mode,
where `to_date`/casts *throw* on malformed input (the real data contains a
genuine 6-digit date, `'170823'`), while local Spark used for the unit tests
silently returns `NULL` for the same input under non-ANSI mode. Fixed by
switching to `try_to_timestamp` and regex-guarded casts everywhere date and
numeric parsing happens, with a test fixture added to cover it.

---

## Checking gold against an independent reference

`scripts/parity_check.py` recomputes the gold marts a third way — DuckDB
running SQL transcribed from `data-qa-agent`'s dbt models, straight from the
canonical CSV partitions in `propertyiq_getdata` — and diffs the result
against what the Databricks job produced, month by month:

```bash
uv run python scripts/parity_check.py --profile DEFAULT
```

Additive metrics must match exactly and the script exits non-zero if they
don't; three known, documented labelling differences between the dbt and
Spark ports are reported but not enforced. See the module docstring for
details.

## Genie space

`genie/propertyiq.genie.json` is the PropertyIQ Genie space definition,
checked in for reproducibility only — Genie spaces aren't a bundle resource
type, so `make deploy` doesn't touch it. Recreate it with
`databricks genie create-space`. That API is picky about two things: every
`data_sources.tables` and instruction / sample-question list must be sorted by
identifier or id, and each entry needs a caller-supplied lowercase 32-hex id.

---

## Free Edition constraints, and what each one forced

| Constraint | Consequence here |
|---|---|
| Serverless compute only | No `job_clusters:` anywhere — omitting compute *is* the config |
| One 2X-Small warehouse | Dashboards read pre-aggregated gold, never row-level silver |
| 5 concurrent job tasks | Job is a 4-task sequential DAG |
| Restricted outbound internet | Source data is published into a UC volume ahead of time by a separate local pipeline, never downloaded by the workspace itself |
| No account-level API | Everything is workspace-scoped; no service principals |
| Non-commercial use only | Public open government data only (NSW Valuer General, Rental Bond Board) — no proprietary or customer data |

Sources: [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)

---

## Also worth knowing

**Git folders in the workspace.** You can clone this repo directly into
Databricks (Workspace → Repos → Add Repo) and commit from there. Useful as a
fallback, but the local→bundle path is better: it deploys jobs and dashboards
too, not just files. Use Git folders when you want the workspace to be the
source of truth for a session; use bundles when the repo is.

**Databricks Connect** is a third option — run Spark code from a local Python
process against remote serverless compute. Good for debugging a transform against
real data with a local debugger. Not wired up here because for this pipeline the
local pyspark tests are faster and the job run is more representative.
