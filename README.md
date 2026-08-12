# databricks-spike

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
  jobs/medallion_job.yml       4-task serverless job (generate → bronze → silver → gold)
  dashboards/retail_overview.yml
src/
  notebooks/*.py               Databricks source-format notebooks (render as notebooks up there)
  lib/transforms.py            the actual logic — pure, unit-tested, no I/O
  sql/*.sql                    warehouse queries, runnable from the terminal
dashboards/*.lvdash.json       AI/BI dashboard, deployed as code
tests/                         local Spark tests, no workspace required
scripts/                       setup, auth, SQL runner
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
make deploy           # 1. push dashboards/retail_overview.lvdash.json
                      # 2. rearrange / restyle in the Databricks UI
make pull-dashboard   # 3. write the UI version back into the repo
git commit            # 4. it's code again
```

`embed_credentials: false` in the resource means a deploy **errors** rather than
silently overwriting UI edits you forgot to pull.

> The dashboard SQL hardcodes `workspace.retail_spike.*`. The `.lvdash.json` is
> deployed verbatim — bundle variables are not substituted inside it. If you
> change catalog or schema, change it there too.

---

## The pipeline

A medallion pipeline over synthetic retail orders. It is small on purpose; what
matters is that every layer has a defensible reason to exist.

**00 · generate** — Free Edition can't reach the public internet, so data is
synthesised with Spark and landed as JSON files in a UC Volume. Dirt is injected
deliberately: ~1% duplicates (at-least-once upstream), ~1.5% negative quantities
(returns mis-keyed), ~2% null customers, and junk country codes.

**01 · bronze** — Auto Loader (`cloudFiles`) with `Trigger.AvailableNow`:
streaming file-tracking semantics on a batch cadence. Append-only, schema
declared not inferred, unexpected fields land in `_rescued_data` instead of
failing the run. Bronze is the rebuild point when silver logic turns out wrong.

**02 · silver** — Typed, country codes normalised, deduplicated by
`row_number()` over ingest time (not `dropDuplicates`, so a late correction
deterministically wins). **Invalid rows are kept**, with a `_quality` array
naming every rule they failed. Delta `CHECK` constraints make the contract
enforceable rather than aspirational.

**03 · gold** — Two narrow aggregates the 2X-Small warehouse can scan instantly.
Tables and columns carry `COMMENT`s, which is what lets Genie answer questions
against them in plain English.

Keeping bad rows in silver rather than filtering them means the revenue dashboard
and the data-quality dashboard read the same table — so "what happened to the
missing 8%?" has an answer in the data instead of in someone's memory.

---

## Why the logic isn't in the notebooks

`src/lib/transforms.py` holds every cleaning rule as a pure DataFrame→DataFrame
function. The notebooks import it and do orchestration only.

That buys a 5-second local test cycle (`make test`) instead of a 4-minute job
run, and it makes the rules reviewable as code. The tests cover each rule
individually, plus the cases that are easy to get wrong: duplicates where the
later row corrected a value, rows failing several checks at once, and the
guarantee that gold's totals exclude exactly what quality-summary reports.

---

## Free Edition constraints, and what each one forced

| Constraint | Consequence here |
|---|---|
| Serverless compute only | No `job_clusters:` anywhere — omitting compute *is* the config |
| One 2X-Small warehouse | Dashboards read pre-aggregated gold, never row-level silver |
| 5 concurrent job tasks | Job is a 4-task sequential DAG |
| Restricted outbound internet | Data is generated in-workspace, never downloaded |
| No account-level API | Everything is workspace-scoped; no service principals |
| Non-commercial use only | Synthetic data only — no proprietary or customer data |

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
