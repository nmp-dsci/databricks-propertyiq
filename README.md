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
  jobs/ml_train.yml            feature build → train → promotion gate (on demand + drift-triggered)
  jobs/ml_score.yml            feature build → batch score → drift → conditional retrain
  jobs/ml_forecast.yml         AI_FORECAST vs trained model vs seasonal-naive comparison
  jobs/rag_ingest.yml          bronze land → silver merge → checks → index sync, file-arrival triggered
  pipelines/medallion_pipeline.yml  Declarative Pipeline variant of the medallion, side by side
  serving/rent_estimator_endpoint.yml  parked — Free Edition caps custom endpoints at two, both taken by the agents
  dashboards/property_overview.yml
  dashboards/agent_benchmark.yml       verdict dashboard for the QA-agent benchmark
  dashboards/rag_eval.yml              verdict dashboard for the RAG retrieval benchmark
genie/propertyiq.genie.json    Genie space definition (checked in for reproducibility, not a bundle resource)
src/
  notebooks/*.py               Databricks source-format notebooks (render as notebooks up there)
  notebooks/ml/*.py            feature build, train/register/gate, batch score, drift, forecast compare
  notebooks/rag/*.py           bronze land, silver merge, checks, index sync (embed + upsert + retire)
  lib/transforms.py            the medallion logic — pure, unit-tested, no I/O
  lib/ml_features.py, ml_model.py, ml_forecast.py, ml_monitoring.py   the ML logic — same pattern
  lib/qa_agent.py               the LangGraph QA agent — route → plan_sql → execute → reflect → answer
  lib/qa_agent_model.py         MLflow models-from-code entrypoint (the agent as a ChatAgent)
  lib/agent_eval.py             deterministic graders for the QA-agent benchmark
  lib/rag_export.py             pure transforms for the transcript-lab export (hash, land, merge)
  lib/rag_agent.py              rag_transcript_agent — single/multi/agentic_rag modes, LLM + SQL injected
  lib/rag_agent_model.py        MLflow models-from-code entrypoint (the RAG agent as a ChatAgent)
  lib/ir_metrics.py             retrieval-quality graders for the RAG benchmark
  pipelines/propertyiq/        the Declarative Pipeline variant's transformations (own README)
  sql/*.sql                    warehouse queries, runnable from the terminal
dashboards/*.lvdash.json       AI/BI dashboards, deployed as code
evals/golden_qa.yaml           confirmed golden Q&A set for the QA-agent benchmark
tests/                         local Spark tests, no workspace required
scripts/                       setup, auth, SQL runner, parity check, agent register + benchmark,
                                RAG export (+ dry-run), RAG agent register, RAG eval, chroma subprocess helpers
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
make deploy           # 1. push both dashboards/*.lvdash.json
                      # 2. rearrange / restyle in the Databricks UI
make pull-dashboard   # 3. write the UI version of both back into the repo
git commit            # 4. it's code again
```

`pull-dashboard` loops over both dashboard resources — `property_overview` and
`agent_benchmark` — so a UI edit to either one round-trips the same way.

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

**A second implementation, for comparison.** `resources/pipelines/medallion_pipeline.yml`
and `src/pipelines/propertyiq/` do the same bronze/silver/gold work as a
Lakeflow Declarative Pipeline, importing the exact same `transforms.py`
functions. It writes to its own schema (`propertyiq_dp`), has no trigger, and
never runs unless you type `databricks bundle run medallion_pipeline` — it
exists so the two frameworks can be read side by side, not as a migration. See
`src/pipelines/propertyiq/README.md` for what actually differs between them.

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

## ML: a rent estimator through a full MLOps loop, plus a forecasting comparison

Three model shapes on top of the gold marts, each demonstrating a different
piece of production ML on Free Edition rather than chasing accuracy:

**The rent estimator loop** (`resources/jobs/ml_train.yml`,
`resources/jobs/ml_score.yml`, `src/notebooks/ml/01-04`, `src/lib/ml_features.py`,
`ml_model.py`, `ml_monitoring.py`):

- **`01_build_features.py`** materialises `<ml_schema>.features_rent` from the
  gold marts, with an informational primary key
  (`postcode, property_type, bedroom_band, month`). The same function backs
  both training and batch scoring, so there is no second feature
  implementation to drift out of sync.
- **`02_train_register.py`** trains a challenger, logs it to MLflow, registers
  it in Unity Catalog, then runs the **promotion gate**: the challenger only
  takes the `@champion` alias if it beats the incumbent by a 2% MAE margin on
  the same fresh holdout window (the rule is unit-tested in
  `tests/test_ml_model.py`). Losing challengers stay registered under
  `@challenger`, so the version history is the audit trail. Promotion also
  rolls the serving endpoint forward to the new champion version, since
  serving config pins a version and cannot follow an alias by itself.
- **`03_batch_score.py`** scores the latest month with whatever `@champion`
  currently points at and stamps every row with `model_version` and
  `scored_at`. Scoring is **driver-side pandas**, not
  `mlflow.pyfunc.spark_udf` — the scale-out UDF path is currently broken on
  serverless (its mlflow fails parsing the runtime version string
  `18.x-aarch64-photon-scala2`), and at ~2.5k rows/month a single in-process
  `predict()` is the right tool anyway. Documented in the notebook rather than
  worked around.
- **`04_drift_metrics.py`** computes PSI per feature (latest month vs a
  **trailing 24-month** reference — an all-history reference would make the
  trend in rents itself look like permanent drift) and MAE per month.
  `month_of_year` is excluded from PSI, since a cyclical feature always reads
  as drifted against a full-year window. A feature with an empty reference or
  current window is labelled `unmonitorable` rather than `stable` — PSI is
  `NaN` there, and `NaN > threshold` is always false, so leaving it as
  `stable` would silently read as a clean bill of health. The job's
  `check_drift` condition task reads `max_monitorable_psi()`, the worst PSI
  over features that could actually be compared (excluding `unmonitorable`
  ones, so a `NaN` never propagates into the condition and defeats it);
  above 0.25 ("shifted") it fires `ml_train` automatically via a
  `run_job_task` — retraining triggers because the input moved, not on a
  schedule.

`ml_score` runs off a table-update trigger on `gold_property_rent`, one hop
downstream of the medallion job, same event-driven pattern as the medallion's
file-arrival trigger. `ml_train` has no trigger of its own: it runs on demand
(`databricks bundle run ml_train -t dev`) or when `ml_score` triggers it.

**Real-time serving** (`resources/serving/rent_estimator_endpoint.yml`,
`src/sql/03_ml_ai_query.sql`): the champion behind a scale-to-zero CPU serving
endpoint, queryable over REST or from plain SQL via
`ai_query('propertyiq-rent-estimator', ...)` on the same 2X-Small warehouse
the dashboards use. First request after idle pays a cold start — the
trade-off for a weekly-data demo endpoint costing nothing between queries.

**The forecasting comparison** (`resources/jobs/ml_forecast.yml`,
`src/notebooks/ml/05_forecast_compare.py`, `src/lib/ml_forecast.py`): the same
task, three tools, one backtest. `AI_FORECAST` (the platform SQL table
function, driven over the Statement Execution API with polling past its 50s
synchronous ceiling since forecasting hundreds of series runs longer), a
trained per-series trend + monthly-seasonality model (sklearn — pre-installed,
so no runtime `pip install`; `prophet`/`lightgbm` were ruled out for the same
reason), and seasonal-naive as the baseline either must beat. All three
forecast the same held-out months from the same truncated history; the
**comparison table** (`<ml_schema>.forecast_comparison`) is the deliverable,
not the forecast itself. No trigger — run on demand:
`databricks bundle run ml_forecast -t dev`.

ML tables live in their own `<ml_schema>` (default `propertyiq_ml`), the same
isolation pattern as the pipeline's `propertyiq_dp`: the ML jobs can never
write over the marts the medallion job owns.

---

## Genie space

`genie/propertyiq.genie.json` is the PropertyIQ Genie space definition,
checked in for reproducibility only — Genie spaces aren't a bundle resource
type, so `make deploy` doesn't touch it. Recreate it with
`databricks genie create-space`. That API is picky about two things: every
`data_sources.tables` and instruction / sample-question list must be sorted by
identifier or id, and each entry needs a caller-supplied lowercase 32-hex id.

---

## QA-agent benchmark: Genie vs LangGraph vs Data Pilot

Three ways to answer natural-language questions over the gold marts, scored
against the same confirmed golden question set: the Genie space above, a new
LangGraph agent, and the external `data-qa-agent` app ("Data Pilot", the
sibling project this pipeline's dbt logic was ported from).

**`src/lib/qa_agent.py`** is a deliberate port of Data Pilot's data-agent —
the same graph (`route → plan_sql → execute → reflect → answer`, with a
decision log at every step) re-grounded on the Unity Catalog gold tables and
the serverless SQL warehouse instead of dbt's manifest and Postgres. The graph
is dependency-injected (an LLM callable, a SQL executor), so `tests/test_qa_agent.py`
exercises the whole flow — routing, SQL guardrails, retry-on-error, refusal —
without a network call. `make_databricks_agent` wires the real pieces: FMAPI
via `databricks-sdk` for the LLM leg and the Statement Execution API for SQL.
`databricks-sdk` is used instead of `databricks-langchain` deliberately —
the latter drags in `databricks-connect`, which shadows local `pyspark` and
would break the Spark unit tests.

**`src/lib/qa_agent_model.py`** wraps the graph as an MLflow `ChatAgent`
(models-from-code), so the same file serves the registered model, the
deployed endpoint, and local parity testing. Configuration is environment
variables, with the repo's standing warehouse id (`7f9b6eb116a15acc`) as the
default.

**`scripts/register_agent.py`** (`make register-agent`) logs and registers
the agent to `propertyiq_ml.qa_agent`, smoke-tests it locally, then stands up
a serving endpoint — trying `agents.deploy()` first and falling back to a
plain CPU serving endpoint. It declares the LLM endpoint, SQL warehouse and
the three gold tables as model resources at log time, since `agents.deploy`'s
auto-auth needs them declared to avoid `INSUFFICIENT_PERMISSIONS`. This runs
**locally**, not as a workspace job — `langgraph` isn't preinstalled on
serverless job compute, and logging from the laptop needs nothing
workspace-side. `deploy()` sets the MLflow tracking and registry URIs itself
so it also works standalone (otherwise `agents.deploy` resolves the logged
model against a local sqlite store and fails with "Logged model not found"),
and deletes older deployments of the same model before creating the new one —
Free Edition's provisioned-concurrency quota fits about one served version per
agent, and `agents.deploy` otherwise accumulates versions until it fails with
`Quota Exceeded`.

**Real-time tracing on the deployed endpoint** needs the same two things as
`rag_transcript_agent` (below): `databricks-agents>=1.2.0` in the model's
`pip_requirements`, baked into the serving image rather than just installed on
the laptop, and explicit spans, since this agent also calls FMAPI and the SQL
warehouse via `databricks-sdk` rather than `databricks-langchain` so no
autologger captures those calls. `ask()` carries an
`@mlflow.trace(span_type="AGENT")` root — without it every LLM and SQL call
would land as its own orphan trace, and this agent retries a failed statement
once, which would scatter one logical answer across several unrelated traces.
The chat calls are traced as `LLM`; the warehouse statement is traced `TOOL`
rather than `RETRIEVER`, because this agent answers by executing generated SQL
rather than retrieving documents, and the span's input is the statement
itself — the most useful thing to see when an answer looks wrong.

**`evals/golden_qa.yaml`** is the confirmed golden question set: each case
has a question, a grader spec, and an answer computed by deterministic SQL
against the gold tables and confirmed by hand. Notably, "currently" /
"right now" is defined as the *latest month present in the data* — a reading
Genie's own trailing-12-month default legitimately fails.

**`src/lib/agent_eval.py`** grades an agent's free-text answer against the
golden case deterministically (no LLM judge decides pass/fail): `value`
(a number within tolerance), `topk` (mentions enough of the expected keys),
or `refusal` (declines rather than fabricates, for out-of-scope questions).

**`scripts/run_benchmark.py`** (`make benchmark`) runs every golden case
through all three contenders and writes results to
`workspace.propertyiq_ml.agent_benchmark` on the warehouse. It runs
**locally**, not as a workspace job, because only the laptop can reach all
three: Genie and the deployed LangGraph endpoint are workspace HTTPS APIs,
but Data Pilot is the `data-qa-agent` stack on `localhost:8010`. The
LangGraph agent is queried through its deployed serving endpoint over raw
`/invocations` (ChatAgent endpoints reply with `{"messages": [...]}`, not the
chat-completions `choices` shape the SDK helper expects), falling back to
running the graph in-process if the endpoint isn't ready.

```bash
make register-agent   # log + register the agent, stand up its endpoint
make benchmark        # run all three contenders over the golden set
```

**`dashboards/agent_benchmark.lvdash.json`** (`resources/dashboards/agent_benchmark.yml`)
is the verdict dashboard, deployed like `property_overview` — pass rates and
latency per contender, read straight from `agent_benchmark`.

---

## RAG: transcript·lab's corpus, ported onto Vector Search

The sibling `transcript-rag-agent` project (transcript·lab) is an
evaluation-first RAG workbench over ~100 YouTube transcripts — local Chroma,
MiniLM embeddings, a 20-question golden set with real IR metrics. It had no
Databricks code at all. This branch moves its corpus and its answer paths onto
the lakehouse and then *measures whether that was an improvement*.

**The ETL is a push, because Free Edition cannot pull.** `make rag-export`
snapshots every store in that project and lands one Parquet per entity on
`/Volumes/workspace/rag/landing/<entity>/`, named `<entity>_<sha8>.parquet` —
the same content-hashed, append-only contract `propertyiq_getdata` uses for the
property feed. Unchanged content is never uploaded, so the command is safe to
run whenever and a no-op run costs nothing. That is also why it is manual
rather than scheduled: it is idempotent, and most hours there is nothing new.
`make rag-export-dry` runs the same hash-and-diff without uploading, to check
what a real run would land.

Reading the corpus needs `chromadb` (the vectors live in Chroma's HNSW segment
files, not its SQLite), so `scripts/_chroma_dump.py` runs in *transcript-lab's
own virtualenv* as a subprocess. Pulling chromadb in here would drag
onnxruntime alongside pyspark for no benefit.

| Landed | Rows | Table |
|---|---|---|
| chunks (with their MiniLM vectors) | 2,966 | `silver_chunks` |
| transcripts / segments / summaries | 107 / 80,836 / 105 | `silver_transcripts`, `silver_segments`, `silver_summaries` |
| golden questions | 20 | `silver_golden_qa` |
| graph entities / relations / claims | 14,809 / 7,508 / 9,295 | `silver_graph_*` |

`rag_ingest` (file-arrival triggered) runs Auto Loader → bronze → a silver
reconciliation. Each file is a **full snapshot**, so silver `MERGE`s the newest
one on `(video_id, chunk_index)` and marks anything absent as
`is_current = false` — a soft delete, because transcript·lab recreates chunk
ids on every re-index, so a "missing" row is usually a re-chunk rather than a
retraction. That is also why the merge key is the position plus a `text_sha`
and never the chunk id.

A soft delete in silver still has to be enforced in Vector Search: after
`04_index_sync.py` pushes changed chunks to both indexes, it also **retires**
any key those indexes hold that silver no longer marks current — otherwise a
re-chunked video leaves stale text behind that the agent can still retrieve
and cite, which is worse than a miss because it looks authoritative.

**`rag_transcript_agent`** (`make register-rag-agent`) serves three of
transcript·lab's answer paths from one deployment, chosen per call with
`custom_inputs={"mode": ...}`: `single` (one retrieval), `multi` (decompose →
retrieve per sub-question → synthesize) and `agentic` (a ReAct loop, the
default). `custom_outputs` returns the retrieved chunk keys and the decision
log, which is what lets the eval score *retrieval* rather than only prose.
Answers cite video title and timestamp. `agents.deploy` gives it a Review App.

**Real-time tracing on the deployed endpoint** needs two things beyond a
recent `mlflow`, both easy to miss because they fail silently rather than
erroring: `databricks-agents>=1.2.0` in the model's `pip_requirements` (it has
to be baked into the *serving image*, not just installed on the laptop, or the
Traces tab shows "Upgrade to MLflow 3 to enable real-time tracing" regardless
of the `mlflow` version), and explicit spans, since this agent calls FMAPI and
Vector Search via `databricks-sdk` rather than `databricks-langchain` (same
tradeoff as `qa_agent`, above) so no autologger captures those calls. `ask()`
carries an `@mlflow.trace(span_type="AGENT")` root — serving opens no root span
of its own for a `ChatAgent`, so without it every retriever and LLM call below
would land as its own orphan trace instead of one tree — and `retrieve`/`llm`
are traced as `RETRIEVER`/`LLM` respectively. The `mode` tag is applied inside
`ask()` via `mlflow.update_current_trace`, not in `ChatAgentModel.predict`,
because no trace exists to tag until `ask()` opens one.

**The verdict** (`make rag-eval`, dashboard `rag_eval.lvdash.json`) runs the
golden set against three retrievers so each comparison moves one variable:

| Contender | recall@10 | MRR | NDCG@10 | What it isolates |
|---|---|---|---|---|
| `chroma` | 0.633 | 0.529 | 0.522 | the local baseline |
| `vs_minilm` | 0.633 | 0.529 | 0.522 | **the engine** — same vectors, Databricks |
| `vs_gte` | 0.583 | 0.625 | 0.578 | **the model** — managed `gte-large-en` |

`chroma` and `vs_minilm` matching *exactly* is the control working: Vector
Search reproduces the local index when given the same vectors, so the port is
faithful. Swapping in gte then trades a little recall for noticeably better
ranking — it finds slightly fewer of the expected videos but puts the right one
first more often. Scoring is on video ids because they are stable; the golden
set's chunk anchors were verified against a 23-video corpus (it is 107 now) and
are re-anchored by content hash, with whatever fails to resolve reported rather
than scored as a silent miss.

---

## Free Edition constraints, and what each one forced

| Constraint | Consequence here |
|---|---|
| Serverless compute only | No `job_clusters:` anywhere — omitting compute *is* the config |
| One 2X-Small warehouse | Dashboards read pre-aggregated gold, never row-level silver |
| 5 concurrent job tasks | Job is a 4-task sequential DAG |
| Restricted outbound internet | Source data is published into a UC volume ahead of time by a separate local pipeline, never downloaded by the workspace itself; the forecast model is sklearn (pre-installed), never `prophet`/`lightgbm` via runtime `pip install` |
| No account-level API | Everything is workspace-scoped; no service principals |
| Non-commercial use only | Public open government data only (NSW Valuer General, Rental Bond Board) — no proprietary or customer data |
| `mlflow.pyfunc.spark_udf` broken on serverless | Batch scoring (`03_batch_score.py`) predicts driver-side with pandas instead of a distributed UDF — also the right scale for ~2.5k rows/month |
| Serving config pins a model version, not a UC alias | The training notebook rolls the endpoint's pinned version forward in the same run that flips `@champion` |
| `langgraph` not preinstalled on serverless job compute | Agent registration (`make register-agent`) and the QA-agent benchmark (`make benchmark`) run locally, not as workspace jobs |
| Vector Search **delta-sync** indexes never provision | Their backing pipeline sits on "pending setup of pipeline resources" indefinitely, even though the endpoint reports ONLINE and **direct-access indexes work fine**. `04_index_sync.py` therefore does by hand what delta-sync would do: embed changed chunks with `ai_query`, upsert them, keyed on `text_sha` so nothing is re-embedded needlessly |
| Vector Search queries reject control-plane calls | The index lives behind its own data-plane host, so a raw `api_client.do` query returns `PermissionDenied` — use the typed `w.vector_search_indexes.query_index` instead (upserts, oddly, work either way) |
| **Two** custom model-serving endpoints | Both slots are held by agents (`qa_agent`, `rag_transcript_agent`), so the rent estimator's endpoint is parked (`resources/serving/rent_estimator_endpoint.yml`) — a third fails the whole bundle deploy, not just its own resource. Swapping back is a documented two-step |

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
