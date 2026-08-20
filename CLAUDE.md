# CLAUDE.md — databricks-propertyiq

Local-first Databricks development. Code is authored here and deployed to a
**Databricks Free Edition** workspace with Asset Bundles. Read `README.md` for
the full picture; this file is the working agreement.

## Non-negotiables

- **Serverless only.** Free Edition has no classic compute. Never add a
  `job_clusters:`, `new_cluster:`, `node_type_id` or `num_workers` block —
  omitting compute config *is* the serverless config.
- **No outbound internet.** Free Edition egress is restricted to trusted
  domains. Never write code that downloads a dataset, hits a public API, or
  `pip install`s at runtime from an arbitrary index — this is about the
  workspace's own egress, not about how source data was originally obtained.
- **Public open data only.** This repo's data is NSW Valuer General property
  sales and NSW Rental Bond Board rental lodgements — both public, open
  government datasets, published as an updates-only Parquet feed into a Unity
  Catalog volume (under `landing/sales/` and `landing/lodgements/`) by a
  separate local pipeline (`propertyiq_getdata`) that runs outside the
  workspace. Free Edition is non-commercial use only. Never add proprietary,
  confidential, commercial, or personal data.
- **Never commit `.env`, tokens, or `~/.databrickscfg` contents.**
- **Never write a person's name into this repo.** See below — this one is easy to
  breach by accident.

## No personal names, anywhere in this repo

Nothing committed here may contain the name of a real person: not recruiters,
interviewers, hiring managers, coordinators, panel members, colleagues, customers
or referees. This holds for code, comments, commit messages, docs, notebooks,
test fixtures, synthetic data, and `.lavish/` artifacts alike.

**Why:** this repo is public. Prep material drawn from a private interview pack
routinely carries the names of people who never agreed to appear in a public
GitHub repo, and a search engine does not care that the surrounding context was
flattering.

**How to apply:** use the role instead — "the recruiter", "the coordinator", "the
hiring manager", "the panel", "a Solutions Architect". Roles carry every bit of
meaning the name did for planning purposes. Process detail, round structure,
evaluation criteria and public product facts are all fine to include; they are
common knowledge. It is specifically identities that stay out.

Names belong in the private sibling project (`../ai-engineer-fit/`), not here. If
you are pasting from there, scrub as you paste rather than afterwards.

## File conventions

- `src/notebooks/*.py` are **Databricks source-format notebooks**. They must
  start with `# Databricks notebook source`, separate cells with
  `# COMMAND ----------`, and write markdown as `# MAGIC %md`. Do not convert
  them to `.ipynb` — the whole point is that git sees clean Python.
  Ruff excludes this directory: `spark`, `dbutils` and `display` are injected
  globals and would otherwise flood the lint output.
- **Logic goes in `src/lib/transforms.py`, not in a notebook.** Every function
  there is pure (DataFrame in, DataFrame out — no I/O, no globals) so it can be
  tested locally. Notebooks orchestrate: read a table, call a transform, write a
  table, `display()` something. If you find yourself writing a `when/otherwise`
  chain in a notebook cell, it belongs in `transforms.py` with a test.
- `dashboards/*.lvdash.json` is deployed **verbatim** — bundle variables are not
  substituted inside it, so table names are fully qualified and hardcoded.

## Commands

```bash
make test                        # local Spark unit tests, no workspace (~5s)
make lint / make fmt             # ruff
make validate                    # check the bundle without deploying
make deploy                      # push notebooks, job and dashboard
make run                         # run the job, logs stream to the terminal
make ship                        # test -> deploy -> run
make sql FILE=src/sql/01_explore.sql
make pull-dashboard              # pull UI dashboard edits back into the repo
make register-agent              # log + register the QA agent to UC, roll its endpoint
make benchmark                   # run the 3-way QA benchmark (Genie / LangGraph / Data Pilot) locally
make rag-export                  # land transcript-lab's full corpus on the volume (manual, idempotent)
make register-rag-agent          # log + register rag_transcript_agent to UC and deploy it
make rag-eval                    # score Chroma vs both Vector Search indexes on the golden set
make rag-goldens                 # capture + judge dev agentic answers -> silver_golden_answers
make rag-judge VARIANT=A0        # judge one local agent variant with the dev parity judge
make agent-eval                  # LLM-as-judge over the LIVE endpoint, entirely in-workspace
```

Prefer `make ship` over calling `databricks` directly, so tests always run first.

The ML jobs (`ml_train`, `ml_score`, `ml_forecast`) aren't wired into `make
run`/`make ship` — they run on demand via the CLI directly, e.g.
`databricks bundle run ml_train -t dev`. `ml_train` also runs automatically
when `ml_score`'s drift check trips; `ml_score` itself runs off a table-update
trigger on `gold_property_rent`. See `README.md` for the full ML loop.

The `rag_ingest` job (bronze land → silver merge → checks → index sync) is
also not wired into `make run`/`make ship` — run it via `databricks bundle
run rag_ingest -t dev` after a `make rag-export`. See `README.md` for the
full RAG pipeline.

## When changing a transform

1. Change `src/lib/transforms.py`.
2. Add or update a test in `tests/test_transforms.py` covering the new rule.
3. `make test`.
4. Only then `make deploy && make run`.

Do not verify a logic change by running the job — it is 100× slower and the
failure message is worse.

The same discipline applies to `src/lib/ml_features.py`, `ml_model.py`,
`ml_forecast.py` and `ml_monitoring.py` — pure functions, tested in
`tests/test_ml_*.py`, imported by `src/notebooks/ml/*.py` for orchestration
only.

`src/lib/qa_agent.py` (the LangGraph QA agent) and `src/lib/agent_eval.py`
(the benchmark's graders) follow the same pattern: the graph is
dependency-injected (LLM callable, SQL executor) so `tests/test_qa_agent.py`
and `tests/test_agent_eval.py` run it without a network call. Changing the
grounding rules or a grader requires a corresponding golden case in
`evals/golden_qa.yaml` and a re-run of `make benchmark` — do not tune a
grader against a single agent's phrasing without checking the others still
pass.

## When changing the dashboard

Editing `.lvdash.json` by hand is fine for small things (a title, a query) but
layout and formatting are far quicker in the UI. If you change it in the UI, run
`make pull-dashboard` before committing or the next deploy will conflict.

## Context

This repo backs preparation for a Databricks Solutions Architect interview loop
(design/architecture round, a live pair-programming round on Free Edition, and a
build-demo-pitch round). The work product is PropertyIQ: a real end-to-end
pipeline that lands NSW property sales and rental data through
bronze/silver/gold and surfaces it in a deployed AI/BI dashboard. Code should
be **explainable out loud**: prefer the approach whose trade-off is easy to
articulate over the clever one. Comments should say *why*, since that is what
gets asked.
