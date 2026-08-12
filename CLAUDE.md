# CLAUDE.md — databricks-spike

Local-first Databricks development. Code is authored here and deployed to a
**Databricks Free Edition** workspace with Asset Bundles. Read `README.md` for
the full picture; this file is the working agreement.

## Non-negotiables

- **Serverless only.** Free Edition has no classic compute. Never add a
  `job_clusters:`, `new_cluster:`, `node_type_id` or `num_workers` block —
  omitting compute config *is* the serverless config.
- **No outbound internet.** Free Edition egress is restricted to trusted
  domains. Never write code that downloads a dataset, hits a public API, or
  `pip install`s at runtime from an arbitrary index. Data is generated in-workspace.
- **No real data.** Free Edition is non-commercial use only, and the interview
  brief forbids proprietary or confidential data. Synthetic only.
- **Never commit `.env`, tokens, or `~/.databrickscfg` contents.**

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
```

Prefer `make ship` over calling `databricks` directly, so tests always run first.

## When changing a transform

1. Change `src/lib/transforms.py`.
2. Add or update a test in `tests/test_transforms.py` covering the new rule.
3. `make test`.
4. Only then `make deploy && make run`.

Do not verify a logic change by running the job — it is 100× slower and the
failure message is worse.

## When changing the dashboard

Editing `.lvdash.json` by hand is fine for small things (a title, a query) but
layout and formatting are far quicker in the UI. If you change it in the UI, run
`make pull-dashboard` before committing or the next deploy will conflict.

## Context

This repo backs preparation for a Databricks Solutions Architect interview loop
(design/architecture round, a live pair-programming round on Free Edition, and a
build-demo-pitch round). Code should be **explainable out loud**: prefer the
approach whose trade-off is easy to articulate over the clever one. Comments
should say *why*, since that is what gets asked.
