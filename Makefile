.DEFAULT_GOAL := help
SHELL := /bin/bash

TARGET ?= dev
PROFILE ?= DEFAULT
# Extra bundle flags, e.g. make deploy DB_VARS='--var=warehouse_id=abc123'
DB_VARS ?=
DB := databricks --profile $(PROFILE)

# ---------------------------------------------------------------------------
# Local loop — no workspace needed
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install python deps and the Databricks CLI
	@bash scripts/setup.sh

.PHONY: test
test: ## Run transform unit tests on local Spark
	@uv run pytest -q

.PHONY: lint
lint: ## Lint and format-check
	@uv run ruff check .
	@uv run ruff format --check .

.PHONY: fmt
fmt: ## Auto-format
	@uv run ruff check --fix .
	@uv run ruff format .

# ---------------------------------------------------------------------------
# Workspace loop
# ---------------------------------------------------------------------------

.PHONY: auth
auth: ## Log the CLI in to your workspace (OAuth, falls back to a PAT)
	@bash scripts/auth.sh

.PHONY: validate
validate: ## Type-check the bundle without deploying
	@$(DB) bundle validate -t $(TARGET) $(DB_VARS)

.PHONY: deploy
deploy: validate ## Push notebooks, jobs and dashboards to the workspace
	@$(DB) bundle deploy -t $(TARGET) $(DB_VARS)

.PHONY: run
run: ## Run the medallion job and stream its output here
	@$(DB) bundle run medallion -t $(TARGET) $(DB_VARS)

.PHONY: ship
ship: test deploy run ## test -> deploy -> run, the one command for the inner loop

.PHONY: sync
sync: ## Live-sync this folder to the workspace on every save (leave running)
	@$(DB) sync --watch . "/Workspace/Users/$$($(DB) current-user me | jq -r .userName)/live/databricks-propertyiq"

.PHONY: sql
sql: ## Run a .sql file on the serverless warehouse. Usage: make sql FILE=src/sql/01_explore.sql
	@uv run python scripts/run_sql.py $(FILE)

.PHONY: register-agent
register-agent: ## Log + register the QA agent to UC and roll its endpoint
	@uv run python scripts/register_agent.py

.PHONY: benchmark
benchmark: ## Run the 3-way QA benchmark (Genie / LangGraph / Data Pilot) locally
	@uv run python scripts/run_benchmark.py

.PHONY: rag-export
rag-export: ## Snapshot transcript-lab's corpus and land what changed on the volume
	@uv run python scripts/rag_export.py

.PHONY: rag-export-dry
rag-export-dry: ## Same, but build and hash only — uploads nothing
	@uv run python scripts/rag_export.py --dry-run

.PHONY: pull-dashboard
pull-dashboard: ## Pull UI edits back into dashboards/*.lvdash.json before committing
	@for resource in property_overview agent_benchmark; do \
		$(DB) bundle generate dashboard --resource $$resource -t $(TARGET) --force; \
	done

.PHONY: summary
summary: ## Print deployed resource URLs
	@$(DB) bundle summary -t $(TARGET) $(DB_VARS)

.PHONY: destroy
destroy: ## Remove everything this bundle deployed
	@$(DB) bundle destroy -t $(TARGET)

.PHONY: help
help:
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
