"""Register rag_transcript_agent to Unity Catalog and deploy it.

Runs LOCALLY (`make register-rag-agent`) for the same reason the QA agent's
registration does: langgraph isn't preinstalled on serverless job compute, and
logging from the laptop needs nothing workspace-side — the serving image builds
from the model's own pip requirements.

Two lessons from the s05 agent are applied up front rather than rediscovered:

* **Declare every resource at log time.** agents.deploy injects automatic auth
  only for resources named in `mlflow.models.resources`. The QA agent needed
  three attempts to learn that the *tables* count too; here the Vector Search
  index, both serving endpoints and the source table are all declared on the
  first pass.
* **Delete old deployments before creating a new one.** Free Edition's
  provisioned-concurrency quota fits about one served agent version, and
  agents.deploy accumulates versions until it fails with "Quota Exceeded".

Usage:
  uv run python scripts/register_rag_agent.py
  uv run python scripts/register_rag_agent.py --no-deploy
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CATALOG = "workspace"
SCHEMA = "rag"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.rag_transcript_agent"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_chunks_gte"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"


def log_and_register() -> str:
    from databricks.sdk import WorkspaceClient
    from mlflow.models.resources import (
        DatabricksServingEndpoint,
        DatabricksTable,
        DatabricksVectorSearchIndex,
    )

    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/{user}/rag-transcript-agent")

    with mlflow.start_run(run_name="rag-transcript-agent"):
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(REPO / "src" / "lib" / "rag_agent_model.py"),
            code_paths=[str(REPO / "src" / "lib")],
            pip_requirements=[
                # 3.1.3+ is what agents.deploy needs; it is also the floor for
                # MLflow 3 real-time tracing.
                "mlflow>=3.1.3",
                # Must be in the SERVING image, not just this laptop: without
                # databricks-agents>=1.2 in the container, the endpoint cannot
                # stream traces and the Traces tab shows "Upgrade to MLflow 3
                # to enable real-time tracing" even though mlflow itself is 3.x.
                "databricks-agents>=1.2.0",
                "langgraph>=1.0",
                "databricks-sdk>=0.68",
            ],
            resources=[
                # The retriever's two legs: embed the question, then search.
                DatabricksVectorSearchIndex(index_name=INDEX_NAME),
                DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),
                # The answer leg.
                DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
                # The index's source table — a declared index alone still left
                # the QA agent without USE SCHEMA on its data.
                DatabricksTable(table_name=f"{CATALOG}.{SCHEMA}.silver_chunks"),
            ],
            registered_model_name=MODEL_NAME,
        )
    version = info.registered_model_version
    print(f"registered {MODEL_NAME} v{version}")
    return version


def smoke_test(version: str) -> None:
    """Load the registered model and ask one question in each mode."""
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    question = "What is Unity Catalog and why does it matter?"
    for mode in ("single", "agentic"):
        out = model.predict(
            {
                "messages": [{"role": "user", "content": question}],
                "custom_inputs": {"mode": mode},
            }
        )
        answer = out["messages"][-1]["content"]
        retrieved = (out.get("custom_outputs") or {}).get("chunk_keys") or []
        print(f"  {mode}: {len(retrieved)} chunks -> {answer[:110]}...")
        assert answer, f"empty answer in {mode} mode"


def deploy(version: str) -> None:
    from databricks import agents

    # Set here as well as in log_and_register so `deploy` works standalone:
    # without these, agents.deploy resolves the logged model against a local
    # sqlite store and fails with "Logged model not found".
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # Free Edition fits roughly one served version; clear the old ones first so
    # this deploy doesn't trip the provisioned-concurrency quota.
    try:
        for deployment in agents.list_deployments():
            if deployment.model_name == MODEL_NAME and str(deployment.model_version) != str(
                version
            ):
                print(f"  removing old deployment v{deployment.model_version}")
                agents.delete_deployment(MODEL_NAME, deployment.model_version)
    except Exception as exc:  # noqa: BLE001 — nothing deployed yet is fine
        print(f"  (no existing deployments to clear: {type(exc).__name__})")

    deployment = agents.deploy(MODEL_NAME, version, scale_to_zero=True)
    print(f"agents.deploy OK: {deployment.endpoint_name}")
    print(f"  review app: {getattr(deployment, 'review_app_url', 'n/a')}")


if __name__ == "__main__":
    v = log_and_register()
    smoke_test(v)
    if "--no-deploy" not in sys.argv:
        deploy(v)
