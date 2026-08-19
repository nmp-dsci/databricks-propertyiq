"""Register the QA agent to Unity Catalog and stand up its serving endpoint.

Runs LOCALLY (uv run python scripts/register_agent.py) — deliberately not a
workspace job: langgraph isn't preinstalled on serverless job compute, but
logging from the laptop needs nothing workspace-side, and the serving image
build installs the model's own pip requirements.

Deploy order (s05 plan, decision D2 adjacent):
  1. Log the ChatAgent (models-from-code) and register propertyiq_ml.qa_agent.
  2. Smoke-test the registered model locally (load + one question).
  3. Try agents.deploy() — the Agent Framework path, verified live on Free
     Edition; clears older deployments of the same model first since the
     provisioned-concurrency quota fits about one served version per agent.
  4. Fall back to a plain CPU serving endpoint with auth passed via env vars
     backed by a secret scope, the same serving pattern the s03 rent
     estimator proved live. Retained for the workspaces where agents.deploy
     isn't available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CATALOG = "workspace"
ML_SCHEMA = "propertyiq_ml"
MODEL_NAME = f"{CATALOG}.{ML_SCHEMA}.qa_agent"
ENDPOINT_NAME = "propertyiq-qa-agent"


def log_and_register() -> str:
    from databricks.sdk import WorkspaceClient
    from mlflow.models.resources import (
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
        DatabricksTable,
    )

    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/{user}/propertyiq-qa-agent")

    with mlflow.start_run(run_name="qa-agent"):
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(REPO / "src" / "lib" / "qa_agent_model.py"),
            code_paths=[str(REPO / "src" / "lib")],
            pip_requirements=[
                # 3.1.3+ is the floor agents.deploy documents, and also the one
                # MLflow 3 real-time tracing needs.
                "mlflow>=3.1.3",
                # Must be in the SERVING image, not just on the laptop: without
                # databricks-agents>=1.2 in the container the endpoint cannot
                # stream traces, and the Traces tab shows "Upgrade to MLflow 3
                # to enable real-time tracing" even though mlflow itself is 3.x.
                "databricks-agents>=1.2.0",
                "langgraph>=1.0",
                "databricks-sdk>=0.68",
            ],
            # Declaring the resources is what lets agents.deploy() inject
            # automatic auth into the serving container — without these the
            # served agent has no credentials for the LLM or the warehouse
            # ("default auth: cannot configure default credentials").
            resources=[
                DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"),
                DatabricksSQLWarehouse(warehouse_id="7f9b6eb116a15acc"),
                # The endpoint's service principal needs UC grants on the
                # gold tables too — a declared warehouse alone yields
                # INSUFFICIENT_PERMISSIONS (no USE SCHEMA) at query time.
                DatabricksTable(table_name="workspace.propertyiq.gold_property_rent"),
                DatabricksTable(table_name="workspace.propertyiq.gold_property_sales"),
                DatabricksTable(table_name="workspace.propertyiq.gold_property_yield"),
            ],
            registered_model_name=MODEL_NAME,
        )
    version = info.registered_model_version
    print(f"registered {MODEL_NAME} v{version}")
    return version


def smoke_test(version: str) -> None:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    out = model.predict(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What's the median weekly rent for a 2-bedroom unit "
                        "in postcode 2000 right now?"
                    ),
                }
            ]
        }
    )
    answer = out["messages"][-1]["content"]
    print(f"smoke test answer: {answer[:200]}")
    assert answer, "empty answer from registered model"


def deploy(version: str) -> None:
    try:
        from databricks import agents

        # Set here as well as in log_and_register so `deploy` works standalone:
        # without these, agents.deploy resolves the logged model against a local
        # sqlite store and fails with "Logged model not found".
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

        # Free Edition's provisioned-concurrency quota fits about one served
        # version per agent, and agents.deploy accumulates them until it fails
        # with "Quota Exceeded". Clearing the old ones first is what makes a
        # re-deploy routine instead of a cleanup exercise.
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
        return
    except Exception as exc:  # noqa: BLE001 — Free Edition probe, fall through
        print(f"agents.deploy failed ({type(exc).__name__}: {exc}); trying plain serving")

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    w = WorkspaceClient()
    served = ServedEntityInput(
        name="qa-agent",
        entity_name=MODEL_NAME,
        entity_version=str(version),
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars={
            "DATABRICKS_HOST": "{{secrets/propertyiq/host}}",
            "DATABRICKS_TOKEN": "{{secrets/propertyiq/token}}",
        },
    )
    existing = {e.name for e in w.serving_endpoints.list()}
    if ENDPOINT_NAME in existing:
        w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=[served])
        print(f"updated endpoint {ENDPOINT_NAME} -> v{version}")
    else:
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(served_entities=[served]),
        )
        print(f"created endpoint {ENDPOINT_NAME} (v{version}); build takes a while")


if __name__ == "__main__":
    v = log_and_register()
    smoke_test(v)
    if "--no-deploy" not in sys.argv:
        deploy(v)
