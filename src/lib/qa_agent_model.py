"""MLflow models-from-code entrypoint: the QA agent as a ChatAgent.

Logged by scripts/register_agent.py. At serving time the endpoint (or
agents.deploy's wrapper) instantiates this file; locally the benchmark can
load it with mlflow.pyfunc.load_model for parity testing.

Configuration comes from environment variables so the same file serves in
every context; the committed warehouse id is the repo's standing default.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from lib.qa_agent import ask, make_databricks_agent


class PropertyIQQAAgent(ChatAgent):
    def __init__(self) -> None:
        self._agent = None

    def _graph(self):
        if self._agent is None:
            self._agent = make_databricks_agent(
                warehouse_id=os.environ.get("PROPERTYIQ_WAREHOUSE_ID", "7f9b6eb116a15acc"),
                catalog=os.environ.get("PROPERTYIQ_CATALOG", "workspace"),
                schema=os.environ.get("PROPERTYIQ_SCHEMA", "propertyiq"),
                model_endpoint=os.environ.get(
                    "PROPERTYIQ_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
                ),
            )
        return self._agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[Any] = None,  # noqa: UP045 — matches the ChatAgent signature
        custom_inputs: Optional[dict[str, Any]] = None,  # noqa: UP045
    ) -> ChatAgentResponse:
        question = messages[-1].content
        result = ask(self._graph(), question)
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=result["answer"],
                    id=str(uuid.uuid4()),
                )
            ],
            custom_outputs={"sql": result["sql"], "decision_log": result["decision_log"]},
        )


mlflow.models.set_model(PropertyIQQAAgent())
