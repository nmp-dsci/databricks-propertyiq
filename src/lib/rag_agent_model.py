"""MLflow models-from-code entrypoint: rag_transcript_agent as a ChatAgent.

Logged by scripts/register_rag_agent.py and served by agents.deploy, which also
gives it the Review App chat UI the acceptance test uses.

All three answer modes ship in this one deployment. The caller picks with
`custom_inputs={"mode": "single" | "multi" | "agentic"}`; the default is
agentic. Graphs are built lazily and cached per mode, so a cold container only
compiles the mode actually asked for.

`custom_outputs` hands back the retrieved chunk keys, the distinct video ids and
the decision log — that is what lets the eval score *retrieval* (recall@k, MRR,
NDCG against the golden set) rather than only the final prose.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

import mlflow
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse

from lib.rag_agent import DEFAULT_MODE, MODES, ask, make_databricks_agent


class RagTranscriptAgent(ChatAgent):
    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}

    def _graph(self, mode: str):
        if mode not in self._agents:
            self._agents[mode] = make_databricks_agent(
                index_name=os.environ.get("RAG_INDEX_NAME", "workspace.rag.rag_chunks_gte"),
                model_endpoint=os.environ.get(
                    "RAG_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
                ),
                embedding_endpoint=os.environ.get(
                    "RAG_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
                ),
                mode=mode,
            )
        return self._agents[mode]

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[Any] = None,  # noqa: UP045 — matches the ChatAgent signature
        custom_inputs: Optional[dict[str, Any]] = None,  # noqa: UP045
    ) -> ChatAgentResponse:
        requested = (custom_inputs or {}).get("mode") or DEFAULT_MODE
        # An unknown mode falls back rather than erroring: the Review App is a
        # free-text UI, and a typo should still get an answer.
        mode = requested if requested in MODES else DEFAULT_MODE
        # Tag through ask() rather than here: the trace does not exist until
        # ask() opens its root span, so tagging at this point is a no-op. The
        # tag lets the Traces tab be filtered by answer path, which is the
        # whole point of shipping three of them behind one endpoint.
        result = ask(self._graph(mode), messages[-1].content, tags={"mode": mode})
        return ChatAgentResponse(
            messages=[
                ChatAgentMessage(
                    role="assistant",
                    content=result["answer"],
                    id=str(uuid.uuid4()),
                )
            ],
            custom_outputs={
                "mode": mode,
                "requested_mode": requested,
                "chunk_keys": result["chunk_keys"],
                "video_ids": result["video_ids"],
                "decision_log": result["decision_log"],
            },
        )


mlflow.models.set_model(RagTranscriptAgent())
