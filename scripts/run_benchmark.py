"""The three-way QA-agent benchmark: Genie vs LangGraph vs Data Pilot.

Runs LOCALLY (make benchmark) because only the laptop can reach all three
contenders at once: Genie and the LangGraph serving endpoint are workspace
HTTPS APIs, Data Pilot is the data-qa-agent stack on localhost.

Per confirmed golden case (evals/golden_qa.yaml), each agent gets the same
question; answers are scored with the deterministic graders in
lib.agent_eval and written to {catalog}.{ml_schema}.agent_benchmark on the
warehouse, which the verdict dashboard reads.

Usage:
  uv run python scripts/run_benchmark.py                # all three agents
  uv run python scripts/run_benchmark.py --agents genie,langgraph
  uv run python scripts/run_benchmark.py --dry-run      # don't write Delta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lib.agent_eval import grade, load_cases  # noqa: E402

CATALOG = "workspace"
SCHEMA = "propertyiq"
ML_SCHEMA = "propertyiq_ml"
WAREHOUSE_ID = "7f9b6eb116a15acc"
GENIE_SPACE_ID = "01f197b49f0b1f7e8c55bb4744f64f47"
AGENT_ENDPOINT = "agents_workspace-propertyiq_ml-qa_agent"
# data-qa-agent's compose maps backend-api to host port 8010; its data loads
# on its own schedule, so its latest month can trail the Databricks gold.
DATA_PILOT_API = "http://localhost:8010"
RESULTS_TABLE = f"{CATALOG}.{ML_SCHEMA}.agent_benchmark"


# --------------------------------------------------------------------------
# Contender drivers — each takes a question, returns (answer, latency_s)
# --------------------------------------------------------------------------


def ask_genie(w, question: str) -> str:
    """Drive the Genie space over the Conversations API and flatten the
    reply (text attachments + query results) into one answer string."""
    conv = w.genie.start_conversation_and_wait(GENIE_SPACE_ID, question)
    message = w.genie.get_message(GENIE_SPACE_ID, conv.conversation_id, conv.message_id)
    parts: list[str] = []
    for att in message.attachments or []:
        if att.text is not None:
            parts.append(att.text.content or "")
        if att.query is not None:
            parts.append(att.query.description or "")
            result = w.genie.get_message_attachment_query_result(
                GENIE_SPACE_ID, conv.conversation_id, conv.message_id, att.attachment_id
            )
            sr = result.statement_response
            if sr and sr.manifest and sr.result and sr.result.data_array:
                cols = [c.name for c in sr.manifest.schema.columns]
                rows = sr.result.data_array[:15]
                parts.append(json.dumps([dict(zip(cols, r, strict=False)) for r in rows]))
    return "\n".join(p for p in parts if p)


def ask_langgraph(w, question: str) -> str:
    """Query the deployed agent endpoint; fall back to running the graph
    in-process (same code the endpoint serves) if the endpoint isn't ready."""
    try:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        response = w.serving_endpoints.query(
            name=AGENT_ENDPOINT,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=question)],
        )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 — endpoint cold/missing, run locally
        print(f"    endpoint unavailable ({type(exc).__name__}); running graph locally")
        from lib.qa_agent import ask, make_databricks_agent

        agent = make_databricks_agent(warehouse_id=WAREHOUSE_ID)
        return ask(agent, question)["answer"]


def ask_data_pilot(question: str) -> str:
    """Drive data-qa-agent's backend: dev login, then /ask as user1."""
    import requests

    login = requests.post(
        f"{DATA_PILOT_API}/auth/dev-login", json={"username": "user1"}, timeout=30
    )
    login.raise_for_status()
    token = login.json()["access_token"]
    reply = requests.post(
        f"{DATA_PILOT_API}/ask",
        json={"question": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    reply.raise_for_status()
    body = reply.json()
    parts = [body.get("answer", "")]
    if body.get("rows"):
        cols = body.get("columns", [])
        rows = [dict(zip(cols, r, strict=False)) if cols else r for r in body["rows"][:15]]
        parts.append(json.dumps(rows))
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def run(agents: list[str], dry_run: bool) -> list[dict]:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    cases = load_cases(REPO / "evals" / "golden_qa.yaml")
    drivers = {
        "genie": lambda q: ask_genie(w, q),
        "langgraph": lambda q: ask_langgraph(w, q),
        "data_pilot": ask_data_pilot,
    }
    run_at = datetime.now(UTC).isoformat()
    results = []
    for case in cases:
        print(f"\n== {case['case_key']} ({case['tier']})")
        for agent in agents:
            start = time.time()
            try:
                answer = drivers[agent](case["question"])
                error = ""
            except Exception as exc:  # noqa: BLE001 — a dead agent scores a fail, not a crash
                answer, error = "", f"{type(exc).__name__}: {exc}"
            latency = round(time.time() - start, 1)
            scored = grade(case, answer)
            print(
                f"  {agent:11s} {'PASS' if scored['passed'] else 'FAIL':4s} {latency:6.1f}s"
                f"  {error[:60] if error else answer[:60]!r}"
            )
            results.append(
                {
                    "run_at": run_at,
                    "case_key": case["case_key"],
                    "tier": case["tier"],
                    "question": case["question"],
                    "agent": agent,
                    "answer": answer[:4000],
                    "passed": scored["passed"],
                    "latency_s": latency,
                    "authoring_status": scored["authoring_status"],
                    "error": error[:1000],
                }
            )
    if not dry_run:
        write_results(w, results)
    return results


def _sql_str(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def write_results(w, results: list[dict]) -> None:
    ddl = f"""
        CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
          run_at STRING, case_key STRING, tier STRING, question STRING,
          agent STRING, answer STRING, passed BOOLEAN, latency_s DOUBLE,
          authoring_status STRING, error STRING
        )
        COMMENT 'QA-agent benchmark results: one row per golden case per agent
        per run. passed comes from the deterministic graders in
        src/lib/agent_eval.py; only confirmed cases count toward the verdict.'
    """
    rows = ", ".join(
        "({})".format(
            ", ".join(
                [
                    _sql_str(r["run_at"]),
                    _sql_str(r["case_key"]),
                    _sql_str(r["tier"]),
                    _sql_str(r["question"]),
                    _sql_str(r["agent"]),
                    _sql_str(r["answer"]),
                    "true" if r["passed"] else "false",
                    str(r["latency_s"]),
                    _sql_str(r["authoring_status"]),
                    _sql_str(r["error"]),
                ]
            )
        )
        for r in results
    )
    for statement in (ddl, f"INSERT INTO {RESULTS_TABLE} VALUES {rows}"):
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s"
        )
        deadline = time.time() + 300
        while result.status.state.value in ("PENDING", "RUNNING") and time.time() < deadline:
            time.sleep(5)
            result = w.statement_execution.get_statement(result.statement_id)
        assert result.status.state.value == "SUCCEEDED", result.status
    print(f"\nwrote {len(results)} rows to {RESULTS_TABLE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="genie,langgraph,data_pilot")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run([a.strip() for a in args.agents.split(",")], args.dry_run)
