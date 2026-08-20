"""Capture transcript-lab's agentic answers as the judged golden set.

Phase 1 of the s09 plan: every question in `lib.rag_questions` is answered by
the dev workbench's `rag_agent` setup (the LangGraph ReAct research loop the
Databricks agent is being optimised toward), judged with the dev project's own
RAGAS+depth judge via scripts/_judge_bridge.py, and landed as the
`golden_answers` entity through the same content-hashed landing contract as
every other export — so `rag_ingest` carries it to `silver_golden_answers`
with zero new pipeline machinery.

The workbench, not the agent class, is deliberately the capture surface:
`POST /api/ask` exercises the exact code path the user drives from the chat
UI, and its history persists answer, contexts, references, per-iteration
trace and iteration counts — everything a golden needs — so a re-run reuses
prior answers instead of re-spending DeepSeek calls (`--force-ask` re-asks).

Usage:
  uv run python scripts/export_dev_goldens.py              # ask missing, judge, land
  uv run python scripts/export_dev_goldens.py --dry-run    # build + hash, upload nothing
  uv run python scripts/export_dev_goldens.py --force-ask  # re-ask every question
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lib.rag_export import (  # noqa: E402
    ENTITY_KEYS,
    frame_sha,
    golden_answer_rows,
    landing_name,
    parse_landed,
)
from lib.rag_questions import QUESTIONS  # noqa: E402

CATALOG = "workspace"
SCHEMA = "rag"
VOLUME = "landing"
ENTITY = "golden_answers"
SETUP_KEY = "rag_agent"
DEFAULT_BASE_URL = "http://localhost:8011"
DEFAULT_SOURCE = REPO.parent / "transcript-rag-agent"

# One agentic run streams for minutes; the read timeout only bounds the gap
# between SSE events, not the whole answer.
ASK_TIMEOUT_SECONDS = (10, 300)


def _health(base_url: str) -> dict[str, Any]:
    response = requests.get(f"{base_url}/api/health", timeout=10)
    response.raise_for_status()
    return response.json()


def _history(base_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{base_url}/api/history", timeout=30)
    response.raise_for_status()
    return response.json().get("conversations") or []


def _agent_answer(entry: dict[str, Any]) -> dict[str, Any] | None:
    for answer in entry.get("answers") or []:
        if answer.get("key") == SETUP_KEY and not answer.get("error"):
            return answer
    return None


def _find_entry(history: list[dict[str, Any]], question: str) -> dict[str, Any] | None:
    """Newest history entry answering this exact question with the agent setup."""
    matches = [
        entry
        for entry in history
        if (entry.get("question") or "").strip() == question.strip() and _agent_answer(entry)
    ]
    if not matches:
        return None
    return max(matches, key=lambda entry: entry.get("asked_at") or "")


def _ask(base_url: str, question: str) -> None:
    """Run the agentic setup once, draining the SSE stream until it closes."""
    response = requests.post(
        f"{base_url}/api/ask",
        json={"question": question, "setups": [SETUP_KEY]},
        stream=True,
        timeout=ASK_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    for line in response.iter_lines():
        # The stream is the workbench's progress feed; landing in history is
        # what matters here, so events are drained rather than parsed.
        if line and line.startswith(b"event: error"):
            print("    (stream reported an error event; will validate via history)")
    response.close()


def _queries_from_trace(answer: dict[str, Any]) -> list[str]:
    queries = []
    for step in answer.get("trace") or []:
        if step.get("phase") == "retrieve" and step.get("query"):
            queries.append(str(step["query"]))
    return queries


def collect_answers(base_url: str, force_ask: bool) -> list[dict[str, Any]]:
    """Every question answered by the dev agent, asking only where needed."""
    history = _history(base_url)
    records: list[dict[str, Any]] = []
    for spec in QUESTIONS:
        question = str(spec["question"])
        entry = None if force_ask else _find_entry(history, question)
        if entry is None:
            print(f"  {spec['question_id']}: asking dev {SETUP_KEY} ...")
            _ask(base_url, question)
            history = _history(base_url)
            entry = _find_entry(history, question)
        if entry is None:
            print(f"  {spec['question_id']}: NO usable answer after ask — skipping")
            continue
        answer = _agent_answer(entry)
        assert answer is not None  # _find_entry guarantees it
        records.append(
            {
                **spec,
                "answer": answer.get("answer") or "",
                "answer_model": answer.get("model") or "",
                "contexts": answer.get("contexts") or [],
                "references": answer.get("references") or [],
                "chunk_ids": answer.get("retrieved_chunk_ids") or [],
                "queries": _queries_from_trace(answer),
                "iterations": answer.get("iterations"),
                "llm_calls": answer.get("llm_calls"),
                "terminated_reason": answer.get("terminated_reason") or "",
                "captured_at": entry.get("asked_at") or datetime.now(UTC).isoformat(),
            }
        )
        print(
            f"  {spec['question_id']}: {answer.get('iterations') or '?'} iteration(s), "
            f"{len(answer.get('contexts') or [])} context(s)"
        )
    return records


def judge_records(records: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    """Score every captured answer through the dev judge bridge."""
    venv_python = source / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(f"{venv_python} not found — run `uv sync` in {source} first")

    work = Path(tempfile.mkdtemp(prefix="golden_judge_"))
    request_path = work / "records.jsonl"
    scored_path = work / "scored.jsonl"
    with request_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "id": record["question_id"],
                        "question": record["question"],
                        "answer": record["answer"],
                        "contexts": record["contexts"],
                        "answer_model": record["answer_model"],
                    }
                )
                + "\n"
            )

    result = subprocess.run(  # noqa: S603 — fixed argv, paths validated above
        [
            str(venv_python),
            str(REPO / "scripts" / "_judge_bridge.py"),
            "--source",
            str(source),
            "--input",
            str(request_path),
            "--output",
            str(scored_path),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"judge bridge exited {result.returncode}")

    scored = {
        record["id"]: record
        for record in (
            json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()
        )
    }
    for record in records:
        verdict = scored.get(record["question_id"]) or {}
        record.update(
            {
                "scores": verdict.get("scores") or {},
                "ragas_v1_composite": verdict.get("ragas_v1_composite"),
                "composite": verdict.get("composite"),
                "cap_applied": verdict.get("cap_applied"),
                "grounding_floor_breached": verdict.get("grounding_floor_breached"),
                "judge_model": verdict.get("judge_model"),
                "rubric_version": verdict.get("rubric_version"),
                "judge_error": verdict.get("error"),
            }
        )
    return records


def upload(frame, dry_run: bool) -> None:
    """Land the frame under the entity contract (same shape as rag_export.run)."""
    sha = frame_sha(frame, ENTITY_KEYS[ENTITY])
    if dry_run:
        print(f"\n{ENTITY}: {len(frame)} row(s), sha {sha} — dry run, nothing uploaded")
        return

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound

    w = WorkspaceClient()
    directory = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{ENTITY}"
    existing: set[str] = set()
    try:
        for item in w.files.list_directory_contents(directory):
            parsed = parse_landed(Path(item.path).name)
            if parsed and parsed[0] == ENTITY:
                existing.add(parsed[1])
    except NotFound:
        pass

    if sha in existing:
        print(f"\n{ENTITY}: sha {sha} already landed — nothing to do")
        return

    name = landing_name(ENTITY, sha)
    local = Path(tempfile.mkdtemp(prefix="golden_land_")) / name
    frame.to_parquet(local, index=False)
    with local.open("rb") as handle:
        w.files.upload(f"{directory}/{name}", handle, overwrite=True)
    print(f"\n{ENTITY}: landed {name} ({len(frame)} rows); rag_ingest will pick it up")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("RAG_DEV_URL", DEFAULT_BASE_URL))
    parser.add_argument("--source", help="transcript-rag-agent checkout (or RAG_SOURCE_DIR)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-ask", action="store_true", help="re-ask every question")
    parser.add_argument("--skip-judge", action="store_true", help="capture answers only")
    args = parser.parse_args()

    source = Path(args.source or os.environ.get("RAG_SOURCE_DIR") or DEFAULT_SOURCE).resolve()
    health = _health(args.base_url)
    print(
        f"dev workbench: {args.base_url} — answer model {health.get('answer_model')}, "
        f"judge model {health.get('judge_model')}"
    )

    records = collect_answers(args.base_url, force_ask=args.force_ask)
    if not records:
        raise SystemExit("no answers captured — is the corpus indexed?")
    if not args.skip_judge:
        print(f"\njudging {len(records)} answer(s) with the dev judge ...")
        records = judge_records(records, source)

    frame = golden_answer_rows(records)
    upload(frame, dry_run=args.dry_run)

    judged = [r for r in records if r.get("composite") is not None]
    if judged:
        print("\nquestion      cat     iters  composite")
        for record in records:
            comp = record.get("composite")
            print(
                f"{record['question_id']:<12} {record['category']:<7} "
                f"{record.get('iterations') or 0:>5}  "
                f"{comp if comp is not None else '—'}"
            )


if __name__ == "__main__":
    main()
