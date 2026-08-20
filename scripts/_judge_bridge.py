"""Score (question, answer, contexts) records with transcript-lab's own judge.

Runs in **transcript-rag-agent's virtualenv**, not this repo's, and is invoked
as a subprocess by scripts/rag_judge_eval.py and scripts/export_dev_goldens.py.
Parity is the whole point: the same RagasJudge + DepthJudge + DEPTH_V2 rubric
that grade answers in the dev workbench grade the Databricks agent's answers
here, so a composite from one side is directly comparable with the other.
Re-implementing the metrics in this repo would fork the rubric and quietly
break that comparability — and would also drag ragas/langchain/transformers
into a dependency tree that deliberately stays databricks-sdk-only.

Input JSONL, one record per line:
  {"id": "q01|A0|agentic", "question": "...", "answer": "...",
   "contexts": ["...", ...], "answer_model": "databricks-meta-llama-3-3-70b-instruct"}

Output JSONL, same ids, one record per line:
  {"id": ..., "scores": {faithfulness, answer_relevancy, context_precision,
   insight_depth, specificity, coverage, evidence_breadth, calibration},
   "ragas_v1_composite": ..., "composite": ..., "cap_applied": ...,
   "grounding_floor_breached": ..., "judge_model": ..., "ragas_version": ...,
   "rubric_version": "depth-v2", "elapsed_seconds": ..., "error": null,
   "details": {<depth metric>: {...rationales...}}}

A record that cannot be judged (empty contexts, judge failure) comes back with
`error` set and whatever scores were salvaged — one bad answer must not sink a
whole variant's run.

Two cost controls, both deliberate normalisations rather than shortcuts:

``--max-contexts`` (default 16)
    ragas' context_precision issues ONE judge call PER context, and the
    agentic answers carry 30-70 accumulated contexts — unbounded, a single
    record costs five-plus minutes and a variant run costs hours. Every
    record on every side is judged on its first ``max_contexts`` contexts
    (the agent-ranked best), so the budget is identical for dev goldens and
    Databricks variants and composites stay comparable.
``--workers`` (default 3)
    Records are independent; each worker builds its own judge pair and they
    share nothing. Output order still follows input order.

Usage (the drivers do this for you):
    <sibling>/.venv/bin/python scripts/_judge_bridge.py \
        --source <transcript-rag-agent root> --input in.jsonl --output out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MAX_CONTEXTS = 16
DEFAULT_WORKERS = 3


def _judged_record(ragas_judge, depth_judge, rubric, record: dict, max_contexts: int) -> dict:
    """One record through both judges and the depth-v2 composite."""
    question = record["question"]
    answer = record["answer"]
    contexts = [c for c in (record.get("contexts") or []) if c and str(c).strip()]
    contexts = contexts[:max_contexts]
    out: dict = {"id": record.get("id"), "rubric_version": rubric.version}
    if not answer or not answer.strip():
        out["error"] = "empty answer; not judged"
        return out
    if not contexts:
        out["error"] = "no retrieval contexts; not judged"
        return out

    started = time.monotonic()
    evaluation = ragas_judge.score(
        question, answer, contexts, answer_model=record.get("answer_model")
    )
    scores: dict = dict(evaluation.get("scores") or {})

    depth_error = None
    details: dict = {}
    try:
        for name, breakdown in depth_judge.score(question, answer, contexts).items():
            scores[name] = round(float(breakdown.score), 4)
            if breakdown.details is not None:
                details[name] = breakdown.details
    except Exception as exc:  # noqa: BLE001 — a failed depth call still leaves grounding
        depth_error = f"depth: {exc}"

    composited = rubric.composite(scores)
    errors = [e for e in (evaluation.get("error"), depth_error) if e]
    out.update(
        {
            "scores": scores,
            "ragas_v1_composite": evaluation.get("composite"),
            "composite": composited.composite,
            "cap_applied": composited.cap_applied,
            "grounding_floor_breached": composited.grounding_floor_breached,
            "judge_model": evaluation.get("judge_model"),
            "ragas_version": evaluation.get("ragas_version"),
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "details": details or None,
            "error": "; ".join(errors) if errors else None,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="transcript-rag-agent checkout root")
    parser.add_argument("--input", required=True, help="records JSONL to judge")
    parser.add_argument("--output", required=True, help="scored JSONL to write")
    parser.add_argument("--max-contexts", type=int, default=DEFAULT_MAX_CONTEXTS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    source = Path(args.source).resolve()
    # The dev project's imports, .env discovery and model caches are all
    # relative to its root; run as if launched from there.
    sys.path.insert(0, str(source))
    os.chdir(source)

    from src.config import load_settings
    from src.evals.judge import DEPTH_V2, DepthJudge, RagasJudge

    settings = load_settings()
    local = threading.local()

    def judges() -> tuple:
        # One judge pair per worker thread: the langchain/ragas objects are
        # not documented thread-safe, and building them is cheap next to the
        # judge calls themselves.
        if not hasattr(local, "pair"):
            local.pair = (RagasJudge.from_settings(settings), DepthJudge.from_settings(settings))
        return local.pair

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Resumable by construction: records already in the output file are kept
    # and skipped, and each new verdict is appended (under a lock) the moment
    # it lands — a killed run costs only its in-flight records, which matters
    # in a harness that reaps long-running processes.
    output_path = Path(args.output)
    already: dict[str, dict] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                verdict = json.loads(line)
            except json.JSONDecodeError:
                # A truncated trailing line from a SIGKILL mid-write — the
                # write it belongs to never completed, so drop it and let the
                # record be scored fresh rather than aborting the resume.
                print(f"  skipping truncated output line: {line[:80]!r}", file=sys.stderr)
                continue
            if verdict.get("id") and not verdict.get("error"):
                already[str(verdict["id"])] = verdict
    todo = [record for record in records if str(record.get("id")) not in already]
    print(
        f"judging {len(todo)}/{len(records)} record(s) ({len(already)} already scored), "
        f"{args.workers} worker(s), context budget {args.max_contexts}",
        file=sys.stderr,
    )

    done = 0
    lock = threading.Lock()
    handle = output_path.open("a", encoding="utf-8")

    def score_one(record: dict) -> None:
        nonlocal done
        ragas_judge, depth_judge = judges()
        try:
            scored = _judged_record(ragas_judge, depth_judge, DEPTH_V2, record, args.max_contexts)
        except Exception as exc:  # noqa: BLE001 — carry on; one failure is one row
            scored = {"id": record.get("id"), "error": str(exc)}
        with lock:
            done += 1
            handle.write(json.dumps(scored) + "\n")
            handle.flush()
            print(
                f"  [{done}/{len(todo)}] {scored.get('id')} -> {scored.get('composite')}",
                file=sys.stderr,
            )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(score_one, todo))
    handle.close()


if __name__ == "__main__":
    main()
