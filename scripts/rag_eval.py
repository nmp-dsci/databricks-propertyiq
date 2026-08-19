"""Three-way retrieval benchmark: Chroma vs Vector Search (gte) vs Vector Search (MiniLM).

Runs LOCALLY (`make rag-eval`) for the same reason the s05 QA benchmark does:
only the laptop can reach every contender at once — Chroma is a file on disk,
the indexes are workspace APIs.

The three contenders are chosen so the comparison isolates one variable at a
time:

  chroma       transcript-lab's own index. MiniLM vectors, HNSW, local.
  vs_minilm    the SAME MiniLM vectors, in Vector Search. Difference vs chroma
               is the *engine* alone.
  vs_gte       Databricks-managed gte-large-en embeddings in Vector Search.
               Difference vs vs_minilm is the *embedding model* alone.

Scored with lib.ir_metrics against transcript-lab's own 20-question golden set.
Video ids carry the verdict because they are stable; chunk anchors are
re-anchored by content hash where possible and reported as unresolved where not
(the corpus grew ~6x past the point the golden set was verified).

Usage:
  uv run python scripts/rag_eval.py
  uv run python scripts/rag_eval.py --dry-run          # don't write Delta
  uv run python scripts/rag_eval.py --contenders vs_gte,vs_minilm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lib.ir_metrics import aggregate, reanchor_chunk_ids, score_case  # noqa: E402

CATALOG = "workspace"
SCHEMA = "rag"
WAREHOUSE_ID = "7f9b6eb116a15acc"
GTE_INDEX = f"{CATALOG}.{SCHEMA}.rag_chunks_gte"
MINILM_INDEX = f"{CATALOG}.{SCHEMA}.rag_chunks_minilm"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.rag_eval"
DEFAULT_SOURCE = REPO.parent / "transcript-rag-agent"
TOP_K = 10

COLUMNS = ["chunk_key", "video_id", "title", "source_url", "start_seconds", "text"]


def load_golden(source: Path) -> list[dict]:
    payload = json.loads((source / "src" / "evals" / "golden_dataset.json").read_text())
    return payload.get("entries") or []


def chroma_side(source: Path, questions: list[str]) -> list[dict]:
    """Chroma's own hits plus the MiniLM query vectors, in one subprocess."""
    venv_python = source / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(f"{venv_python} not found; run `uv sync` in transcript-rag-agent")
    result = subprocess.run(  # noqa: S603 — fixed argv, path validated above
        [
            str(venv_python),
            str(REPO / "scripts" / "_chroma_query.py"),
            "--chroma-path",
            str(source / ".yt-agent" / "chroma"),
            "--top-k",
            str(TOP_K),
        ],
        input=json.dumps(questions),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"chroma query failed:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout)


def query_index(w, index_name: str, vector: list[float]) -> list[dict]:
    response = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=COLUMNS,
        query_vector=vector,
        num_results=TOP_K,
    )
    rows = (response.result.data_array if response.result else None) or []
    return [dict(zip(COLUMNS + ["score"], row, strict=False)) for row in rows]


def anchor_maps(w) -> tuple[dict[str, str], dict[str, str]]:
    """Maps for re-anchoring stale golden chunk ids onto today's corpus.

    `chunk_id` is transcript-lab's own id as it was at export time, so a golden
    id that still exists resolves through its text hash to whatever chunk_key
    now holds that text.
    """
    import time as _time

    statement = f"""
        SELECT chunk_id, text_sha, chunk_key
        FROM {CATALOG}.{SCHEMA}.silver_chunks WHERE is_current
    """
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, wait_timeout="50s"
    )
    deadline = _time.time() + 300
    while result.status.state.value in ("PENDING", "RUNNING") and _time.time() < deadline:
        _time.sleep(5)
        result = w.statement_execution.get_statement(result.statement_id)
    rows = (result.result.data_array if result.result else None) or []
    sha_by_chunk_id = {row[0]: row[1] for row in rows}
    key_by_sha = {row[1]: row[2] for row in rows}
    return sha_by_chunk_id, key_by_sha


def run(contenders: list[str], dry_run: bool, source: Path) -> list[dict]:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    golden = load_golden(source)
    questions = [entry["question"] for entry in golden]
    print(f"golden questions: {len(questions)}")

    print("querying local Chroma (and embedding the questions with MiniLM)...")
    local = chroma_side(source, questions)

    sha_by_chunk_id, key_by_sha = anchor_maps(w)
    run_at = datetime.now(UTC).isoformat()
    results: list[dict] = []

    for entry, local_row in zip(golden, local, strict=True):
        expected_videos = entry.get("expected_video_ids") or []
        resolved, unresolved = reanchor_chunk_ids(
            entry.get("expected_chunk_ids") or [], sha_by_chunk_id, key_by_sha
        )
        print(f"\n== {entry['id']} ({entry.get('domain')})")
        if unresolved:
            print(f"   {len(resolved)}/{len(resolved) + len(unresolved)} chunk anchors resolved")

        for contender in contenders:
            start = time.time()
            try:
                if contender == "chroma":
                    hits = local_row["hits"]
                elif contender == "vs_minilm":
                    hits = query_index(w, MINILM_INDEX, local_row["embedding"])
                elif contender == "vs_gte":
                    vector = (
                        w.serving_endpoints.query(
                            name=EMBEDDING_ENDPOINT, input=[entry["question"]]
                        )
                        .data[0]
                        .embedding
                    )
                    hits = query_index(w, GTE_INDEX, vector)
                else:
                    raise ValueError(f"unknown contender {contender}")
                error = ""
            except Exception as exc:  # noqa: BLE001 — a broken contender scores 0, not a crash
                hits, error = [], f"{type(exc).__name__}: {exc}"
            latency = round(time.time() - start, 2)

            retrieved_videos = list(dict.fromkeys(hit.get("video_id") for hit in hits))
            retrieved_keys = [hit.get("chunk_key") for hit in hits]
            video_scores = score_case(retrieved_videos, expected_videos, k=TOP_K)
            chunk_scores = score_case(retrieved_keys, resolved, k=TOP_K) if resolved else {}

            print(
                f"   {contender:<10} recall@{TOP_K}={video_scores[f'recall_at_{TOP_K}']:.2f} "
                f"mrr={video_scores['mrr']:.2f} ndcg={video_scores[f'ndcg_at_{TOP_K}']:.2f} "
                f"{latency:>5.2f}s {error[:40]}"
            )
            results.append(
                {
                    "run_at": run_at,
                    "case_id": entry["id"],
                    "domain": entry.get("domain") or "",
                    "question_type": entry.get("question_type") or "local",
                    "contender": contender,
                    "video_recall": video_scores[f"recall_at_{TOP_K}"],
                    "video_mrr": video_scores["mrr"],
                    "video_ndcg": video_scores[f"ndcg_at_{TOP_K}"],
                    "chunk_recall": chunk_scores.get(f"recall_at_{TOP_K}"),
                    "chunk_anchors_resolved": len(resolved),
                    "chunk_anchors_unresolved": len(unresolved),
                    "latency_s": latency,
                    "retrieved_videos": ",".join(v for v in retrieved_videos if v),
                    "error": error[:500],
                }
            )

    print("\n" + "=" * 62)
    print(f"{'contender':<12} {'recall@10':>10} {'MRR':>8} {'NDCG@10':>9} {'latency':>9}")
    for contender in contenders:
        rows = [r for r in results if r["contender"] == contender]
        means = aggregate(
            [
                {"recall": r["video_recall"], "mrr": r["video_mrr"], "ndcg": r["video_ndcg"]}
                for r in rows
            ]
        )
        latency = sum(r["latency_s"] for r in rows) / len(rows) if rows else 0
        print(
            f"{contender:<12} {means.get('recall', 0):>10.3f} {means.get('mrr', 0):>8.3f} "
            f"{means.get('ndcg', 0):>9.3f} {latency:>8.2f}s"
        )

    if not dry_run:
        write_results(w, results)
    return results


def write_results(w, results: list[dict]) -> None:
    from databricks.sdk.service.sql import StatementParameterListItem as Param

    _execute(
        w,
        f"""
        CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
          run_at STRING, case_id STRING, domain STRING, question_type STRING,
          contender STRING, video_recall DOUBLE, video_mrr DOUBLE, video_ndcg DOUBLE,
          chunk_recall DOUBLE, chunk_anchors_resolved INT, chunk_anchors_unresolved INT,
          latency_s DOUBLE, retrieved_videos STRING, error STRING
        )
        COMMENT 'Retrieval benchmark: transcript-lab Chroma vs the two Vector
        Search indexes, scored with src/lib/ir_metrics.py against the 20-question
        golden set. Video-level metrics carry the verdict; chunk-level is
        reported only where stale anchors re-resolved by content hash.'
        """,
    )
    insert = f"""
        INSERT INTO {RESULTS_TABLE} VALUES
        (:run_at, :case_id, :domain, :question_type, :contender, :video_recall,
         :video_mrr, :video_ndcg, :chunk_recall, :chunk_anchors_resolved,
         :chunk_anchors_unresolved, :latency_s, :retrieved_videos, :error)
    """
    for row in results:
        _execute(
            w,
            insert,
            [
                Param(name="run_at", value=row["run_at"]),
                Param(name="case_id", value=row["case_id"]),
                Param(name="domain", value=row["domain"]),
                Param(name="question_type", value=row["question_type"]),
                Param(name="contender", value=row["contender"]),
                Param(name="video_recall", value=str(row["video_recall"]), type="DOUBLE"),
                Param(name="video_mrr", value=str(row["video_mrr"]), type="DOUBLE"),
                Param(name="video_ndcg", value=str(row["video_ndcg"]), type="DOUBLE"),
                Param(
                    name="chunk_recall",
                    value=None if row["chunk_recall"] is None else str(row["chunk_recall"]),
                    type="DOUBLE",
                ),
                Param(
                    name="chunk_anchors_resolved",
                    value=str(row["chunk_anchors_resolved"]),
                    type="INT",
                ),
                Param(
                    name="chunk_anchors_unresolved",
                    value=str(row["chunk_anchors_unresolved"]),
                    type="INT",
                ),
                Param(name="latency_s", value=str(row["latency_s"]), type="DOUBLE"),
                Param(name="retrieved_videos", value=row["retrieved_videos"]),
                Param(name="error", value=row["error"]),
            ],
        )
    print(f"\nwrote {len(results)} rows to {RESULTS_TABLE}")


def _execute(w, statement: str, parameters: list | None = None) -> None:
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, parameters=parameters, wait_timeout="50s"
    )
    deadline = time.time() + 300
    while result.status.state.value in ("PENDING", "RUNNING") and time.time() < deadline:
        time.sleep(5)
        result = w.statement_execution.get_statement(result.statement_id)
    if result.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"statement {result.status.state.value}: {result.status.error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--contenders", default="chroma,vs_minilm,vs_gte")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = parser.parse_args()
    run(
        [c.strip() for c in args.contenders.split(",")],
        args.dry_run,
        Path(args.source).expanduser().resolve(),
    )
