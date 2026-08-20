"""Pure transforms behind the transcript-lab -> lakehouse exporter.

`scripts/rag_export.py` is the driver (subprocess, filesystem, upload); this
module holds everything worth unit-testing: the landing-file contract, the
content hash that makes re-exports free, and the parsers that turn
transcript-lab's stores into flat rows.

Two shapes need decoding before they can land as Delta:

* `raw_transcripts` documents are a packed JSON blob rather than transcript
  text — Chroma's document body carries `{segments, summary_embedding,
  description, tags, transcript_languages}` (see transcript-rag-agent's
  `src/rag/storage.py::_raw_document_body`). Segments are the interesting part
  and become their own table.
* the graph cache is one JSON file per chunk keyed by
  `sha256(chunk_id + "\\n" + text)`, each holding entities/relations/claims that
  explode into three tables.

The landing contract is deliberately the same one `propertyiq_getdata`
publishes for the property feed: `<entity>_<sha8>.parquet`, append-only, newest
file per entity wins (`transforms.resolve_versions` reads it the same way).
Each export is a full snapshot of an entity, so the hash is over the whole
frame — identical content means no file is written at all, which is what makes
`make rag-export` safe to run whenever.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

# Every entity the exporter can land, with the natural key silver MERGEs on.
# Order matters only for readable logs.
ENTITY_KEYS: dict[str, tuple[str, ...]] = {
    "chunks": ("video_id", "chunk_index"),
    "chunks_contextual": ("video_id", "chunk_index"),
    "transcripts": ("video_id",),
    "segments": ("video_id", "segment_index"),
    "summaries": ("video_id",),
    "golden_qa": ("case_id",),
    "golden_answers": ("question_id",),
    "eval_runs": ("run_id",),
    "graph_entities": ("chunk_sha", "name"),
    "graph_relations": ("chunk_sha", "source", "target", "relation_type"),
    "graph_claims": ("chunk_sha", "claim_index"),
    "themes": ("theme_id",),
}


def _stable_repr(value: Any) -> str:
    """Deterministic text for one cell, including embedding vectors."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(_stable_repr(item) for item in value)
    if hasattr(value, "tolist"):  # numpy array / scalar
        return _stable_repr(value.tolist())
    if isinstance(value, (dict, set)):
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return str(value)


def frame_sha(frame: pd.DataFrame, key_columns: Sequence[str] = ()) -> str:
    """8-hex content hash of a frame, stable across runs and row order.

    Sorting by the natural key first means a re-export that happens to read
    rows in a different order still hashes the same, so an unchanged corpus
    lands nothing.
    """
    if frame.empty:
        return "empty000"
    usable = [column for column in key_columns if column in frame.columns]
    ordered = frame.sort_values(usable).reset_index(drop=True) if usable else frame
    hasher = hashlib.sha256()
    for column in sorted(ordered.columns):
        hasher.update(f"\x1e{column}\x1e".encode())
        hasher.update("\x1f".join(ordered[column].map(_stable_repr)).encode())
    return hasher.hexdigest()[:8]


def landing_name(entity: str, sha: str) -> str:
    """`<entity>_<sha8>.parquet` — the same contract the property feed uses."""
    return f"{entity}_{sha}.parquet"


def parse_landed(filename: str) -> tuple[str, str] | None:
    """Split a landed filename back into (entity, sha), or None if off-contract."""
    stem = filename.removesuffix(".parquet")
    if stem == filename or "_" not in stem:
        return None
    entity, _, sha = stem.rpartition("_")
    if not entity or len(sha) != 8:
        return None
    return entity, sha


# ---------------------------------------------------------------------------
# Chroma collection parsers
# ---------------------------------------------------------------------------


def chunk_rows(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """`transcript_chunks` -> the load-bearing table behind the index.

    `embedding_text` is rebuilt exactly as transcript-lab embeds it
    (`context_header + "\\n" + text`) so a Databricks-managed embedding sees the
    same input the local MiniLM one did — otherwise the retrieval comparison
    would be measuring two different texts.
    """
    rows = []
    for record in records:
        meta = record.get("metadata") or {}
        text = record.get("document") or ""
        header = meta.get("context_header") or ""
        rows.append(
            {
                # chunk_id is transcript-lab's own id and is regenerated on every
                # local re-index; chunk_key is ours and is stable, which is what
                # Vector Search addresses rows by.
                "chunk_key": f"{meta.get('video_id')}:{_as_int(meta.get('chunk_index'))}",
                "chunk_id": record.get("id"),
                "video_id": meta.get("video_id"),
                "chunk_index": _as_int(meta.get("chunk_index")),
                "text": text,
                "context_header": header,
                "embedding_text": f"{header}\n{text}" if header else text,
                "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                "transcript_id": meta.get("transcript_id"),
                "source_url": meta.get("source_url"),
                "title": meta.get("title"),
                "channel_id": meta.get("channel_id"),
                "channel_name": meta.get("channel_name"),
                "upload_date": meta.get("upload_date"),
                "start_seconds": _as_float(meta.get("start_seconds")),
                "end_seconds": _as_float(meta.get("end_seconds")),
                "start_segment_index": _as_int(meta.get("start_segment_index")),
                "end_segment_index": _as_int(meta.get("end_segment_index")),
                "segment_count": _as_int(meta.get("segment_count")),
                "embedding": _as_vector(record.get("embedding")),
            }
        )
    return pd.DataFrame(rows, columns=_CHUNK_COLUMNS)


_CHUNK_COLUMNS = [
    "chunk_key",
    "chunk_id",
    "video_id",
    "chunk_index",
    "text",
    "context_header",
    "embedding_text",
    "text_sha",
    "transcript_id",
    "source_url",
    "title",
    "channel_id",
    "channel_name",
    "upload_date",
    "start_seconds",
    "end_seconds",
    "start_segment_index",
    "end_segment_index",
    "segment_count",
    "embedding",
]


def transcript_rows(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """`raw_transcripts` -> one row per video, blob unpacked into columns."""
    rows = []
    for record in records:
        meta = record.get("metadata") or {}
        body = _load_json(record.get("document"))
        rows.append(
            {
                "video_id": meta.get("video_id"),
                "transcript_id": meta.get("transcript_id"),
                "title": meta.get("title"),
                "channel_id": meta.get("channel_id"),
                "channel_name": meta.get("channel_name"),
                "source_url": meta.get("source_url"),
                "provider": meta.get("provider"),
                "language": meta.get("language"),
                "upload_date": meta.get("upload_date"),
                "duration_seconds": _as_float(meta.get("duration_seconds")),
                "view_count": _as_int(meta.get("view_count")),
                "like_count": _as_int(meta.get("like_count")),
                "thumbnail_url": meta.get("thumbnail_url"),
                "segment_count": _as_int(meta.get("segment_count")),
                "description": body.get("description") or meta.get("description"),
                "tags": _as_str_list(body.get("tags")),
                "transcript_languages": _as_str_list(body.get("transcript_languages")),
                "summary": meta.get("summary"),
                "summary_model": meta.get("summary_model"),
                "summary_generated_at": meta.get("summary_generated_at"),
                "fetched_at": meta.get("fetched_at"),
            }
        )
    return pd.DataFrame(rows)


def segment_rows(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Explode each raw transcript's packed `segments[]` into its own table."""
    rows = []
    for record in records:
        meta = record.get("metadata") or {}
        video_id = meta.get("video_id")
        body = _load_json(record.get("document"))
        for index, segment in enumerate(body.get("segments") or []):
            if not isinstance(segment, dict):
                continue
            rows.append(
                {
                    "video_id": video_id,
                    "segment_index": index,
                    "text": segment.get("text"),
                    "offset_ms": _as_int(segment.get("offset_ms")),
                    "duration_ms": _as_int(segment.get("duration_ms")),
                    "start_seconds": _as_float(segment.get("start_seconds")),
                    "end_seconds": _as_float(segment.get("end_seconds")),
                    "language": segment.get("language"),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "video_id",
            "segment_index",
            "text",
            "offset_ms",
            "duration_ms",
            "start_seconds",
            "end_seconds",
            "language",
        ],
    )


def summary_rows(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """`transcript_summaries` -> one row per video."""
    rows = []
    for record in records:
        meta = record.get("metadata") or {}
        rows.append(
            {
                "video_id": meta.get("video_id"),
                "transcript_id": meta.get("transcript_id"),
                "title": meta.get("title"),
                "source_url": meta.get("source_url"),
                "summary": record.get("document"),
                "summary_model": meta.get("summary_model"),
                "summary_generated_at": meta.get("summary_generated_at"),
                "summary_embedding_model": meta.get("summary_embedding_model"),
                "language": meta.get("language"),
                "segment_count": _as_int(meta.get("segment_count")),
                "chunk_count": _as_int(meta.get("chunk_count")),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Committed JSON artifacts
# ---------------------------------------------------------------------------


def golden_rows(payload: dict[str, Any]) -> pd.DataFrame:
    """transcript-lab's golden set -> `eval_golden_qa`.

    `expected_chunk_ids` is carried but NOT trusted: it was verified against a
    23-video corpus that has since grown past 100, and chunk ids are recreated
    on every local re-index. Phase 5 re-anchors chunk-level hits by text hash;
    `expected_video_ids` stays authoritative in the meantime.
    """
    corpus = payload.get("corpus") or {}
    rows = []
    for entry in payload.get("entries") or []:
        rows.append(
            {
                "case_id": entry.get("id"),
                "question": entry.get("question"),
                "reference_answer": entry.get("reference_answer"),
                "expected_video_ids": _as_str_list(entry.get("expected_video_ids")),
                "expected_chunk_ids": _as_str_list(entry.get("expected_chunk_ids")),
                "domain": entry.get("domain"),
                "question_type": entry.get("question_type") or "local",
                "notes": entry.get("notes") or "",
                "corpus_videos": _as_int(corpus.get("videos")),
                "corpus_chunks": _as_int(corpus.get("chunks")),
                "corpus_verified_at": corpus.get("verified_at"),
                "anchor_status": "stale_pending_reanchor",
            }
        )
    return pd.DataFrame(rows)


def golden_answer_rows(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Judged dev-workbench answers -> `golden_answers` (plan s09, phase 1).

    Each record is one question answered by transcript-lab's agentic
    `rag_agent` and scored by its own judge (via scripts/_judge_bridge.py).
    These rows are the optimisation target: `composite` is the bar a
    Databricks variant has to meet, `iterations` the research behaviour it
    should reproduce on broad questions.

    Lists ride as JSON strings (the `payload_json` convention above): the
    consumers are the judge harness and a dashboard, neither of which needs
    them exploded, and a flat schema keeps Auto Loader inference boring.
    """
    rows = []
    for record in records:
        scores = record.get("scores") or {}
        rows.append(
            {
                "question_id": record.get("question_id"),
                "category": record.get("category"),
                "domain": record.get("domain"),
                "question": record.get("question"),
                "source_case_id": record.get("source_case_id") or "",
                "answer_md": record.get("answer") or "",
                "answer_model": record.get("answer_model") or "",
                "contexts_json": json.dumps(record.get("contexts") or [], ensure_ascii=False),
                "references_json": json.dumps(record.get("references") or [], ensure_ascii=False),
                "chunk_ids_json": json.dumps(record.get("chunk_ids") or [], ensure_ascii=False),
                "queries_json": json.dumps(record.get("queries") or [], ensure_ascii=False),
                "iterations": _as_int(record.get("iterations")) or 0,
                "llm_calls": _as_int(record.get("llm_calls")) or 0,
                "terminated_reason": record.get("terminated_reason") or "",
                "faithfulness": _as_float(scores.get("faithfulness")),
                "answer_relevancy": _as_float(scores.get("answer_relevancy")),
                "context_precision": _as_float(scores.get("context_precision")),
                "insight_depth": _as_float(scores.get("insight_depth")),
                "specificity": _as_float(scores.get("specificity")),
                "coverage": _as_float(scores.get("coverage")),
                "evidence_breadth": _as_float(scores.get("evidence_breadth")),
                "calibration": _as_float(scores.get("calibration")),
                "ragas_v1_composite": _as_float(record.get("ragas_v1_composite")),
                "composite": _as_float(record.get("composite")),
                "cap_applied": bool(record.get("cap_applied")),
                "grounding_floor_breached": bool(record.get("grounding_floor_breached")),
                "judge_model": record.get("judge_model") or "",
                "rubric_version": record.get("rubric_version") or "",
                "judge_error": record.get("judge_error") or "",
                "captured_at": record.get("captured_at") or "",
                "source": "transcript-lab/rag_agent",
            }
        )
    return pd.DataFrame(rows)


def eval_run_rows(runs: Iterable[tuple[str, dict[str, Any]]]) -> pd.DataFrame:
    """Committed eval snapshots -> `eval_runs`.

    Four different run shapes (ablation / eval / matrix / critique) share only a
    handful of fields, so the varying body is kept as a JSON string rather than
    forced into one flat schema — this table is provenance, not analytics.
    """
    rows = []
    for filename, payload in runs:
        rows.append(
            {
                "run_id": payload.get("run_id") or filename,
                "source_file": filename,
                "kind": payload.get("kind") or _kind_from_name(filename),
                "created_at": payload.get("created_at"),
                "entry_count": _as_int(payload.get("entry_count") or payload.get("entries")),
                "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows)


def theme_rows(payload: Any) -> pd.DataFrame:
    """`themes.json` -> `graph_themes` (cross-video topic clusters)."""
    themes = payload.get("themes") if isinstance(payload, dict) else payload
    rows = []
    for theme in themes or []:
        if not isinstance(theme, dict):
            continue
        rows.append(
            {
                "theme_id": theme.get("theme_id"),
                "title": theme.get("title"),
                "summary": theme.get("summary"),
                "domain": theme.get("domain"),
                "member_count": _as_int(theme.get("member_count")),
                "video_count": _as_int(theme.get("video_count")),
                "channel_count": _as_int(theme.get("channel_count")),
                "cross_video": bool(theme.get("cross_video")),
                "videos": _as_str_list(theme.get("videos")),
            }
        )
    return pd.DataFrame(rows)


def graph_rows(
    extractions: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Graph cache -> (entities, relations, claims).

    Reading the on-disk cache instead of Neo4j is deliberate: it is the same
    extraction output, needs no running database, and keys every row by the
    chunk hash so a re-extraction of one chunk only rewrites its own rows.
    """
    entities, relations, claims = [], [], []
    for chunk_sha, payload in extractions:
        for entity in payload.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            entities.append(
                {
                    "chunk_sha": chunk_sha,
                    "name": entity.get("name"),
                    "entity_type": entity.get("type"),
                    "aliases": _as_str_list(entity.get("aliases")),
                }
            )
        for relation in payload.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            relations.append(
                {
                    "chunk_sha": chunk_sha,
                    "source": relation.get("source"),
                    "target": relation.get("target"),
                    "relation_type": relation.get("type"),
                    "weight": _as_float(relation.get("weight")),
                }
            )
        for index, claim in enumerate(payload.get("claims") or []):
            if not isinstance(claim, dict):
                continue
            claims.append(
                {
                    "chunk_sha": chunk_sha,
                    "claim_index": index,
                    "text": claim.get("text"),
                    "polarity": claim.get("polarity"),
                    "subject": claim.get("subject") or claim.get("entity"),
                }
            )
    return (
        pd.DataFrame(entities, columns=["chunk_sha", "name", "entity_type", "aliases"]),
        pd.DataFrame(
            relations, columns=["chunk_sha", "source", "target", "relation_type", "weight"]
        ),
        pd.DataFrame(claims, columns=["chunk_sha", "claim_index", "text", "polarity", "subject"]),
    )


# ---------------------------------------------------------------------------
# Coercions — transcript-lab's stores are permissive, Delta is not
# ---------------------------------------------------------------------------


def _load_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _as_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    return [float(item) for item in value]


def _kind_from_name(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    return stem.split("-", 1)[0] if "-" in stem else stem.removesuffix(".json")
