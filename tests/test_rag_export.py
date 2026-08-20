"""Tests for the transcript-lab exporter's pure transforms.

The fixtures mirror real shapes taken from transcript-rag-agent's stores: a
chunk carries its context header separately from its text, a raw transcript's
document body is a packed JSON blob rather than prose, and the graph cache is
keyed by chunk hash.
"""

from __future__ import annotations

import json

import pandas as pd

from lib.rag_export import (
    ENTITY_KEYS,
    chunk_rows,
    eval_run_rows,
    frame_sha,
    golden_answer_rows,
    golden_rows,
    graph_rows,
    landing_name,
    parse_landed,
    segment_rows,
    summary_rows,
    theme_rows,
    transcript_rows,
)

CHUNK_RECORDS = [
    {
        "id": "chunk:abc123:0",
        "document": "Unity Catalog governs tables and models.",
        "metadata": {
            "video_id": "abc123",
            "chunk_index": 0,
            "context_header": "Databricks Overview [00:00-01:00]",
            "title": "Databricks Overview",
            "channel_name": "Databricks",
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "segment_count": 4,
        },
        "embedding": [0.1, 0.2, 0.3],
    },
    {
        "id": "chunk:abc123:1",
        "document": "Delta Lake supports MERGE and time travel.",
        "metadata": {"video_id": "abc123", "chunk_index": 1},
        "embedding": [0.4, 0.5, 0.6],
    },
]

RAW_RECORDS = [
    {
        "id": "raw_transcript:abc123",
        # The real store packs this blob into the document body instead of text.
        "document": json.dumps(
            {
                "segments": [
                    {"text": "hello", "offset_ms": 0, "start_seconds": 0.0},
                    {"text": "world", "offset_ms": 1500, "start_seconds": 1.5},
                ],
                "description": "A talk about the lakehouse",
                "tags": ["databricks", "data"],
                "transcript_languages": ["en"],
                "summary_embedding": [0.9, 0.8],
            }
        ),
        "metadata": {
            "video_id": "abc123",
            "title": "Databricks Overview",
            "channel_name": "Databricks",
            "segment_count": 2,
            "view_count": "1234",
            "duration_seconds": 610.5,
        },
    }
]


def test_chunk_key_is_stable_and_independent_of_the_churning_chunk_id():
    frame = chunk_rows(CHUNK_RECORDS)
    assert list(frame["chunk_key"]) == ["abc123:0", "abc123:1"]
    # A local re-index rewrites chunk_id but not the (video, index) position.
    renamed = json.loads(json.dumps(CHUNK_RECORDS))
    renamed[0]["id"] = "chunk:abc123:0:v2"
    assert chunk_rows(renamed).iloc[0]["chunk_key"] == "abc123:0"


def test_chunk_rows_rebuilds_embedding_text_and_hashes_text():
    frame = chunk_rows(CHUNK_RECORDS)
    assert len(frame) == 2
    first = frame.iloc[0]
    # embedding_text must match what transcript-lab embeds: header + newline + text
    assert (
        first["embedding_text"]
        == "Databricks Overview [00:00-01:00]\nUnity Catalog governs tables and models."
    )
    assert first["video_id"] == "abc123"
    assert first["chunk_index"] == 0
    assert first["embedding"] == [0.1, 0.2, 0.3]
    assert len(first["text_sha"]) == 16
    # A chunk with no header embeds its bare text rather than a stray newline.
    assert frame.iloc[1]["embedding_text"] == "Delta Lake supports MERGE and time travel."


def test_chunk_rows_text_sha_tracks_content_not_position():
    changed = json.loads(json.dumps(CHUNK_RECORDS))  # deep copy
    changed[0]["document"] = "Unity Catalog governs tables, models and volumes."
    before = chunk_rows(CHUNK_RECORDS).iloc[0]["text_sha"]
    after = chunk_rows(changed).iloc[0]["text_sha"]
    assert before != after


def test_transcript_rows_unpacks_the_json_document_blob():
    frame = transcript_rows(RAW_RECORDS)
    row = frame.iloc[0]
    assert row["description"] == "A talk about the lakehouse"
    assert row["tags"] == ["databricks", "data"]
    assert row["transcript_languages"] == ["en"]
    assert row["view_count"] == 1234  # coerced from the string the store holds
    assert row["duration_seconds"] == 610.5


def test_segment_rows_explode_with_stable_indexes():
    frame = segment_rows(RAW_RECORDS)
    assert list(frame["segment_index"]) == [0, 1]
    assert list(frame["text"]) == ["hello", "world"]
    assert frame.iloc[1]["offset_ms"] == 1500
    assert set(frame["video_id"]) == {"abc123"}


def test_summary_rows_take_the_document_as_the_summary():
    records = [
        {
            "id": "summary:abc123",
            "document": "This video explains the lakehouse.",
            "metadata": {"video_id": "abc123", "summary_model": "gpt", "chunk_count": "12"},
        }
    ]
    row = summary_rows(records).iloc[0]
    assert row["summary"] == "This video explains the lakehouse."
    assert row["chunk_count"] == 12


def test_golden_rows_flag_anchors_as_stale():
    payload = {
        "corpus": {"videos": 23, "chunks": 488, "verified_at": "2026-07-27"},
        "entries": [
            {
                "id": "q1",
                "question": "What is Unity Catalog?",
                "reference_answer": "A governance layer.",
                "expected_video_ids": ["abc123"],
                "expected_chunk_ids": ["chunk:abc123:0"],
                "domain": "property",
            }
        ],
    }
    row = golden_rows(payload).iloc[0]
    assert row["case_id"] == "q1"
    assert row["expected_video_ids"] == ["abc123"]
    # The corpus grew ~6x past the verification point, so chunk anchors are not
    # trusted until phase 5 re-anchors them by text hash.
    assert row["anchor_status"] == "stale_pending_reanchor"
    assert row["question_type"] == "local"  # defaulted, matching GoldenEntry
    assert row["corpus_videos"] == 23


def test_eval_run_rows_keep_varied_shapes_as_json():
    runs = [
        ("ablation-2026-07.json", {"run_id": "ab1", "kind": "ablation", "cells": [1, 2]}),
        ("matrix-2026-07.json", {"run_id": "mx1", "entry_count": 20}),
    ]
    frame = eval_run_rows(runs)
    assert list(frame["run_id"]) == ["ab1", "mx1"]
    assert frame.iloc[0]["kind"] == "ablation"
    assert frame.iloc[1]["kind"] == "matrix"  # inferred from the filename
    assert json.loads(frame.iloc[0]["payload_json"])["cells"] == [1, 2]


def test_theme_rows_handle_both_wrapped_and_bare_payloads():
    themes = [{"theme_id": "t1", "title": "MLOps", "member_count": 4, "cross_video": True}]
    bare = theme_rows(themes)
    wrapped = theme_rows({"themes": themes})
    assert len(bare) == len(wrapped) == 1
    assert bare.iloc[0]["theme_id"] == "t1"
    assert bool(bare.iloc[0]["cross_video"]) is True


def test_graph_rows_split_into_three_tables_keyed_by_chunk_sha():
    extractions = [
        (
            "deadbeef",
            {
                "entities": [{"name": "Unity Catalog", "type": "product", "aliases": ["UC"]}],
                "relations": [
                    {"source": "Unity Catalog", "target": "Delta", "type": "governs", "weight": 0.8}
                ],
                "claims": [{"text": "UC governs tables", "polarity": "positive"}],
            },
        )
    ]
    entities, relations, claims = graph_rows(extractions)
    assert entities.iloc[0]["name"] == "Unity Catalog"
    assert entities.iloc[0]["aliases"] == ["UC"]
    assert relations.iloc[0]["weight"] == 0.8
    assert claims.iloc[0]["claim_index"] == 0
    assert {entities.iloc[0]["chunk_sha"], relations.iloc[0]["chunk_sha"]} == {"deadbeef"}


def test_graph_rows_tolerate_missing_sections():
    entities, relations, claims = graph_rows([("sha1", {"entities": []})])
    assert entities.empty and relations.empty and claims.empty
    # Columns still exist so the Parquet schema stays stable across exports.
    assert list(claims.columns) == ["chunk_sha", "claim_index", "text", "polarity", "subject"]


# ---------------------------------------------------------------------------
# The landing contract — this is what makes re-exports free
# ---------------------------------------------------------------------------


def test_frame_sha_is_row_order_independent():
    frame = chunk_rows(CHUNK_RECORDS)
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    assert frame_sha(frame, ("video_id", "chunk_index")) == frame_sha(
        shuffled, ("video_id", "chunk_index")
    )


def test_frame_sha_changes_when_content_changes():
    frame = chunk_rows(CHUNK_RECORDS)
    edited = frame.copy()
    edited.loc[0, "text"] = "something else entirely"
    assert frame_sha(frame) != frame_sha(edited)


def test_frame_sha_notices_a_changed_embedding():
    # A re-embed with the same text must still land: the vector is the payload.
    frame = chunk_rows(CHUNK_RECORDS)
    edited = frame.copy()
    edited.at[0, "embedding"] = [9.9, 9.9, 9.9]
    assert frame_sha(frame) != frame_sha(edited)


def test_frame_sha_is_stable_across_calls():
    frame = chunk_rows(CHUNK_RECORDS)
    assert frame_sha(frame) == frame_sha(chunk_rows(CHUNK_RECORDS))


def test_empty_frame_has_a_sentinel_sha():
    assert frame_sha(pd.DataFrame()) == "empty000"


def test_landing_name_and_parse_round_trip():
    name = landing_name("chunks", "82f5daf4")
    assert name == "chunks_82f5daf4.parquet"
    assert parse_landed(name) == ("chunks", "82f5daf4")
    # Entities with underscores survive the round trip.
    assert parse_landed(landing_name("graph_entities", "aaf2581d")) == (
        "graph_entities",
        "aaf2581d",
    )


def test_parse_landed_rejects_off_contract_names():
    assert parse_landed("chunks.parquet") is None
    assert parse_landed("chunks_short.parquet") is None
    assert parse_landed("chunks_82f5daf4.csv") is None


# ---------------------------------------------------------------------------
# golden_answers (s09) — judged dev answers as the optimisation target
# ---------------------------------------------------------------------------

GOLDEN_ANSWER_RECORD = {
    "question_id": "b01",
    "category": "broad",
    "domain": "databricks",
    "question": "Guide me on Databricks for an SA interview?",
    "source_case_id": None,
    "answer": "## Key Findings\n1. Know the lakehouse. [1]",
    "answer_model": "deepseek-v4-flash",
    "contexts": ["chunk text one", "chunk text two"],
    "references": [{"label": 1, "video_id": "v1"}],
    "chunk_ids": ["c1", "c2"],
    "queries": ["databricks lakehouse", "unity catalog"],
    "iterations": 7,
    "llm_calls": 8,
    "terminated_reason": "completed",
    "captured_at": "2026-08-20T04:00:00+00:00",
    "scores": {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9,
        "context_precision": 0.8,
        "insight_depth": 0.7,
        "specificity": 0.6,
        "coverage": 0.5,
        "evidence_breadth": 0.4,
        "calibration": 0.3,
    },
    "ragas_v1_composite": 0.9,
    "composite": 0.61,
    "cap_applied": False,
    "grounding_floor_breached": False,
    "judge_model": "deepseek-v4-flash",
    "rubric_version": "depth-v2",
    "judge_error": None,
}


def test_golden_answer_rows_flatten_scores_and_json_encode_lists():
    frame = golden_answer_rows([GOLDEN_ANSWER_RECORD])
    row = frame.iloc[0]
    assert row["question_id"] == "b01"
    assert row["composite"] == 0.61
    assert row["evidence_breadth"] == 0.4
    assert row["iterations"] == 7
    assert json.loads(row["contexts_json"]) == ["chunk text one", "chunk text two"]
    assert json.loads(row["queries_json"]) == ["databricks lakehouse", "unity catalog"]
    assert row["source"] == "transcript-lab/rag_agent"
    assert row["judge_error"] == ""


def test_golden_answer_rows_tolerate_an_unjudged_record():
    unjudged = {
        "question_id": "n01",
        "category": "narrow",
        "domain": "property",
        "question": "q",
        "answer": "a",
        "judge_error": "no retrieval contexts; not judged",
    }
    frame = golden_answer_rows([unjudged])
    row = frame.iloc[0]
    assert pd.isna(row["composite"])
    assert pd.isna(row["faithfulness"])
    assert row["cap_applied"] is False or row["cap_applied"] == False  # noqa: E712 — numpy bool
    assert row["judge_error"] == "no retrieval contexts; not judged"


def test_golden_answers_entity_registered_with_question_id_key():
    assert ENTITY_KEYS["golden_answers"] == ("question_id",)
    frame = golden_answer_rows([GOLDEN_ANSWER_RECORD])
    assert frame_sha(frame, ENTITY_KEYS["golden_answers"]) != "empty000"
