"""Tests for rag_transcript_agent's three answer modes.

Both dependencies are scripted, so the graph's control flow is under test with
no network: which nodes run, how many retrievals each mode issues, and what
happens when retrieval comes back empty or the ReAct step returns nonsense.
"""

from __future__ import annotations

import pytest

from lib.rag_agent import (
    MODES,
    _parse_action,
    _timestamp,
    ask,
    build_agent,
    dedupe_chunks,
    format_chunks,
)


def chunk(key: str, score: float = 0.5, text: str = "some transcript text") -> dict:
    return {
        "chunk_key": key,
        "video_id": key.split(":")[0],
        "title": f"Video {key.split(':')[0]}",
        "channel_name": "Databricks",
        "start_seconds": 65.0,
        "text": text,
        "score": score,
    }


class ScriptedLLM:
    """Returns queued replies in order; records what it was asked."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.replies.pop(0) if self.replies else "final answer [Video v1 @ 01:05]"


class ScriptedRetriever:
    """Returns a fixed hit list per call; records every query it saw."""

    def __init__(self, results: list[list[dict]] | None = None):
        self.results = list(results or [])
        self.queries: list[str] = []

    def __call__(self, query: str, k: int = 8) -> list[dict]:
        self.queries.append(query)
        return self.results.pop(0) if self.results else [chunk("v1:0")]


# ---------------------------------------------------------------------------
# single
# ---------------------------------------------------------------------------


def test_single_mode_retrieves_once_with_the_question_as_asked():
    llm = ScriptedLLM(["Unity Catalog governs data. [Video v1 @ 01:05]"])
    retriever = ScriptedRetriever([[chunk("v1:0"), chunk("v1:1")]])
    result = ask(build_agent(llm, retriever, mode="single"), "What is Unity Catalog?")

    assert retriever.queries == ["What is Unity Catalog?"]
    assert result["chunk_keys"] == ["v1:0", "v1:1"]
    assert "Unity Catalog governs data" in result["answer"]
    assert any("retrieve:" in line for line in result["decision_log"])


def test_answer_prompt_carries_the_excerpts_and_the_grounding_rules():
    llm = ScriptedLLM(["answer"])
    retriever = ScriptedRetriever([[chunk("v1:0", text="Delta Lake supports MERGE")]])
    ask(build_agent(llm, retriever, mode="single"), "Does Delta support MERGE?")

    system, user = llm.calls[-1]
    assert "Answer ONLY from the retrieved excerpts" in system["content"]
    assert "Delta Lake supports MERGE" in user["content"]
    # The citation format needs the timestamp to be in the prompt.
    assert "@ 01:05" in user["content"]


def test_empty_retrieval_refuses_instead_of_inventing():
    llm = ScriptedLLM([])
    retriever = ScriptedRetriever([[]])
    result = ask(build_agent(llm, retriever, mode="single"), "What is quantum gravity?")

    assert "couldn't find anything" in result["answer"]
    assert result["chunk_keys"] == []
    # The model must never be asked to answer with no evidence.
    assert llm.calls == []


# ---------------------------------------------------------------------------
# multi
# ---------------------------------------------------------------------------


def test_multi_mode_decomposes_then_retrieves_per_subquestion():
    llm = ScriptedLLM(["What is Unity Catalog?\nHow does Delta sharing work?", "combined answer"])
    retriever = ScriptedRetriever([[chunk("v1:0")], [chunk("v2:0")]])
    result = ask(build_agent(llm, retriever, mode="multi"), "Compare UC and Delta sharing")

    assert retriever.queries == ["What is Unity Catalog?", "How does Delta sharing work?"]
    assert set(result["chunk_keys"]) == {"v1:0", "v2:0"}
    assert any("decompose: 2" in line for line in result["decision_log"])


def test_multi_mode_caps_subquestions():
    llm = ScriptedLLM(["q1\nq2\nq3\nq4\nq5", "answer"])
    retriever = ScriptedRetriever()
    ask(build_agent(llm, retriever, mode="multi"), "a sprawling question")
    assert len(retriever.queries) == 3  # MAX_SUBQUESTIONS


def test_multi_mode_falls_back_to_the_original_question_when_decompose_is_empty():
    llm = ScriptedLLM(["   ", "answer"])
    retriever = ScriptedRetriever()
    ask(build_agent(llm, retriever, mode="multi"), "the original")
    assert retriever.queries == ["the original"]


# ---------------------------------------------------------------------------
# agentic
# ---------------------------------------------------------------------------


def test_agentic_mode_loops_until_the_model_says_it_has_enough():
    llm = ScriptedLLM(
        [
            '{"action": "search", "query": "unity catalog lineage"}',
            '{"action": "answer"}',
            "grounded answer",
        ]
    )
    retriever = ScriptedRetriever([[chunk("v1:0")], [chunk("v2:0")]])
    result = ask(build_agent(llm, retriever, mode="agentic", protocol="v1"), "How does UC track lineage?")

    # One seed retrieval, then one model-chosen follow-up.
    assert retriever.queries == ["How does UC track lineage?", "unity catalog lineage"]
    assert set(result["chunk_keys"]) == {"v1:0", "v2:0"}
    assert any("search again" in line for line in result["decision_log"])


def test_agentic_mode_stops_at_the_hop_cap():
    # Always asks for another search; the cap is what ends it.
    llm = ScriptedLLM(['{"action": "search", "query": "more"}'] * 10 + ["answer"])
    retriever = ScriptedRetriever()
    result = ask(build_agent(llm, retriever, mode="agentic", protocol="v1"), "unanswerable")

    assert len(retriever.queries) == 1 + 3  # seed + MAX_HOPS
    assert any("hop cap" in line for line in result["decision_log"])


def test_agentic_mode_answers_when_the_react_step_is_unparseable():
    llm = ScriptedLLM(["I think we should probably search more?", "answer"])
    retriever = ScriptedRetriever()
    result = ask(build_agent(llm, retriever, mode="agentic", protocol="v1"), "a question")

    assert len(retriever.queries) == 1  # seed only — no runaway loop
    assert result["answer"] == "answer"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_dedupe_keeps_the_best_scoring_copy():
    deduped = dedupe_chunks([chunk("v1:0", 0.2), chunk("v1:0", 0.9), chunk("v2:0", 0.5)])
    assert [c["chunk_key"] for c in deduped] == ["v1:0", "v2:0"]
    assert deduped[0]["score"] == 0.9  # ranked by score, best copy kept


def test_ask_reports_distinct_videos_for_the_ir_metrics():
    llm = ScriptedLLM(["answer"])
    retriever = ScriptedRetriever([[chunk("v1:0"), chunk("v1:1"), chunk("v2:0")]])
    result = ask(build_agent(llm, retriever, mode="single"), "q")
    assert result["video_ids"] == ["v1", "v2"]


def test_timestamp_formatting():
    assert _timestamp(65) == "01:05"
    assert _timestamp(3599.4) == "59:59"
    assert _timestamp(None) == "00:00"
    assert _timestamp("not a number") == "00:00"


def test_format_chunks_handles_the_empty_case():
    assert "no excerpts" in format_chunks([])


def test_parse_action_tolerates_fences_and_prose():
    assert _parse_action('```json\n{"action": "answer"}\n```')["action"] == "answer"
    assert _parse_action('Sure! {"action": "search", "query": "x"}')["query"] == "x"
    assert _parse_action("no json at all")["action"] == "answer"


def test_unknown_mode_is_rejected_early():
    with pytest.raises(ValueError, match="unknown mode"):
        build_agent(ScriptedLLM([]), ScriptedRetriever(), mode="graphrag")


def test_every_declared_mode_builds():
    for mode in MODES:
        assert build_agent(ScriptedLLM([]), ScriptedRetriever(), mode=mode) is not None
