"""Protocol v2 (the s09 research-protocol port) — scripted, no network.

v1's binary sufficiency check is pinned by tests/test_rag_agent.py; these
tests pin what v2 changes — gap-analysis decide prompt, repeated-query guard,
the raised hop budget, the Key Findings answer contract — and, just as
deliberately, what v2 must NOT change: single mode and v1 defaults.
"""

from __future__ import annotations

import json

import pytest
from test_rag_agent import ScriptedLLM, ScriptedRetriever, chunk

from lib.rag_agent import (
    DEFAULT_PROTOCOL,
    PROTOCOL_MAX_HOPS,
    ask,
    build_agent,
    normalize_content,
)


def search_reply(query: str, gaps: list[str] | None = None) -> str:
    action = {"action": "search", "query": query}
    if gaps:
        action["gaps"] = gaps
    return json.dumps(action)


def test_default_protocol_is_still_v1():
    assert DEFAULT_PROTOCOL == "v1"
    assert PROTOCOL_MAX_HOPS["v1"] == 3


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError, match="unknown protocol"):
        build_agent(ScriptedLLM([]), ScriptedRetriever(), mode="agentic", protocol="v9")


def test_v2_decide_prompt_carries_checklist_calibration_and_asked_queries():
    llm = ScriptedLLM([json.dumps({"action": "answer"}), "answer [Video v1 @ 01:05]"])
    retriever = ScriptedRetriever([[chunk("v1:0")]])
    agent = build_agent(llm, retriever, mode="agentic", protocol="v2")
    ask(agent, "Give me a guide to Databricks for a solutions architect interview")

    decide_prompt = llm.calls[0][1]["content"]
    assert "3-6 focused searches" in decide_prompt
    assert "never repeat these" in decide_prompt
    # The seed query is already in the asked list the model sees.
    assert "Give me a guide to Databricks" in decide_prompt.split("Queries already searched")[1]
    assert '"gaps"' in decide_prompt


def test_v2_loops_through_focused_queries_and_accumulates_evidence():
    llm = ScriptedLLM(
        [
            search_reply("lakehouse architecture basics", gaps=["architecture"]),
            search_reply("unity catalog governance", gaps=["governance"]),
            json.dumps({"action": "answer"}),
            "## Key Findings\n1. ... [Video v1 @ 01:05]",
        ]
    )
    retriever = ScriptedRetriever([[chunk("v1:0")], [chunk("v2:0")], [chunk("v3:0")]])
    agent = build_agent(llm, retriever, mode="agentic", protocol="v2")
    result = ask(agent, "broad question")

    # seed + two focused hops, every query distinct
    assert retriever.queries == [
        "broad question",
        "lakehouse architecture basics",
        "unity catalog governance",
    ]
    assert set(result["chunk_keys"]) == {"v1:0", "v2:0", "v3:0"}
    assert any("gaps: architecture" in line for line in result["decision_log"])


def test_v2_blocks_a_repeated_query_instead_of_spending_a_hop():
    llm = ScriptedLLM(
        [
            search_reply("unity catalog"),
            search_reply("unity catalog"),  # repeat — must be blocked
            "answer [Video v1 @ 01:05]",
        ]
    )
    retriever = ScriptedRetriever([[chunk("v1:0")], [chunk("v2:0")]])
    agent = build_agent(llm, retriever, mode="agentic", protocol="v2")
    result = ask(agent, "question")

    assert retriever.queries == ["question", "unity catalog"]
    assert any("repeated query" in line for line in result["decision_log"])


def test_v2_hop_budget_is_six_and_stops_the_loop():
    replies = [search_reply(f"query {i}") for i in range(10)] + ["answer [Video v1 @ 01:05]"]
    llm = ScriptedLLM(replies)
    retriever = ScriptedRetriever()
    agent = build_agent(llm, retriever, mode="agentic", protocol="v2")
    result = ask(agent, "question")

    # seed + 6 hops; the 7th decide hits the cap
    assert len(retriever.queries) == 1 + PROTOCOL_MAX_HOPS["v2"]
    assert any("hop cap (6)" in line for line in result["decision_log"])


def test_max_hops_override_beats_the_protocol_default():
    replies = [search_reply(f"query {i}") for i in range(5)] + ["answer [Video v1 @ 01:05]"]
    agent = build_agent(
        ScriptedLLM(replies), ScriptedRetriever(), mode="agentic", protocol="v2", max_hops=2
    )
    result = ask(agent, "question")
    assert any("hop cap (2)" in line for line in result["decision_log"])


def test_v2_answer_contract_asks_for_key_findings():
    llm = ScriptedLLM([json.dumps({"action": "answer"}), "## Key Findings ..."])
    agent = build_agent(llm, ScriptedRetriever([[chunk("v1:0")]]), mode="agentic", protocol="v2")
    ask(agent, "question")
    assert "## Key Findings" in llm.calls[-1][1]["content"]


def test_v2_does_not_touch_single_mode():
    llm = ScriptedLLM(["plain answer [Video v1 @ 01:05]"])
    retriever = ScriptedRetriever([[chunk("v1:0")]])
    agent = build_agent(llm, retriever, mode="single", protocol="v2")
    ask(agent, "question")

    assert retriever.queries == ["question"]  # still exactly one retrieval
    assert "## Key Findings" not in llm.calls[-1][1]["content"]


# ---------------------------------------------------------------------------
# normalize_content — FMAPI reasoning models return typed part lists
# ---------------------------------------------------------------------------


def test_normalize_content_passes_strings_through():
    assert normalize_content("plain") == "plain"
    assert normalize_content(None) == ""


def test_normalize_content_extracts_text_parts_and_drops_reasoning():
    content = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        {"type": "text", "text": '{"action": "answer"}'},
    ]
    assert normalize_content(content) == '{"action": "answer"}'


def test_normalize_content_joins_multiple_text_parts():
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, "c"]
    assert normalize_content(content) == "a\nb\nc"
