"""The shipped agentic mode — tool-calling ReAct loop, scripted, no network.

The model researches by CALLING retrieve_transcript_chunks (native function
calling), not by answering a decide prompt: searching is the path of least
resistance, which the s09 matrix showed is the only shape whose research
behaviour survives every model temperament.
"""

from __future__ import annotations

import json

from test_rag_agent import ScriptedRetriever, chunk

from lib.rag_agent import DEFAULT_PROTOCOL, RETRIEVE_TOOL, ask, build_agent, build_react_agent


def tool_call(query: str, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "retrieve_transcript_chunks",
            "arguments": json.dumps({"query": query}),
        },
    }


class ScriptedToolLLM:
    """Returns queued assistant messages; records (messages, tools) pairs."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def __call__(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append((messages, tools))
        while self.replies:
            reply = self.replies.pop(0)
            if tools is None and reply.get("tool_calls"):
                # An OpenAI-compatible server cannot call tools it was not
                # offered; skip queued tool-call replies on a tool-less turn.
                continue
            return reply
        return {"content": "fallback answer [Video v1 @ 01:05]", "tool_calls": None}


def test_react_is_the_default_agentic_build():
    llm = ScriptedToolLLM([{"content": "answer [Video v1 @ 01:05]", "tool_calls": None}])
    agent = build_agent(llm, ScriptedRetriever(), mode="agentic")
    assert DEFAULT_PROTOCOL == "react"
    result = ask(agent, "q")
    assert result["answer"].startswith("answer")


def test_react_loops_search_read_search_until_plain_answer():
    llm = ScriptedToolLLM(
        [
            {"content": None, "tool_calls": [tool_call("databricks lakehouse", "c1")]},
            {"content": None, "tool_calls": [tool_call("unity catalog", "c2")]},
            {"content": "## Key Findings\n1. ... [Video v1 @ 01:05]", "tool_calls": None},
        ]
    )
    retriever = ScriptedRetriever([[chunk("v1:0")], [chunk("v2:0")]])
    result = ask(build_react_agent(llm, retriever), "guide me on Databricks")

    assert retriever.queries == ["databricks lakehouse", "unity catalog"]
    assert set(result["chunk_keys"]) == {"v1:0", "v2:0"}
    # The system prompt carries the research protocol; the tool spec is offered.
    first_messages, first_tools = llm.calls[0]
    assert first_messages[0]["role"] == "system"
    assert "Research protocol" in first_messages[0]["content"]
    assert first_tools == [RETRIEVE_TOOL]
    # Tool results flowed back as tool-role messages tied to their call ids.
    later_messages, _ = llm.calls[2]
    tool_roles = [m for m in later_messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_roles] == ["c1", "c2"]


def test_react_parallel_tool_calls_in_one_turn_all_execute():
    llm = ScriptedToolLLM(
        [
            {
                "content": None,
                "tool_calls": [tool_call("resumes", "c1"), tool_call("linkedin", "c2")],
            },
            {"content": "answer [Video v1 @ 01:05]", "tool_calls": None},
        ]
    )
    retriever = ScriptedRetriever([[chunk("v1:0")], [chunk("v2:0")]])
    result = ask(build_react_agent(llm, retriever), "job search advice?")
    assert retriever.queries == ["resumes", "linkedin"]
    assert len(result["chunk_keys"]) == 2


def test_react_hop_cap_forces_a_toolless_answer():
    replies = [
        {"content": None, "tool_calls": [tool_call(f"q{i}", f"c{i}")]} for i in range(20)
    ] + [{"content": "forced answer [Video v1 @ 01:05]", "tool_calls": None}]
    llm = ScriptedToolLLM(replies)
    retriever = ScriptedRetriever()
    result = ask(build_react_agent(llm, retriever, max_iterations=3), "q")

    assert len(retriever.queries) == 3
    # The capped call offers no tools, so the model cannot keep searching.
    final_messages, final_tools = llm.calls[-1]
    assert final_tools is None
    assert any("Stop researching" in (m.get("content") or "") for m in final_messages)
    assert result["answer"].startswith("forced answer")
    assert any("hop cap" in line for line in result["decision_log"])


def test_react_unparseable_tool_arguments_fall_back_to_the_question():
    bad = {
        "id": "c1",
        "type": "function",
        "function": {"name": "retrieve_transcript_chunks", "arguments": "{not json"},
    }
    llm = ScriptedToolLLM(
        [
            {"content": None, "tool_calls": [bad]},
            {"content": "answer [Video v1 @ 01:05]", "tool_calls": None},
        ]
    )
    retriever = ScriptedRetriever([[chunk("v1:0")]])
    ask(build_react_agent(llm, retriever), "the original question")
    assert retriever.queries == ["the original question"]


def test_react_empty_final_content_is_the_honest_miss():
    llm = ScriptedToolLLM([{"content": None, "tool_calls": None}])
    result = ask(build_react_agent(llm, ScriptedRetriever()), "q")
    assert "couldn't find anything" in result["answer"]


def test_react_normalizes_reasoning_model_content_lists():
    content = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
        {"type": "text", "text": "clean answer [Video v1 @ 01:05]"},
    ]
    llm = ScriptedToolLLM([{"content": content, "tool_calls": None}])
    result = ask(build_react_agent(llm, ScriptedRetriever()), "q")
    assert result["answer"] == "clean answer [Video v1 @ 01:05]"
