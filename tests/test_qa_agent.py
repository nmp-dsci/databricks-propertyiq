"""Tests for the LangGraph QA agent — scripted LLM and executor, no network."""

from __future__ import annotations

import pytest

from lib.qa_agent import ask, build_agent, ensure_select_only


class ScriptedLLM:
    """Returns queued responses in order; records what it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_ensure_select_only_accepts_select_and_cte():
    assert ensure_select_only("SELECT 1").startswith("SELECT")
    assert ensure_select_only("```sql\nWITH t AS (SELECT 1) SELECT * FROM t\n```").startswith(
        "WITH"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "DROP TABLE gold_property_rent",
        "SELECT 1; DROP TABLE x",
        "UPDATE t SET a=1",
        "CREATE TABLE t AS SELECT 1",
    ],
)
def test_ensure_select_only_rejects_writes(bad):
    with pytest.raises(ValueError):
        ensure_select_only(bad)


def test_happy_path_routes_plans_executes_answers():
    llm = ScriptedLLM(
        [
            "ANSWER",
            "SELECT median_weekly_rent FROM workspace.propertyiq.gold_property_rent LIMIT 1",
            "The median weekly rent is $1,350 as at 2026-07.",
        ]
    )
    rows = [{"median_weekly_rent": 1350.0}]
    agent = build_agent(llm, lambda sql: rows)
    result = ask(agent, "Median rent for a 2-bed unit in 2000?")
    assert "1,350" in result["answer"]
    assert result["sql"].startswith("SELECT")
    assert any("route: answer" in line for line in result["decision_log"])
    assert any("execute: 1 rows" in line for line in result["decision_log"])


def test_out_of_scope_question_is_refused_without_sql():
    llm = ScriptedLLM(["REFUSE"])
    agent = build_agent(llm, lambda sql: (_ for _ in ()).throw(AssertionError("no SQL expected")))
    result = ask(agent, "Which selling agent sold the most houses?")
    assert "don't have data" in result["answer"]
    assert result["sql"] == ""


def test_failed_sql_retries_once_with_error_context_then_succeeds():
    llm = ScriptedLLM(
        [
            "ANSWER",
            "SELECT wrong_col FROM workspace.propertyiq.gold_property_rent",
            "SELECT median_weekly_rent FROM workspace.propertyiq.gold_property_rent",
            "Fixed: $1,350.",
        ]
    )
    calls = {"n": 0}

    def flaky_sql(sql):
        calls["n"] += 1
        if "wrong_col" in sql:
            raise RuntimeError("COLUMN_NOT_FOUND: wrong_col")
        return [{"median_weekly_rent": 1350.0}]

    agent = build_agent(llm, flaky_sql)
    result = ask(agent, "Median rent?")
    assert calls["n"] == 2
    assert "1,350" in result["answer"]
    # The retry prompt must carry the error back to the model
    retry_prompt = llm.calls[2][-1]["content"]
    assert "COLUMN_NOT_FOUND" in retry_prompt


def test_persistent_failure_is_reported_honestly():
    llm = ScriptedLLM(
        [
            "ANSWER",
            "SELECT a FROM t",
            "SELECT b FROM t",
        ]
    )
    agent = build_agent(llm, lambda sql: (_ for _ in ()).throw(RuntimeError("TABLE_NOT_FOUND")))
    result = ask(agent, "Median rent?")
    assert "couldn't answer" in result["answer"]
    assert "TABLE_NOT_FOUND" in result["answer"]


def test_non_select_llm_output_is_rejected_and_reported():
    llm = ScriptedLLM(
        [
            "ANSWER",
            "DROP TABLE gold_property_rent",
            "DELETE FROM gold_property_rent",
        ]
    )
    executed = []
    agent = build_agent(llm, lambda sql: executed.append(sql) or [])
    result = ask(agent, "Median rent?")
    assert executed == []  # the guardrail never let anything through
    assert "couldn't answer" in result["answer"]
