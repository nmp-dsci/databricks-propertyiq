"""Native FMAPI judges (lib.native_judges) — scripted LLM, no network."""

from __future__ import annotations

import json

from lib.native_judges import (
    DEPTH_METRICS,
    MAX_CONTEXTS,
    format_contexts,
    make_native_scorers,
    parse_json_verdict,
)


class ScriptedJudge:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, messages: list[dict]) -> str:
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else "{}"


def test_parse_json_verdict_tolerates_fences_and_prose():
    assert parse_json_verdict('```json\n{"score": 0.8, "rationale": "ok"}\n```') == {
        "score": 0.8,
        "rationale": "ok",
    }
    assert parse_json_verdict('The verdict: {"score": 1.0, "rationale": "good"} thanks') == {
        "score": 1.0,
        "rationale": "good",
    }
    assert parse_json_verdict("no json here") == {}
    assert parse_json_verdict("") == {}


def test_format_contexts_caps_count_and_length():
    contexts = [f"context {i} " + "x" * 2000 for i in range(30)]
    formatted = format_contexts(contexts)
    assert formatted.count("[") == MAX_CONTEXTS
    assert "context 0" in formatted and "context 15" in formatted
    assert "context 16" not in formatted


def test_groundedness_and_relevance_scorers_return_named_feedback():
    judge = ScriptedJudge(
        [
            json.dumps({"score": 0.9, "rationale": "well supported"}),
            json.dumps({"score": 0.7, "rationale": "mostly on topic"}),
        ]
    )
    groundedness, relevance, _ = make_native_scorers(llm=judge)
    g = groundedness(inputs={"question": "q"}, outputs={"response": "a", "contexts": ["c1", "c2"]})
    r = relevance(inputs={"question": "q"}, outputs={"response": "a"})
    assert g.name == "native_groundedness" and g.value == 0.9
    assert "well supported" in g.rationale
    assert r.name == "native_relevance" and r.value == 0.7
    # The groundedness prompt carried the contexts; relevance deliberately not.
    assert "c1" in judge.prompts[0] and "c1" not in judge.prompts[1]


def test_depth_scorer_returns_all_five_metrics():
    verdict = {
        metric: {"score": 0.5 + i / 10, "rationale": f"r{i}"}
        for i, metric in enumerate(DEPTH_METRICS)
    }
    judge = ScriptedJudge([json.dumps(verdict)])
    _, _, depth = make_native_scorers(llm=judge)
    feedbacks = depth(inputs={"question": "q"}, outputs={"response": "a", "contexts": ["c"]})
    assert [f.name for f in feedbacks] == [f"native_{m}" for m in DEPTH_METRICS]
    assert feedbacks[0].value == 0.5
    assert feedbacks[4].value == 0.9


def test_unparseable_judge_reply_yields_unscored_not_crash():
    judge = ScriptedJudge(["total nonsense", "also nonsense", "still nonsense"])
    groundedness, relevance, depth = make_native_scorers(llm=judge)
    assert groundedness(inputs={}, outputs={}).value == "unscored"
    assert relevance(inputs={}, outputs={}).value == "unscored"
    assert all(f.value == "unscored" for f in depth(inputs={}, outputs={}))


def test_scores_clamped_into_unit_interval():
    judge = ScriptedJudge([json.dumps({"score": 1.7, "rationale": "over"})])
    groundedness, _, _ = make_native_scorers(llm=judge)
    assert groundedness(inputs={}, outputs={}).value == 1.0
