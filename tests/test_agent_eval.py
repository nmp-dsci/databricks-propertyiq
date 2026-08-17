"""Tests for the benchmark graders and the golden-set loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from lib.agent_eval import (
    extract_numbers,
    grade,
    load_cases,
    refusal_grade,
    topk_grade,
    value_grade,
)

GOLDEN = Path(__file__).parent.parent / "evals" / "golden_qa.yaml"


def test_extract_numbers_handles_currency_percent_and_commas():
    nums = extract_numbers("Median is $1,350 per week (6.95% yield, 5,098,794 total).")
    assert 1350.0 in nums
    assert 6.95 in nums
    assert 5098794.0 in nums


def test_value_grade_within_and_outside_tolerance():
    assert value_grade("The median weekly rent is $1,350.", 1350.0)
    assert value_grade("around $1,400 a week", 1350.0, tolerance_pct=5.0)  # 3.7% off
    assert not value_grade("around $1,500 a week", 1350.0, tolerance_pct=5.0)
    assert not value_grade("", 1350.0)


def test_value_grade_ignores_unrelated_numbers_outside_band():
    # A basis month and a bond count shouldn't rescue a wrong value.
    assert not value_grade("As at 2026-07 there were 173 bonds.", 1350.0)


def test_topk_grade_counts_whole_word_hits():
    answer = "Top postcodes: 2063, 2030 and 2095 lead the ranking."
    assert topk_grade(answer, ["2063", "2030", "2095", "2088", "2026"], min_hits=3)
    assert not topk_grade(answer, ["2088", "2026", "2110"], min_hits=2)
    # '2063' inside a longer number must not count as a hit
    assert not topk_grade("value 120632 is irrelevant", ["2063"], min_hits=1)


def test_refusal_grade():
    assert refusal_grade("I don't have data on selling agents.")
    assert refusal_grade("That is outside the scope of the property datasets.")
    assert not refusal_grade("The top agent sold 42 houses.")


def test_golden_set_loads_and_grades_candidate_answers():
    cases = load_cases(GOLDEN)
    assert len(cases) == 3
    # Each case's own candidate answer must pass its grader — if the golden
    # answer can't pass, no agent ever could.
    for case in cases:
        result = grade(case, case["candidate_answer"])
        assert result["passed"], f"candidate answer fails its own grader: {case['case_key']}"


def test_grade_rejects_unknown_kind():
    case = {"case_key": "x", "tier": "T1", "grader": {"kind": "vibes"}}
    with pytest.raises(ValueError):
        grade(case, "answer")
