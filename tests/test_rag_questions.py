"""The judged eval's question set — shape and stratification invariants."""

from __future__ import annotations

import pytest

from lib.rag_questions import BROAD, CANONICAL_QUESTION_ID, NARROW, QUESTIONS, by_id


def test_twelve_questions_six_per_stratum():
    assert len(QUESTIONS) == 12
    assert sum(1 for q in QUESTIONS if q["category"] == BROAD) == 6
    assert sum(1 for q in QUESTIONS if q["category"] == NARROW) == 6


def test_question_ids_unique_and_prefixed_by_stratum():
    ids = [q["question_id"] for q in QUESTIONS]
    assert len(set(ids)) == 12
    for question in QUESTIONS:
        prefix = "b" if question["category"] == BROAD else "n"
        assert str(question["question_id"]).startswith(prefix)


def test_canonical_sa_interview_question_present_and_broad():
    canonical = by_id(CANONICAL_QUESTION_ID)
    assert canonical["category"] == BROAD
    assert "solutions architect" in str(canonical["question"]).lower()
    assert "databricks" in str(canonical["question"]).lower()


def test_by_id_raises_on_unknown():
    with pytest.raises(KeyError):
        by_id("zz99")


def test_every_question_has_text_and_domain():
    for question in QUESTIONS:
        assert str(question["question"]).strip()
        assert str(question["domain"]).strip()
