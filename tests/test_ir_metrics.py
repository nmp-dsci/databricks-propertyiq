"""Tests for the retrieval metrics behind the Chroma-vs-Vector-Search verdict."""

from __future__ import annotations

import pytest

from lib.ir_metrics import (
    aggregate,
    mrr,
    ndcg_at_k,
    reanchor_chunk_ids,
    recall_at_k,
    score_case,
)


def test_recall_counts_relevant_items_found_in_the_top_k():
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=10) == 1.0
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=10) == 0.5
    assert recall_at_k(["x", "y"], ["a"], k=10) == 0.0


def test_recall_respects_the_cutoff():
    # The relevant item sits at rank 3, outside k=2.
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
    assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0


def test_mrr_is_the_reciprocal_of_the_first_hit():
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_mrr_uses_the_earliest_relevant_hit_when_several_match():
    assert mrr(["x", "b", "a"], ["a", "b"]) == 0.5


def test_ndcg_is_one_for_a_perfect_ranking_and_falls_with_position():
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=10) == pytest.approx(1.0)
    # Same items, worse order — still found, but ranked below an irrelevant hit.
    demoted = ndcg_at_k(["x", "a", "b"], ["a", "b"], k=10)
    assert 0.0 < demoted < 1.0


def test_ndcg_normalisation_makes_questions_comparable():
    # One expected video vs three: a perfect ranking scores 1.0 either way.
    assert ndcg_at_k(["a"], ["a"], k=10) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=10) == pytest.approx(1.0)


def test_duplicate_retrievals_do_not_inflate_scores():
    # Multi-hop modes retrieve the same video repeatedly; ranking is by first
    # appearance, and a repeat must not count as a second hit.
    assert recall_at_k(["a", "a", "a"], ["a", "b"], k=10) == 0.5
    assert mrr(["a", "a"], ["a"]) == 1.0


def test_empty_inputs_score_zero_rather_than_dividing_by_zero():
    assert recall_at_k([], ["a"]) == 0.0
    assert mrr([], ["a"]) == 0.0
    assert ndcg_at_k([], ["a"]) == 0.0
    # A case with no expected items is unscoreable, not perfect.
    assert recall_at_k(["a"], []) == 0.0
    assert ndcg_at_k(["a"], []) == 0.0


def test_score_case_names_metrics_by_their_cutoff():
    scores = score_case(["a"], ["a"], k=5)
    assert set(scores) == {"recall_at_5", "mrr", "ndcg_at_5"}


def test_aggregate_averages_across_questions():
    means = aggregate([{"mrr": 1.0}, {"mrr": 0.0}])
    assert means["mrr"] == 0.5
    assert aggregate([]) == {}


def test_aggregate_treats_a_missing_metric_as_zero():
    # A contender that errored on one question must not be flattered by it.
    assert aggregate([{"mrr": 1.0}, {}])["mrr"] == 0.5


def test_reanchor_resolves_by_content_hash_and_reports_the_rest():
    resolved, unresolved = reanchor_chunk_ids(
        ["chunk:v1:0", "chunk:v9:7"],
        text_sha_by_chunk_id={"chunk:v1:0": "sha_a", "chunk:v9:7": "sha_gone"},
        chunk_key_by_text_sha={"sha_a": "v1:3"},
    )
    # The chunk moved position (0 -> 3) but its text is unchanged, so it resolves.
    assert resolved == ["v1:3"]
    # Its text is no longer in the corpus — a confirmation pass has to rule.
    assert unresolved == ["chunk:v9:7"]


def test_reanchor_reports_ids_it_has_no_recorded_text_for():
    resolved, unresolved = reanchor_chunk_ids(
        ["chunk:unknown:0"], text_sha_by_chunk_id={}, chunk_key_by_text_sha={}
    )
    assert resolved == []
    assert unresolved == ["chunk:unknown:0"]
