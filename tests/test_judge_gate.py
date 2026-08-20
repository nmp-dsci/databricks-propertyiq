"""Gate arithmetic (lib.judge_gate) — the s09 success gate as unit-tested code."""

from __future__ import annotations

from lib.judge_gate import aggregate, gate_verdict, mean


def _row(qid: str, category: str, hops: int, composite: float | None) -> dict:
    return {"question_id": qid, "category": category, "hops": hops, "composite": composite}


GOLDENS = [
    _row("b01", "broad", 8, 0.80),
    _row("b02", "broad", 6, 0.70),
    _row("n01", "narrow", 5, 0.60),
]


def test_mean_ignores_missing():
    assert mean([]) is None
    assert mean([1.0, 2.0]) == 1.5


def test_aggregate_by_category_skips_unjudged_rows():
    rows = [_row("b01", "broad", 3, 0.5), _row("b02", "broad", 5, None)]
    summary = aggregate(rows, "broad")
    assert summary["n"] == 2.0
    assert summary["composite"] == 0.5  # None excluded, not counted as zero
    assert summary["hops"] == 4.0


def test_gate_passes_when_hops_and_composite_clear():
    variant = [
        _row("b01", "broad", 4, 0.78),
        _row("b02", "broad", 3, 0.70),
        _row("n01", "narrow", 1, 0.62),
    ]
    verdict = gate_verdict(variant, GOLDENS)
    assert verdict["passed"] is True
    names = [check["name"] for check in verdict["checks"]]
    assert names == ["broad_hops", "broad_composite"]


def test_gate_fails_on_single_hop_broad_questions():
    variant = [_row("b01", "broad", 1, 0.90), _row("b02", "broad", 1, 0.90)]
    verdict = gate_verdict(variant, GOLDENS)
    assert verdict["passed"] is False
    failed = {check["name"] for check in verdict["checks"] if not check["passed"]}
    assert failed == {"broad_hops"}


def test_gate_fails_when_composite_below_golden_tolerance():
    # golden broad mean = 0.75; floor at 5% tolerance = 0.7125
    variant = [_row("b01", "broad", 5, 0.70), _row("b02", "broad", 4, 0.70)]
    verdict = gate_verdict(variant, GOLDENS)
    assert verdict["passed"] is False
    failed = {check["name"] for check in verdict["checks"] if not check["passed"]}
    assert failed == {"broad_composite"}


def test_narrow_no_regression_against_baseline():
    baseline = [_row("n01", "narrow", 1, 0.70)]
    ok = [_row("b01", "broad", 4, 0.80), _row("n01", "narrow", 1, 0.68)]
    bad = [_row("b01", "broad", 4, 0.80), _row("n01", "narrow", 6, 0.50)]
    assert gate_verdict(ok, GOLDENS, baseline_rows=baseline)["passed"] is True
    verdict = gate_verdict(bad, GOLDENS, baseline_rows=baseline)
    assert verdict["passed"] is False
    failed = {check["name"] for check in verdict["checks"] if not check["passed"]}
    assert failed == {"narrow_no_regression"}


def test_narrow_check_skipped_without_baseline():
    variant = [_row("b01", "broad", 4, 0.80), _row("n01", "narrow", 1, 0.01)]
    verdict = gate_verdict(variant, GOLDENS)
    assert {check["name"] for check in verdict["checks"]} == {"broad_hops", "broad_composite"}
    assert verdict["passed"] is True


def test_unjudged_variant_fails_rather_than_passes():
    variant = [_row("b01", "broad", 4, None)]
    verdict = gate_verdict(variant, GOLDENS)
    assert verdict["passed"] is False
