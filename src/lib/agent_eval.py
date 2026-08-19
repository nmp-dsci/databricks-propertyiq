"""Deterministic graders for the QA-agent benchmark.

The benchmark's verdict is decided here, not by an LLM judge: each golden
case declares a grader spec, and these functions score an agent's free-text
answer against the confirmed reference. Judges (mlflow.genai) only annotate.

Grader kinds:
  value  — the answer must contain a number within tolerance_pct of expected.
  topk   — the answer must mention at least min_hits of the expected keys
           (postcodes, suburbs, ...).
  refusal — the answer must decline rather than fabricate (for out-of-scope
           questions in the full set).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# $1,350.50 / 6.95% / 5098794 — capture the numeric part, ignore $ , %
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "no data",
    "not available",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not in the data",
    "outside the scope",
    "out of scope",
    "not something i can",
)


def extract_numbers(text: str) -> list[float]:
    """Every number in the text, commas stripped. Dates and postcodes come
    along too — graders compare against an expected value, so unrelated
    numbers only matter if they happen to sit inside the tolerance band."""
    return [float(m.group().replace(",", "")) for m in _NUMBER_RE.finditer(text)]


def value_grade(answer: str, expected: float, tolerance_pct: float = 5.0) -> bool:
    if not answer:
        return False
    band = abs(expected) * tolerance_pct / 100.0
    return any(abs(n - expected) <= band for n in extract_numbers(answer))


def topk_grade(answer: str, expected_keys: list[str], min_hits: int) -> bool:
    if not answer:
        return False
    hits = sum(1 for key in expected_keys if re.search(rf"\b{re.escape(key)}\b", answer))
    return hits >= min_hits


def refusal_grade(answer: str) -> bool:
    return bool(answer) and any(marker in answer.lower() for marker in _REFUSAL_MARKERS)


def grade(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """Score one agent answer against one golden case."""
    spec = case["grader"]
    kind = spec["kind"]
    if kind == "value":
        passed = value_grade(answer, float(spec["expected"]), float(spec.get("tolerance_pct", 5.0)))
    elif kind == "topk":
        passed = topk_grade(answer, list(spec["expected_keys"]), int(spec["min_hits"]))
    elif kind == "refusal":
        passed = refusal_grade(answer)
    else:
        raise ValueError(f"unknown grader kind: {kind!r}")
    return {
        "case_key": case["case_key"],
        "tier": case["tier"],
        "grader_kind": kind,
        "passed": passed,
        "authoring_status": case.get("authoring_status", "draft"),
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        pack = yaml.safe_load(f)
    cases = pack["cases"]
    for case in cases:
        for field in ("case_key", "question", "tier", "grader", "golden_sql"):
            if field not in case:
                raise ValueError(f"case {case.get('case_key', '?')!r} missing {field!r}")
    return cases
