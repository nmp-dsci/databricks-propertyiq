"""Gate arithmetic for the judged eval — pure functions, no I/O.

The s09 plan's success gate, as code the matrix runner and the tests share:

  broad questions   mean hops >= ``min_broad_hops`` AND mean depth-v2
                    composite within ``composite_tolerance`` (relative) of the
                    dev golden broad mean — or above it.
  narrow questions  no regression: mean composite within
                    ``narrow_tolerance`` (absolute) of the A0 baseline's
                    narrow mean. Matching dev's hop count is deliberately NOT
                    required here — a narrow question answered well in one
                    hop is correct behaviour, whatever dev does.

Rows are plain dicts with at least ``category``, and ``composite`` / ``hops``
where scored; a row whose composite is None or NaN (judge failure) is excluded
from means rather than counted as zero, mirroring the judge's own
renormalisation rule.
"""

from __future__ import annotations

import math
from typing import Any

BROAD = "broad"
NARROW = "narrow"

DEFAULT_MIN_BROAD_HOPS = 3.0
DEFAULT_COMPOSITE_TOLERANCE = 0.05
DEFAULT_NARROW_TOLERANCE = 0.05


def _values(rows: list[dict[str, Any]], field: str, category: str | None = None) -> list[float]:
    values = []
    for row in rows:
        if category is not None and row.get("category") != category:
            continue
        value = row.get(field)
        if isinstance(value, (int, float)) and not math.isnan(value):
            values.append(float(value))
    return values


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def aggregate(rows: list[dict[str, Any]], category: str | None = None) -> dict[str, float | None]:
    """Mean composite / hops / metric scores for one stratum (or all rows)."""
    fields = (
        "composite",
        "ragas_v1_composite",
        "hops",
        "unique_videos",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "insight_depth",
        "specificity",
        "coverage",
        "evidence_breadth",
        "calibration",
    )
    out: dict[str, float | None] = {
        "n": float(len([r for r in rows if category is None or r.get("category") == category]))
    }
    for field in fields:
        out[field] = mean(_values(rows, field, category))
    return out


def _check(name: str, passed: bool, actual: float | None, target: str, detail: str) -> dict:
    return {"name": name, "passed": passed, "actual": actual, "target": target, "detail": detail}


def gate_verdict(
    variant_rows: list[dict[str, Any]],
    golden_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]] | None = None,
    min_broad_hops: float = DEFAULT_MIN_BROAD_HOPS,
    composite_tolerance: float = DEFAULT_COMPOSITE_TOLERANCE,
    narrow_tolerance: float = DEFAULT_NARROW_TOLERANCE,
) -> dict[str, Any]:
    """The s09 gate for one variant. ``passed`` only when every check passes.

    ``baseline_rows`` is the A0 run the narrow no-regression check compares
    against; omit it when scoring A0 itself and the check is skipped (A0
    *defines* the baseline).
    """
    checks: list[dict] = []

    broad_hops = mean(_values(variant_rows, "hops", BROAD))
    checks.append(
        _check(
            "broad_hops",
            broad_hops is not None and broad_hops >= min_broad_hops,
            broad_hops,
            f">= {min_broad_hops}",
            "broad questions must actually be researched, not seed-and-answered",
        )
    )

    broad_composite = mean(_values(variant_rows, "composite", BROAD))
    golden_broad = mean(_values(golden_rows, "composite", BROAD))
    if golden_broad is None:
        checks.append(
            _check("broad_composite", False, broad_composite, "n/a", "no judged goldens to compare")
        )
    else:
        floor = round(golden_broad * (1 - composite_tolerance), 4)
        checks.append(
            _check(
                "broad_composite",
                broad_composite is not None and broad_composite >= floor,
                broad_composite,
                f">= {floor} (golden {golden_broad} -{composite_tolerance:.0%})",
                "depth-v2 composite on broad questions vs the dev golden bar",
            )
        )

    if baseline_rows is not None:
        narrow = mean(_values(variant_rows, "composite", NARROW))
        base_narrow = mean(_values(baseline_rows, "composite", NARROW))
        if base_narrow is None:
            checks.append(
                _check(
                    "narrow_no_regression",
                    True,
                    narrow,
                    "n/a",
                    "baseline had no judged narrow rows",
                )
            )
        else:
            floor = round(base_narrow - narrow_tolerance, 4)
            checks.append(
                _check(
                    "narrow_no_regression",
                    narrow is not None and narrow >= floor,
                    narrow,
                    f">= {floor} (A0 narrow {base_narrow} - {narrow_tolerance})",
                    "protocol changes must not damage the questions that were already right",
                )
            )

    return {"passed": all(check["passed"] for check in checks), "checks": checks}
