"""Retrieval metrics, ported from transcript-lab's `src/evals/ir_metrics.py`.

Deterministic and judge-free on purpose: these say whether the *right documents
came back*, which is the question a retrieval-engine comparison actually turns
on. Answer quality is a separate, LLM-scored concern.

All three metrics take a ranked list of retrieved ids and a set of relevant
ones:

  recall@k  did the relevant items show up at all in the top k
  MRR       how high did the *first* relevant item land (1/rank)
  NDCG@k    the whole ranking's quality, discounting by position

**Scored at video level by default.** transcript-lab's golden set pins both
`expected_video_ids` and `expected_chunk_ids`, but the chunk anchors were
verified against a 23-video corpus that has since grown past 100, and the local
pipeline recreates chunk ids on every re-index. Video ids are stable, so they
carry the verdict; `reanchor_chunk_ids` recovers chunk-level anchors by content
hash where it can, and says plainly when it cannot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """Fraction of relevant items appearing in the top k."""
    wanted = {item for item in relevant if item}
    if not wanted:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    return len(wanted & set(top)) / len(wanted)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant hit; 0.0 if none appear."""
    wanted = {item for item in relevant if item}
    if not wanted:
        return 0.0
    for position, item in enumerate(dict.fromkeys(retrieved), start=1):
        if item in wanted:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """NDCG with binary relevance over the top k.

    The ideal ranking puts every relevant item first, so IDCG sums the discount
    over min(len(relevant), k) positions — that normalisation is what keeps a
    question with 1 expected video comparable to one with 4.
    """
    wanted = {item for item in relevant if item}
    if not wanted:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    dcg = sum(
        1.0 / math.log2(position + 1) for position, item in enumerate(top, 1) if item in wanted
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_case(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> dict[str, float]:
    """All three metrics for one question."""
    return {
        f"recall_at_{k}": recall_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        f"ndcg_at_{k}": ndcg_at_k(retrieved, relevant, k),
    }


def aggregate(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across questions; empty input scores zero, not NaN."""
    if not scores:
        return {}
    keys = sorted({key for score in scores for key in score})
    return {key: sum(score.get(key, 0.0) for score in scores) / len(scores) for key in keys}


def reanchor_chunk_ids(
    expected_chunk_ids: Iterable[str],
    text_sha_by_chunk_id: dict[str, str],
    chunk_key_by_text_sha: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Translate stale golden chunk ids into today's chunk keys, by content.

    Returns `(resolved_keys, unresolved_ids)`. An id resolves when we still know
    the text it pointed at *and* that exact text is still in the corpus; the
    unresolved remainder is what an s09-style confirmation pass has to look at
    rather than silently score as a miss.
    """
    resolved, unresolved = [], []
    for chunk_id in expected_chunk_ids:
        sha = text_sha_by_chunk_id.get(chunk_id)
        key = chunk_key_by_text_sha.get(sha) if sha else None
        if key:
            resolved.append(key)
        else:
            unresolved.append(chunk_id)
    return resolved, unresolved
