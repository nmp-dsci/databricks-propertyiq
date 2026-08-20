"""Databricks-native LLM judges — mlflow.genai scorers on an FMAPI model.

Decision D1's second layer (plan s09): the parity verdict comes from the dev
project's own RAGAS judge via scripts/_judge_bridge.py, but assessments that
live *on the platform* — attached to MLflow evaluation runs, visible next to
traces, queryable without a sibling repo — come from these scorers. They
mirror the rubric's shape (grounding + the five depth metrics) without
claiming score-parity with it: a different judge model reading a different
prompt produces a different number, which is exactly why the gate never mixes
the two layers.

The judge model is an FMAPI pay-per-token endpoint (default gpt-oss-120b,
proven live in P0), so nothing here needs a serving slot, an external API key,
or anything beyond the workspace itself.
"""

from __future__ import annotations

import json
from typing import Any

from lib.rag_agent import make_fmapi_llm

DEFAULT_JUDGE_ENDPOINT = "databricks-gpt-oss-120b"

# The same judge-time context budget scripts/_judge_bridge.py applies, so both
# judge layers read the same evidence.
MAX_CONTEXTS = 16
MAX_CONTEXT_CHARS = 800

DEPTH_METRICS = ("insight_depth", "specificity", "coverage", "evidence_breadth", "calibration")

GROUNDEDNESS_PROMPT = """\
You grade whether an ANSWER is supported by the CONTEXTS it cites.
Score 1.0 when every factual claim in the answer appears in some context;
subtract proportionally for each unsupported claim. Quoting is not required —
paraphrase counts as support. Ignore style entirely.

Reply with JSON only: {{"score": <0.0-1.0>, "rationale": "<one sentence>"}}

QUESTION: {question}

CONTEXTS:
{contexts}

ANSWER:
{answer}"""

RELEVANCE_PROMPT = """\
You grade whether an ANSWER actually addresses the QUESTION asked.
Score 1.0 when it answers the question directly and completely; lower when it
answers something adjacent, hedges without content, or buries the answer.
Groundedness is NOT your concern here — only responsiveness.

Reply with JSON only: {{"score": <0.0-1.0>, "rationale": "<one sentence>"}}

QUESTION: {question}

ANSWER:
{answer}"""

DEPTH_PROMPT = """\
You grade how much an ANSWER is WORTH given the CONTEXTS, on five metrics:
- insight_depth: does it synthesise across sources, or restate one? 1.0 = it
  connects and contrasts; 0.0 = a paraphrase of a single context.
- specificity: are its claims concrete and checkable (names, numbers, steps)?
- coverage: how many distinct facets of the question does it represent?
- evidence_breadth: how much of the available context evidence does it use?
- calibration: does it claim only what the evidence supports, flagging gaps?
Length is not depth — a long repetitive answer scores low on insight_depth.

Reply with JSON only:
{{"insight_depth": {{"score": <0.0-1.0>, "rationale": "<one sentence>"}},
  "specificity": {{"score": <0.0-1.0>, "rationale": "<one sentence>"}},
  "coverage": {{"score": <0.0-1.0>, "rationale": "<one sentence>"}},
  "evidence_breadth": {{"score": <0.0-1.0>, "rationale": "<one sentence>"}},
  "calibration": {{"score": <0.0-1.0>, "rationale": "<one sentence>"}}}}

QUESTION: {question}

CONTEXTS ({context_count} shown):
{contexts}

ANSWER:
{answer}"""


def parse_json_verdict(raw: str) -> dict[str, Any]:
    """The judge's JSON, tolerating fences, prose and stray text around it."""
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {}


def _score(verdict: dict[str, Any], key: str | None = None) -> float | None:
    node = verdict.get(key) if key else verdict
    if not isinstance(node, dict):
        return None
    value = node.get("score")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


def _rationale(verdict: dict[str, Any], key: str | None = None) -> str:
    node = verdict.get(key) if key else verdict
    if isinstance(node, dict):
        return str(node.get("rationale") or "")
    return ""


def format_contexts(contexts: list[str]) -> str:
    trimmed = [str(c)[:MAX_CONTEXT_CHARS] for c in contexts[:MAX_CONTEXTS] if str(c).strip()]
    if not trimmed:
        return "(none)"
    return "\n\n---\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(trimmed))


def make_native_scorers(
    model_endpoint: str = DEFAULT_JUDGE_ENDPOINT,
    llm=None,
    workspace_client=None,
) -> list:
    """The three native judges as mlflow.genai scorers.

    ``llm`` is injectable for tests (messages -> str); by default it is the
    FMAPI endpoint via the same wrapper the agent itself uses, reasoning-model
    content normalisation included.
    """
    from mlflow.entities import Feedback
    from mlflow.genai.scorers import scorer

    judge = llm or make_fmapi_llm(model_endpoint, workspace_client=workspace_client)

    def _ask_judge(prompt: str) -> dict[str, Any]:
        return parse_json_verdict(judge([{"role": "user", "content": prompt}]))

    @scorer
    def native_groundedness(inputs: dict, outputs: dict) -> Feedback:
        verdict = _ask_judge(
            GROUNDEDNESS_PROMPT.format(
                question=inputs.get("question", ""),
                contexts=format_contexts(outputs.get("contexts") or []),
                answer=outputs.get("response", ""),
            )
        )
        return Feedback(
            name="native_groundedness",
            value=_score(verdict) if _score(verdict) is not None else "unscored",
            rationale=_rationale(verdict),
        )

    @scorer
    def native_relevance(inputs: dict, outputs: dict) -> Feedback:
        verdict = _ask_judge(
            RELEVANCE_PROMPT.format(
                question=inputs.get("question", ""),
                answer=outputs.get("response", ""),
            )
        )
        return Feedback(
            name="native_relevance",
            value=_score(verdict) if _score(verdict) is not None else "unscored",
            rationale=_rationale(verdict),
        )

    @scorer
    def native_depth(inputs: dict, outputs: dict) -> list[Feedback]:
        contexts = outputs.get("contexts") or []
        verdict = _ask_judge(
            DEPTH_PROMPT.format(
                question=inputs.get("question", ""),
                context_count=min(len(contexts), MAX_CONTEXTS),
                contexts=format_contexts(contexts),
                answer=outputs.get("response", ""),
            )
        )
        feedbacks = []
        for metric in DEPTH_METRICS:
            value = _score(verdict, metric)
            feedbacks.append(
                Feedback(
                    name=f"native_{metric}",
                    value=value if value is not None else "unscored",
                    rationale=_rationale(verdict, metric),
                )
            )
        return feedbacks

    return [native_groundedness, native_relevance, native_depth]
