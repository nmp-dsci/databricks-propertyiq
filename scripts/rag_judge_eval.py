"""Run the question set through one Databricks agent variant and judge it.

The s09 optimisation loop's instrument: builds one agent variant (mode ×
protocol × model), runs all 12 questions through the *local* agent build —
the same `lib.rag_agent` graph the serving endpoint wraps — counts real
retrieval hops via an injected counting retriever, judges every answer with
transcript-lab's own judge (scripts/_judge_bridge.py, decision D1's parity
layer), then records everything three ways:

  * an MLflow run per (variant, mode) in the `rag-agent-evals` experiment,
    with per-question traces tagged variant/mode/question_id;
  * rows in `workspace.rag.judge_scores` for the dashboard;
  * a printed summary + the s09 gate verdict when goldens are available.

Usage:
  uv run python scripts/rag_judge_eval.py --variant A0                    # protocol v1, llama
  uv run python scripts/rag_judge_eval.py --variant A1 --protocol v2
  uv run python scripts/rag_judge_eval.py --variant A2-gptoss --protocol v2 \
      --model databricks-gpt-oss-120b
  uv run python scripts/rag_judge_eval.py --gate A1 --baseline A0         # verdict only
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from lib import rag_agent  # noqa: E402
from lib.judge_gate import aggregate, gate_verdict  # noqa: E402
from lib.rag_questions import QUESTIONS  # noqa: E402

CATALOG = "workspace"
SCHEMA = "rag"
WAREHOUSE_ID = "7f9b6eb116a15acc"
INDEX_NAME = f"{CATALOG}.{SCHEMA}.rag_chunks_gte"
SCORES_TABLE = f"{CATALOG}.{SCHEMA}.judge_scores"
GOLDEN_TABLE = f"{CATALOG}.{SCHEMA}.silver_golden_answers"
DEFAULT_MODEL = "databricks-meta-llama-3-3-70b-instruct"
DEFAULT_SOURCE = REPO.parent / "transcript-rag-agent"
EXPERIMENT = "rag-agent-evals"

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "insight_depth",
    "specificity",
    "coverage",
    "evidence_breadth",
    "calibration",
)


class CountingRetriever:
    """Wraps the real retriever; the hop count IS the retrieval-call count."""

    def __init__(self, retrieve):
        self._retrieve = retrieve
        self.calls = 0
        self.rows: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.calls = 0
        self.rows = []

    def __call__(self, query: str, k: int = rag_agent.TOP_K) -> list[dict[str, Any]]:
        self.calls += 1
        found = self._retrieve(query, k)
        self.rows.extend(found)
        return found


def run_questions(
    agent, counter: CountingRetriever, variant: str, mode: str, questions: list[dict] | None = None
) -> list[dict]:
    rows = []
    for spec in questions or QUESTIONS:
        counter.reset()
        started = time.monotonic()
        error = ""
        answer = ""
        chunks: list[dict[str, Any]] = []
        try:
            result = rag_agent.ask(
                agent,
                str(spec["question"]),
                tags={"variant": variant, "mode": mode, "question_id": str(spec["question_id"])},
            )
            answer = result.get("answer") or ""
            chunks = rag_agent.dedupe_chunks(counter.rows)
        except Exception as exc:  # noqa: BLE001 — one failed question is one row
            error = str(exc)
        rows.append(
            {
                "question_id": spec["question_id"],
                "category": spec["category"],
                "domain": spec["domain"],
                "question": spec["question"],
                "answer": answer,
                "contexts": [chunk.get("text") or "" for chunk in chunks],
                "hops": counter.calls,
                "unique_videos": len({c.get("video_id") for c in chunks if c.get("video_id")}),
                "latency_s": round(time.monotonic() - started, 2),
                "answer_chars": len(answer),
                "error": error,
            }
        )
        state = f"ERROR {error[:60]}" if error else f"{counter.calls} hop(s), {len(chunks)} chunks"
        print(f"  {spec['question_id']} [{mode}]: {state}, {rows[-1]['latency_s']}s")
    return rows


def judge_rows(rows: list[dict], source: Path, answer_model: str, variant_label: str) -> None:
    """Score rows in place through the dev judge bridge (parity layer)."""
    venv_python = source / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise SystemExit(f"{venv_python} not found — run `uv sync` in {source} first")
    # Stable dir keyed on (variant, request content) so a rerun after a kill
    # resumes — while changed answers change the key and judge fresh, never
    # reusing verdicts from a different run of the same variant.
    payload = "".join(
        json.dumps(
            {
                "id": row["question_id"],
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "answer_model": answer_model,
            }
        )
        + "\n"
        for row in rows
        if not row["error"] and row["answer"]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    work = Path(tempfile.gettempdir()) / f"propertyiq_judge_{variant_label}_{digest}"
    work.mkdir(exist_ok=True)
    request_path, scored_path = work / "in.jsonl", work / "out.jsonl"
    request_path.write_text(payload, encoding="utf-8")
    result = subprocess.run(  # noqa: S603 — fixed argv, paths validated above
        [
            str(venv_python),
            str(REPO / "scripts" / "_judge_bridge.py"),
            "--source",
            str(source),
            "--input",
            str(request_path),
            "--output",
            str(scored_path),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"judge bridge exited {result.returncode}")
    scored = {
        record["id"]: record
        for record in (
            json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()
        )
    }
    for row in rows:
        verdict = scored.get(row["question_id"]) or {}
        scores = verdict.get("scores") or {}
        row.update({metric: scores.get(metric) for metric in METRICS})
        row.update(
            {
                "ragas_v1_composite": verdict.get("ragas_v1_composite"),
                "composite": verdict.get("composite"),
                "cap_applied": bool(verdict.get("cap_applied")),
                "grounding_floor_breached": bool(verdict.get("grounding_floor_breached")),
                "judge_model": verdict.get("judge_model") or "",
                "rubric_version": verdict.get("rubric_version") or "",
                "judge_error": verdict.get("error") or "",
            }
        )


def native_evaluate(
    rows: list[dict], config: dict[str, Any], judge_endpoint: str, goldens: list[dict] | None
) -> None:
    """Decision D1's native layer: mlflow.genai.evaluate over the same rows.

    Outputs are precomputed (the rows just ran), so no predict_fn — evaluate
    attaches the native scorers' Feedback to its own run in the same
    experiment, plus built-in Correctness against the dev golden answer when
    one exists for the question.
    """
    import mlflow

    from lib.native_judges import make_native_scorers

    golden_answers: dict[str, str] = {}
    for golden in goldens or []:
        if golden.get("question_id") and golden.get("answer_md"):
            golden_answers[str(golden["question_id"])] = str(golden["answer_md"])

    data = []
    for row in rows:
        if row.get("error") or not row.get("answer"):
            continue
        record: dict[str, Any] = {
            "inputs": {"question": row["question"]},
            "outputs": {"response": row["answer"], "contexts": row["contexts"]},
        }
        expected = golden_answers.get(str(row["question_id"]))
        if expected:
            record["expectations"] = {"expected_response": expected}
        data.append(record)
    if not data:
        print("  native layer: nothing to evaluate")
        return

    scorers = make_native_scorers(judge_endpoint)
    if golden_answers:
        from mlflow.genai.scorers import Correctness

        scorers.append(Correctness(model=f"databricks:/{judge_endpoint}"))

    results = mlflow.genai.evaluate(data=data, scorers=scorers)
    print(f"  native evaluate run {results.run_id}")
    for name, value in sorted((results.metrics or {}).items()):
        print(f"    {name}: {value}")


def log_mlflow(rows: list[dict], config: dict[str, Any]) -> str:
    import mlflow
    from databricks.sdk import WorkspaceClient

    user = WorkspaceClient().current_user.me().user_name
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/Users/{user}/{EXPERIMENT}")
    with mlflow.start_run(run_name=f"{config['variant']}-{config['mode']}") as run:
        mlflow.log_params(config)
        for stratum in ("broad", "narrow", None):
            label = stratum or "all"
            summary = aggregate(rows, stratum)
            for field in ("composite", "hops", "faithfulness", "coverage", "evidence_breadth"):
                value = summary.get(field)
                if value is not None:
                    mlflow.log_metric(f"{label}_{field}", value)
        path = Path(tempfile.mkdtemp(prefix="judge_rows_")) / "rows.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        mlflow.log_artifact(str(path))
        return run.info.run_id


def insert_scores(rows: list[dict], config: dict[str, Any], run_id: str) -> None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementParameterListItem as Param

    w = WorkspaceClient()
    _execute(
        w,
        f"""
        CREATE TABLE IF NOT EXISTS {SCORES_TABLE} (
          run_at STRING, run_id STRING, variant STRING, mode STRING, protocol STRING,
          model_endpoint STRING, question_id STRING, category STRING, domain STRING,
          hops INT, unique_videos INT,
          faithfulness DOUBLE, answer_relevancy DOUBLE, context_precision DOUBLE,
          insight_depth DOUBLE, specificity DOUBLE, coverage DOUBLE,
          evidence_breadth DOUBLE, calibration DOUBLE,
          ragas_v1_composite DOUBLE, composite DOUBLE,
          cap_applied BOOLEAN, grounding_floor_breached BOOLEAN,
          judge_model STRING, rubric_version STRING,
          latency_s DOUBLE, answer_chars INT, judge_error STRING, error STRING
        )
        COMMENT 'Judged agent-variant runs (s09): every Databricks variant scored by
        transcript-lab''s own RAGAS+depth judge, with real hop counts from the
        counting retriever. Compare against silver_golden_answers (the dev bar).'
        """,
    )
    insert = f"""
        INSERT INTO {SCORES_TABLE} VALUES
        (:run_at, :run_id, :variant, :mode, :protocol, :model_endpoint, :question_id,
         :category, :domain, :hops, :unique_videos, :faithfulness, :answer_relevancy,
         :context_precision, :insight_depth, :specificity, :coverage, :evidence_breadth,
         :calibration, :ragas_v1_composite, :composite, :cap_applied,
         :grounding_floor_breached, :judge_model, :rubric_version, :latency_s,
         :answer_chars, :judge_error, :error)
    """
    run_at = datetime.now(UTC).isoformat()

    def _num(row: dict, field: str) -> Param:
        value = row.get(field)
        return Param(name=field, value=None if value is None else str(value), type="DOUBLE")

    for row in rows:
        parameters = [
            Param(name="run_at", value=run_at),
            Param(name="run_id", value=run_id),
            Param(name="variant", value=config["variant"]),
            Param(name="mode", value=config["mode"]),
            Param(name="protocol", value=config["protocol"]),
            Param(name="model_endpoint", value=config["model_endpoint"]),
            Param(name="question_id", value=row["question_id"]),
            Param(name="category", value=row["category"]),
            Param(name="domain", value=row["domain"]),
            Param(name="hops", value=str(row.get("hops") or 0), type="INT"),
            Param(name="unique_videos", value=str(row.get("unique_videos") or 0), type="INT"),
            *[_num(row, metric) for metric in METRICS],
            _num(row, "ragas_v1_composite"),
            _num(row, "composite"),
            Param(
                name="cap_applied", value=str(bool(row.get("cap_applied"))).lower(), type="BOOLEAN"
            ),
            Param(
                name="grounding_floor_breached",
                value=str(bool(row.get("grounding_floor_breached"))).lower(),
                type="BOOLEAN",
            ),
            Param(name="judge_model", value=row.get("judge_model") or ""),
            Param(name="rubric_version", value=row.get("rubric_version") or ""),
            _num(row, "latency_s"),
            Param(name="answer_chars", value=str(row.get("answer_chars") or 0), type="INT"),
            Param(name="judge_error", value=row.get("judge_error") or ""),
            Param(name="error", value=row.get("error") or ""),
        ]
        _execute(w, insert, parameters)
    print(f"  {len(rows)} row(s) -> {SCORES_TABLE}")


def _execute(w, statement: str, parameters: list | None = None):
    result = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID, statement=statement, parameters=parameters, wait_timeout="50s"
    )
    deadline = time.time() + 300
    while result.status.state.value in ("PENDING", "RUNNING") and time.time() < deadline:
        time.sleep(5)
        result = w.statement_execution.get_statement(result.statement_id)
    if result.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"statement {result.status.state.value}: {result.status.error}")
    return result


def _rows_from_result(result) -> list[dict[str, Any]]:
    columns = [c.name for c in result.manifest.schema.columns]
    data = result.result.data_array or [] if result.result else []
    out = []
    for raw in data:
        row = dict(zip(columns, raw, strict=False))
        for key, value in row.items():
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    row[key] = float(value)
        out.append(row)
    return out


def fetch_variant_rows(w, variant: str, mode: str = "agentic") -> list[dict[str, Any]]:
    """The most recent run's rows for one variant, straight from UC."""
    from databricks.sdk.service.sql import StatementParameterListItem as Param

    result = _execute(
        w,
        f"""
        SELECT question_id, category, hops, composite FROM {SCORES_TABLE}
        WHERE variant = :variant AND mode = :mode
          AND run_at = (SELECT max(run_at) FROM {SCORES_TABLE}
                        WHERE variant = :variant AND mode = :mode)
        """,
        [Param(name="variant", value=variant), Param(name="mode", value=mode)],
    )
    return _rows_from_result(result)


def fetch_golden_rows(w) -> list[dict[str, Any]]:
    result = _execute(
        w,
        f"SELECT question_id, category, iterations AS hops, composite FROM {GOLDEN_TABLE} "
        "WHERE is_current",
    )
    return _rows_from_result(result)


def golden_rows_full(w) -> list[dict[str, Any]]:
    """Goldens with the reference answers — what Correctness grades against."""
    try:
        result = _execute(
            w,
            f"SELECT question_id, category, answer_md, composite FROM {GOLDEN_TABLE} "
            "WHERE is_current",
        )
        return _rows_from_result(result)
    except Exception:  # noqa: BLE001 — goldens not landed yet; Correctness just drops out
        return []


def print_summary(rows: list[dict], golden_rows: list[dict] | None) -> None:
    print(f"\n{'':<6} {'cat':<7} {'hops':>4} {'videos':>6} {'faith':>6} {'cover':>6} {'comp':>6}")
    for row in rows:
        comp = row.get("composite")
        print(
            f"{row['question_id']:<6} {row['category']:<7} {row.get('hops', 0):>4} "
            f"{row.get('unique_videos', 0):>6} "
            f"{_fmt(row.get('faithfulness')):>6} {_fmt(row.get('coverage')):>6} {_fmt(comp):>6}"
        )
    for stratum in ("broad", "narrow"):
        summary = aggregate(rows, stratum)
        print(
            f"  mean [{stratum}]: hops {summary['hops']}, composite {summary['composite']}, "
            f"faithfulness {summary['faithfulness']}, coverage {summary['coverage']}"
        )
    if golden_rows:
        for stratum in ("broad", "narrow"):
            summary = aggregate(golden_rows, stratum)
            print(f"  golden [{stratum}]: hops {summary['hops']}, composite {summary['composite']}")


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def print_gate(variant: str, baseline: str | None) -> None:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    variant_rows = fetch_variant_rows(w, variant)
    golden_rows = fetch_golden_rows(w)
    baseline_rows = fetch_variant_rows(w, baseline) if baseline else None
    verdict = gate_verdict(variant_rows, golden_rows, baseline_rows=baseline_rows)
    print(f"\ngate verdict for {variant}: {'PASSED' if verdict['passed'] else 'FAILED'}")
    for check in verdict["checks"]:
        flag = "ok " if check["passed"] else "FAIL"
        print(f"  [{flag}] {check['name']}: {check['actual']} (target {check['target']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", help="label recorded with the run, e.g. A0, A1, A2-gptoss")
    parser.add_argument("--protocol", default="v1", help="agentic decide protocol (v1|v2)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", default="agentic")
    parser.add_argument(
        "--questions",
        default="",
        help="comma-separated question_id subset (default: all 12); smokes and re-runs",
    )
    parser.add_argument("--max-hops", type=int, default=None)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-uc", action="store_true")
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument(
        "--native",
        action="store_true",
        help="also run mlflow.genai.evaluate with the FMAPI native judges (D1 layer 2)",
    )
    parser.add_argument("--judge-model", default="databricks-gpt-oss-120b")
    parser.add_argument("--gate", help="print the s09 gate verdict for this variant and exit")
    parser.add_argument("--baseline", help="baseline variant for the narrow no-regression check")
    args = parser.parse_args()

    if args.gate:
        print_gate(args.gate, args.baseline)
        return
    if not args.variant:
        parser.error("--variant is required (or use --gate)")

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    llm = rag_agent.make_fmapi_llm(args.model, workspace_client=w)
    counter = CountingRetriever(rag_agent.make_retriever(INDEX_NAME, workspace_client=w))

    golden_rows: list[dict] | None = None
    # Goldens may not be landed yet; the gate print just goes quiet without them.
    with contextlib.suppress(Exception):
        golden_rows = fetch_golden_rows(w)

    selected = QUESTIONS
    if args.questions:
        wanted = {q.strip() for q in args.questions.split(",") if q.strip()}
        selected = [spec for spec in QUESTIONS if spec["question_id"] in wanted]
        missing = wanted - {str(spec["question_id"]) for spec in selected}
        if missing:
            parser.error(f"unknown question id(s): {', '.join(sorted(missing))}")

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        config = {
            "variant": args.variant,
            "mode": mode,
            "protocol": args.protocol,
            "model_endpoint": args.model,
            "max_hops": args.max_hops or "",
            "index": INDEX_NAME,
            "top_k": rag_agent.TOP_K,
            "questions": len(selected),
        }
        print(f"\n=== {args.variant} · {mode} · {args.protocol} · {args.model} ===")
        agent = rag_agent.build_agent(
            llm, counter, mode=mode, protocol=args.protocol, max_hops=args.max_hops
        )
        rows = run_questions(agent, counter, args.variant, mode, questions=selected)
        if not args.skip_judge:
            print(f"judging {len(rows)} answer(s) with the dev judge ...")
            judge_rows(
                rows,
                Path(args.source),
                answer_model=args.model,
                variant_label=f"{args.variant}_{mode}",
            )
        run_id = ""
        if not args.skip_mlflow:
            run_id = log_mlflow(rows, config)
            print(f"  mlflow run {run_id}")
        if not args.skip_uc and not args.skip_judge:
            insert_scores(rows, config, run_id)
        if args.native:
            print("native judge layer (mlflow.genai.evaluate) ...")
            native_evaluate(rows, config, args.judge_model, golden_rows_full(w))
        print_summary(rows, golden_rows)
        if mode == "agentic" and golden_rows and not args.skip_judge:
            baseline_rows = None
            if args.baseline:
                with contextlib.suppress(Exception):
                    baseline_rows = fetch_variant_rows(w, args.baseline)
            verdict = gate_verdict(rows, golden_rows, baseline_rows=baseline_rows)
            print(f"\ngate: {'PASSED' if verdict['passed'] else 'FAILED'}")
            for check in verdict["checks"]:
                flag = "ok " if check["passed"] else "FAIL"
                print(f"  [{flag}] {check['name']}: {check['actual']} (target {check['target']})")


if __name__ == "__main__":
    main()
