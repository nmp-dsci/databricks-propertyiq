# Databricks notebook source
# MAGIC %md
# MAGIC # rag 05 · Agent eval — LLM-as-judge over the LIVE endpoint, in-workspace
# MAGIC
# MAGIC The recurring MLOps loop for `rag_transcript_agent`, running entirely on
# MAGIC Databricks (s09 close-out): every question in `silver_golden_answers` is
# MAGIC asked of the **deployed serving endpoint** — the thing users actually
# MAGIC talk to, not a laptop rebuild — and the answers are judged by an FMAPI
# MAGIC model through `ai_query`, in SQL, on the warehouse-less serverless
# MAGIC session. Nothing here needs pip, egress, or a sibling repo:
# MAGIC
# MAGIC * **behaviour** — hops parsed from the agent's own decision log, unique
# MAGIC   videos from its cited chunk keys: did it research or seed-and-answer?
# MAGIC * **quality** — groundedness / relevance / five depth metrics +
# MAGIC   correctness vs the dev golden answer, each an `ai_query` judge call
# MAGIC   returning strict JSON;
# MAGIC * **record** — rows appended to `judge_scores` (rubric `native-v1`, so
# MAGIC   they are never confused with the laptop parity judge's `depth-v2`
# MAGIC   rows) feeding the rag_eval dashboard's Judge page.
# MAGIC
# MAGIC The one thing that stays outside the workspace is golden *capture*: the
# MAGIC dev RAGAS judge needs DeepSeek + ragas, which Free Edition serverless
# MAGIC cannot reach (no egress, no pip). Scoring the deployed agent does not.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "rag")
dbutils.widgets.text("endpoint", "agents_workspace-rag-rag_transcript_agent")
dbutils.widgets.text("mode", "agentic")
dbutils.widgets.text("variant", "live-react")
dbutils.widgets.text("protocol", "react")
dbutils.widgets.text("judge_endpoint", "databricks-gpt-oss-120b")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
endpoint = dbutils.widgets.get("endpoint")
mode = dbutils.widgets.get("mode")
variant = dbutils.widgets.get("variant")
protocol = dbutils.widgets.get("protocol")
judge_endpoint = dbutils.widgets.get("judge_endpoint")
prefix = f"{catalog}.{schema}"

# COMMAND ----------

import json
import re
import time
from datetime import datetime, timezone

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
run_at = datetime.now(timezone.utc).isoformat()

questions = [
    row.asDict()
    for row in spark.sql(
        f"""SELECT question_id, category, domain, question, answer_md AS golden_answer
            FROM {prefix}.silver_golden_answers WHERE is_current
            ORDER BY question_id"""
    ).collect()
]
print(f"{len(questions)} questions, endpoint {endpoint}, mode {mode}")

# COMMAND ----------


def ask_endpoint(question: str) -> dict:
    """One question through the deployed ChatAgent, with its custom outputs."""
    started = time.monotonic()
    response = w.api_client.do(
        "POST",
        f"/serving-endpoints/{endpoint}/invocations",
        body={
            "messages": [{"role": "user", "content": question}],
            "custom_inputs": {"mode": mode},
        },
    )
    latency = round(time.monotonic() - started, 2)
    answer = ""
    for message in reversed(response.get("messages") or []):
        if message.get("role") == "assistant" and message.get("content"):
            answer = str(message["content"])
            break
    custom = response.get("custom_outputs") or {}
    log = [str(line) for line in custom.get("decision_log") or []]
    # The react loop logs every batch as "tool: +X excerpts across N call(s)";
    # the decide protocols log one "search:" line per hop. Count both shapes so
    # the same eval scores any deployed protocol.
    hops = sum(int(m) for line in log for m in re.findall(r"across (\d+) call", line))
    hops += sum(1 for line in log if line.startswith("search:"))
    hops += sum(1 for line in log if line.startswith("retrieve:"))
    return {
        "answer": answer,
        "chunk_keys": [str(key) for key in custom.get("chunk_keys") or []],
        "video_ids": [str(v) for v in custom.get("video_ids") or [] if v],
        "hops": hops,
        "latency_s": latency,
        "decision_log": log,
    }


results = []
for spec in questions:
    try:
        result = ask_endpoint(spec["question"])
        error = ""
    except Exception as exc:  # noqa: BLE001 — one failed question is one row
        result = {"answer": "", "chunk_keys": [], "video_ids": [], "hops": 0, "latency_s": 0.0}
        error = str(exc)[:500]
    results.append({**spec, **result, "error": error})
    print(
        f"  {spec['question_id']}: {result['hops']} hop(s), "
        f"{len(result['video_ids'])} video(s), {result['latency_s']}s"
        + (f" ERROR {error[:60]}" if error else "")
    )

# COMMAND ----------

# Persist answers (with contexts resolved from the cited chunk keys) so the
# judging step is plain SQL and re-runnable without re-asking the endpoint.
from pyspark.sql import functions as F  # noqa: E402

answers_df = spark.createDataFrame(
    [
        {
            "run_at": run_at,
            "variant": variant,
            "mode": mode,
            "protocol": protocol,
            "endpoint": endpoint,
            "question_id": r["question_id"],
            "category": r["category"],
            "domain": r["domain"],
            "question": r["question"],
            "golden_answer": r["golden_answer"] or "",
            "answer": r["answer"],
            "chunk_keys": r["chunk_keys"],
            "unique_videos": len(r["video_ids"]),
            "hops": r["hops"],
            "latency_s": float(r["latency_s"]),
            "error": r["error"],
        }
        for r in results
    ]
)

chunks = spark.table(f"{prefix}.silver_chunks").filter("is_current").select("chunk_key", "text")
contexts = (
    answers_df.select("question_id", F.explode_outer("chunk_keys").alias("chunk_key"))
    .join(chunks, "chunk_key", "left")
    .groupBy("question_id")
    # Judge context budget 16 — the same cap the laptop parity judge applies,
    # so neither judge reads unbounded evidence.
    .agg(F.slice(F.collect_list("text"), 1, 16).alias("context_texts"))
)
answers_df = answers_df.join(contexts, "question_id", "left")
answers_df.write.mode("append").saveAsTable(f"{prefix}.agent_eval_answers")
print(f"answers -> {prefix}.agent_eval_answers @ {run_at}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Judge with `ai_query` — three judge calls per answer, pure SQL

# COMMAND ----------

GROUNDEDNESS = (
    "You grade whether an ANSWER is supported by the CONTEXTS it cites. Score 1.0 when every "
    "factual claim appears in some context; subtract proportionally per unsupported claim. "
    'Reply with JSON only: {"score": <0.0-1.0>, "rationale": "<one sentence>"}'
)
RELEVANCE_AND_CORRECTNESS = (
    "You grade an ANSWER against the QUESTION and a REFERENCE answer. relevance: does it "
    "address the question directly? correctness: does it agree with the reference on the "
    "facts they both cover (extra grounded detail is not a fault)? Reply with JSON only: "
    '{"relevance": {"score": <0.0-1.0>, "rationale": "<one sentence>"}, '
    '"correctness": {"score": <0.0-1.0>, "rationale": "<one sentence>"}}'
)
DEPTH = (
    "You grade how much an ANSWER is WORTH given the CONTEXTS, on five metrics — "
    "insight_depth (synthesises across sources vs restates one), specificity (concrete, "
    "checkable claims), coverage (distinct facets of the question represented), "
    "evidence_breadth (how much of the context evidence is used), calibration (claims only "
    "what the evidence supports). Length is not depth. Reply with JSON only: "
    '{"insight_depth": {"score": <0.0-1.0>}, "specificity": {"score": <0.0-1.0>}, '
    '"coverage": {"score": <0.0-1.0>}, "evidence_breadth": {"score": <0.0-1.0>}, '
    '"calibration": {"score": <0.0-1.0>}}'
)

spark.sql(f"""
CREATE OR REPLACE TEMP VIEW judged AS
WITH latest AS (
  SELECT *, concat_ws('\\n\\n---\\n\\n', context_texts) AS contexts_block
  FROM {prefix}.agent_eval_answers
  WHERE run_at = '{run_at}' AND error = '' AND answer != ''
),
verdicts AS (
  SELECT *,
    ai_query('{judge_endpoint}',
      concat('{GROUNDEDNESS}', '\\n\\nQUESTION: ', question,
             '\\n\\nCONTEXTS:\\n', contexts_block, '\\n\\nANSWER:\\n', answer)) AS g_raw,
    ai_query('{judge_endpoint}',
      concat('{RELEVANCE_AND_CORRECTNESS}', '\\n\\nQUESTION: ', question,
             '\\n\\nREFERENCE:\\n', golden_answer, '\\n\\nANSWER:\\n', answer)) AS rc_raw,
    ai_query('{judge_endpoint}',
      concat('{DEPTH}', '\\n\\nQUESTION: ', question,
             '\\n\\nCONTEXTS:\\n', contexts_block, '\\n\\nANSWER:\\n', answer)) AS d_raw
  FROM latest
)
SELECT *,
  from_json(regexp_extract(g_raw, '\\\\{{.*\\\\}}', 0), 'score DOUBLE, rationale STRING') AS g,
  from_json(regexp_extract(rc_raw, '\\\\{{.*\\\\}}', 0),
    'relevance STRUCT<score DOUBLE, rationale STRING>, correctness STRUCT<score DOUBLE, rationale STRING>') AS rc,
  from_json(regexp_extract(d_raw, '\\\\{{.*\\\\}}', 0),
    'insight_depth STRUCT<score DOUBLE>, specificity STRUCT<score DOUBLE>, coverage STRUCT<score DOUBLE>, evidence_breadth STRUCT<score DOUBLE>, calibration STRUCT<score DOUBLE>') AS d
FROM verdicts
""")

# COMMAND ----------

# Same table the laptop harness writes, different rubric label: native-v1 is
# the in-workspace judge (gpt-oss via ai_query), never comparable 1:1 with the
# parity judge's depth-v2 numbers — the dashboard separates them by rubric.
spark.sql(f"""
INSERT INTO {prefix}.judge_scores
SELECT
  run_at, '' AS run_id, variant, mode, protocol,
  '{judge_endpoint}' AS model_endpoint,
  question_id, category, domain, hops, unique_videos,
  g.score AS faithfulness,
  rc.relevance.score AS answer_relevancy,
  CAST(NULL AS DOUBLE) AS context_precision,
  d.insight_depth.score AS insight_depth,
  d.specificity.score AS specificity,
  d.coverage.score AS coverage,
  d.evidence_breadth.score AS evidence_breadth,
  d.calibration.score AS calibration,
  rc.correctness.score AS ragas_v1_composite,
  round(coalesce(g.score, 0) * 0.25 + coalesce(rc.relevance.score, 0) * 0.15
      + coalesce(d.insight_depth.score, 0) * 0.20 + coalesce(d.specificity.score, 0) * 0.10
      + coalesce(d.coverage.score, 0) * 0.10 + coalesce(d.evidence_breadth.score, 0) * 0.15
      + coalesce(d.calibration.score, 0) * 0.05, 4) AS composite,
  false AS cap_applied,
  coalesce(g.score, 0) < 0.6 AS grounding_floor_breached,
  '{judge_endpoint}' AS judge_model,
  'native-v1' AS rubric_version,
  latency_s,
  length(answer) AS answer_chars,
  '' AS judge_error,
  '' AS error
FROM judged
""")
print("scores -> judge_scores (rubric native-v1)")

# COMMAND ----------

summary = spark.sql(f"""
SELECT category,
       count(*) AS n,
       round(avg(hops), 2) AS hops,
       round(avg(unique_videos), 1) AS videos,
       round(avg(faithfulness), 3) AS groundedness,
       round(avg(coverage), 3) AS coverage,
       round(avg(ragas_v1_composite), 3) AS correctness_vs_golden,
       round(avg(composite), 3) AS composite
FROM {prefix}.judge_scores
WHERE run_at = '{run_at}'
GROUP BY category ORDER BY category
""")
display(summary)

golden = spark.sql(f"""
SELECT category, round(avg(iterations), 1) AS dev_hops
FROM {prefix}.silver_golden_answers WHERE is_current GROUP BY category
""")
display(golden)

rows = {r["category"]: r for r in summary.collect()}
broad_hops = rows.get("broad", {})["hops"] if "broad" in rows else 0
assert broad_hops and broad_hops > 1.5, (
    f"broad questions averaged {broad_hops} hops — the deployed agent is not "
    "researching; check RAG_PROTOCOL / the served model"
)
print(f"OK: deployed agent researches (broad mean {broad_hops} hops)")
