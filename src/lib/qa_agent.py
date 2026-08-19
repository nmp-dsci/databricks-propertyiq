"""The PropertyIQ QA agent — Data Pilot's data-agent ported to Databricks.

Same graph as data-qa-agent's agent (route -> plan_sql -> execute -> reflect
-> answer, with a decision log throughout), re-grounded on the Unity Catalog
gold tables instead of the dbt manifest and executing on the serverless
warehouse instead of Postgres.

The graph is dependency-injected: `build_agent(llm, run_sql)` takes an LLM
callable (messages -> str) and a SQL executor (sql -> list[dict] | raises),
so unit tests script both and never touch a network. `make_databricks_agent`
wires the real ChatDatabricks + Statement Execution API pair.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

import mlflow
from langgraph.graph import END, StateGraph

MAX_SQL_RETRIES = 1
MAX_RESULT_ROWS = 50

# The grounding rules ported from the Genie space instructions and
# data-qa-agent's dbt grounding — the part of the port that makes or breaks
# the trap questions.
GROUNDING_RULES = """\
You answer questions about NSW property data using exactly three tables:

{catalog}.{schema}.gold_property_rent
  (postcode, property_type, bedroom_band, month, total_weekly_rent, n_rented,
   avg_weekly_rent, median_weekly_rent, min_weekly_rent, max_weekly_rent)
  Monthly rental-bond aggregates per postcode / property_type / bedroom_band.

{catalog}.{schema}.gold_property_sales
  (postcode, suburb, property_type, area_band, zoning, month, total_sale_value,
   n_sold, avg_sale_price, median_sale_price, min_sale_price, max_sale_price)
  Monthly sale aggregates per postcode / suburb / property_type / area_band / zoning.

{catalog}.{schema}.gold_property_yield
  (postcode, property_type, month, total_sale_value, n_sold, total_weekly_rent,
   n_rented, avg_sale_price, avg_weekly_rent, gross_yield_pct)
  Monthly gross rental yield per postcode / property_type.

Hard rules — never violate these:
1. Averages across groups MUST be re-aggregated from the additive columns:
   SUM(total_weekly_rent)/SUM(n_rented) or SUM(total_sale_value)/SUM(n_sold).
   Never average the avg_* or median_* columns across rows.
2. Rankings MUST apply a volume floor (e.g. HAVING SUM(n_sold) >= 30 for
   annual sales rankings, n_rented >= 10 for rent) so thin cells don't win.
3. The rent table has NO suburb column. Rent and sales join on postcode,
   property_type and month only. Never invent a suburb-level rent join.
4. property_type values are lowercase: 'house', 'unit'.
5. "Right now" / "current" means the latest month present in the table.
6. Generate exactly ONE read-only SELECT statement. No DDL, DML or comments.
7. Ranking questions ("which postcodes...") return the top 5-10 rows, and the
   answer lists them — never just the single best.
"""


class AgentState(TypedDict, total=False):
    question: str
    route: str  # "answer" | "refuse"
    sql: str
    rows: list[dict[str, Any]]
    error: str
    retries: int
    answer: str
    decision_log: Annotated[list[str], lambda a, b: a + b]


def ensure_select_only(sql: str) -> str:
    """The same guardrail Data Pilot applies before execution."""
    cleaned = re.sub(r"^\s*```(?:sql)?|```\s*$", "", sql.strip(), flags=re.MULTILINE).strip()
    cleaned = cleaned.rstrip(";").strip()
    if not re.match(r"(?is)^\s*(select|with)\b", cleaned):
        raise ValueError(f"only SELECT statements are allowed, got: {cleaned[:80]!r}")
    forbidden = re.compile(
        r"(?is)\b(insert|update|delete|merge|drop|alter|create|grant|revoke|truncate)\b"
    )
    if forbidden.search(cleaned):
        raise ValueError("statement contains a non-read-only keyword")
    if ";" in cleaned:
        raise ValueError("multiple statements are not allowed")
    return cleaned


def build_agent(
    llm: Callable[[list[dict[str, str]]], str],
    run_sql: Callable[[str], list[dict[str, Any]]],
    catalog: str = "workspace",
    schema: str = "propertyiq",
):
    """Compile the LangGraph graph with injected LLM and SQL executor."""
    grounding = GROUNDING_RULES.format(catalog=catalog, schema=schema)

    def route(state: AgentState) -> AgentState:
        verdict = llm(
            [
                {"role": "system", "content": grounding},
                {
                    "role": "user",
                    "content": (
                        "Can the question below be answered from these three tables? "
                        "Reply with exactly one word: ANSWER or REFUSE.\n\n"
                        f"Question: {state['question']}"
                    ),
                },
            ]
        )
        decision = "answer" if "ANSWER" in verdict.upper() else "refuse"
        return {"route": decision, "decision_log": [f"route: {decision}"]}

    def plan_sql(state: AgentState) -> AgentState:
        messages = [
            {"role": "system", "content": grounding},
            {
                "role": "user",
                "content": (
                    "Write the single SELECT statement that answers this question. "
                    "Return only SQL, no prose.\n\n"
                    f"Question: {state['question']}"
                ),
            },
        ]
        if state.get("error"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous SQL failed with: {state['error']}\n"
                        f"Previous SQL: {state.get('sql', '')}\nFix it and return only SQL."
                    ),
                }
            )
        raw = llm(messages)
        try:
            sql = ensure_select_only(raw)
            return {"sql": sql, "error": "", "decision_log": [f"plan_sql: {sql[:120]}"]}
        except ValueError as exc:
            return {"sql": "", "error": str(exc), "decision_log": [f"plan_sql rejected: {exc}"]}

    def execute(state: AgentState) -> AgentState:
        if not state.get("sql"):
            return {"rows": [], "decision_log": ["execute: skipped, no valid SQL"]}
        try:
            rows = run_sql(state["sql"])[:MAX_RESULT_ROWS]
            return {"rows": rows, "error": "", "decision_log": [f"execute: {len(rows)} rows"]}
        except Exception as exc:  # noqa: BLE001 — surfacing any executor failure to reflect
            return {"rows": [], "error": str(exc), "decision_log": [f"execute failed: {exc}"]}

    def should_retry(state: AgentState) -> str:
        if state.get("error") and state.get("retries", 0) < MAX_SQL_RETRIES:
            return "retry"
        return "answer"

    def bump_retries(state: AgentState) -> AgentState:
        return {"retries": state.get("retries", 0) + 1, "decision_log": ["reflect: retrying"]}

    def answer(state: AgentState) -> AgentState:
        if state.get("error") and not state.get("rows"):
            return {
                "answer": (
                    "I couldn't answer that from the property data — the query failed: "
                    f"{state['error']}"
                ),
                "decision_log": ["answer: reported failure honestly"],
            }
        text = llm(
            [
                {"role": "system", "content": grounding},
                {
                    "role": "user",
                    "content": (
                        "Phrase a short insight answering the question from these rows. "
                        "State the value(s), the basis month, and any volume caveat. "
                        "Do not invent numbers not present in the rows.\n\n"
                        f"Question: {state['question']}\n"
                        f"SQL: {state.get('sql', '')}\n"
                        f"Rows: {state.get('rows', [])}"
                    ),
                },
            ]
        )
        return {"answer": text, "decision_log": ["answer: phrased insight"]}

    def refuse(state: AgentState) -> AgentState:
        return {
            "answer": (
                "I don't have data to answer that. I can answer questions about NSW "
                "property rents, sales and gross yields by postcode, property type "
                "and month."
            ),
            "decision_log": ["answer: refused (out of scope)"],
        }

    graph = StateGraph(AgentState)
    graph.add_node("route", route)
    graph.add_node("plan_sql", plan_sql)
    graph.add_node("execute", execute)
    graph.add_node("bump_retries", bump_retries)
    graph.add_node("answer", answer)
    graph.add_node("refuse", refuse)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route", lambda s: s["route"], {"answer": "plan_sql", "refuse": "refuse"}
    )
    graph.add_edge("plan_sql", "execute")
    graph.add_conditional_edges(
        "execute", should_retry, {"retry": "bump_retries", "answer": "answer"}
    )
    graph.add_edge("bump_retries", "plan_sql")
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()


@mlflow.trace(span_type="AGENT", name="qa_agent")
def ask(agent, question: str) -> dict[str, Any]:
    """Run one question through a compiled agent; returns answer + trace.

    Traced as the root span so the LLM and SQL spans below it land in one tree.
    MLflow does not open a root for a ChatAgent's predict, so without this each
    call would be filed as its own orphan trace — and a retried statement, the
    thing you most want to inspect, would scatter across unrelated traces.
    """
    state = agent.invoke({"question": question, "retries": 0, "decision_log": []})
    return {
        "answer": state.get("answer", ""),
        "sql": state.get("sql", ""),
        "decision_log": state.get("decision_log", []),
    }


def make_databricks_agent(
    warehouse_id: str,
    catalog: str = "workspace",
    schema: str = "propertyiq",
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
):
    """Wire the graph to the real FMAPI model + serverless warehouse.

    Deliberately uses the databricks-sdk for both the LLM and SQL legs:
    databricks-langchain would also work, but it drags in databricks-connect,
    which shadows local pyspark and breaks the unit-test Spark session.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()

    @mlflow.trace(span_type="LLM", name=model_endpoint)
    def llm(messages: list[dict[str, str]]) -> str:
        response = w.serving_endpoints.query(
            name=model_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"]) for m in messages
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    # TOOL rather than RETRIEVER: this agent answers by executing generated SQL
    # on the warehouse, and the span's input is the statement itself — which is
    # the single most useful thing to see when an answer looks wrong.
    @mlflow.trace(span_type="TOOL", name="warehouse_sql")
    def run_sql(sql: str) -> list[dict[str, Any]]:
        import time

        result = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id, statement=sql, wait_timeout="50s"
        )
        deadline = time.time() + 300
        while result.status.state.value in ("PENDING", "RUNNING") and time.time() < deadline:
            time.sleep(5)
            result = w.statement_execution.get_statement(result.statement_id)
        if result.status.state.value != "SUCCEEDED":
            raise RuntimeError(f"statement {result.status.state.value}: {result.status.error}")
        columns = [c.name for c in result.manifest.schema.columns]
        data = result.result.data_array or [] if result.result else []
        return [dict(zip(columns, row, strict=False)) for row in data]

    return build_agent(llm, run_sql, catalog=catalog, schema=schema)
