"""rag_transcript_agent — transcript-lab's answer paths on Vector Search.

transcript-lab ships four comparable answer paths over one retrieval stack;
three of them port cleanly onto Databricks and live here as *modes* of a single
agent rather than three deployments:

  single   one retrieval, then answer. transcript-lab's single-hop
           `rag_transcript_agent` path.
  multi    decompose -> retrieve per sub-question -> synthesize. Its recursive
           multi-hop path, depth-capped.
  agentic  a ReAct loop where the model decides what to retrieve and when to
           stop. Its LangGraph agentic path, and the default.

(The fourth path, GraphRAG, needs the Neo4j community summaries; the graph
tables are exported but that mode is out of scope here.)

Same dependency-injection shape as `qa_agent.build_agent`: the graph takes a
`llm` callable and a `retrieve` callable, so tests script both and never touch a
network. `make_databricks_agent` wires the real FMAPI model plus the Vector
Search index.

One retrieval subtlety worth stating: the index query needs an embedding of the
*question*, produced by the same model that embedded the chunks. That is why
the gte index is the agent's — the embedding endpoint is reachable from the
serving container, whereas MiniLM would mean shipping sentence-transformers
into the image. The MiniLM index exists for the local parity eval instead.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

import mlflow
from langgraph.graph import END, StateGraph

MODES = ("single", "multi", "agentic")
DEFAULT_MODE = "agentic"
TOP_K = 8
MAX_SUBQUESTIONS = 3
MAX_HOPS = 3

# Agentic decide protocols (s09). v1 is the original binary sufficiency check;
# v2 ports transcript-lab's research protocol — sub-topic gap analysis, effort
# calibration ("broad question → 3-6 focused searches"), no repeated queries —
# which is what turned the dev agent into a researcher while v1 stopped after
# one hop: with eight plausible excerpts already attached, "do you have enough?"
# is nearly always answerable with yes.
#
# v3 is v2 without the seed retrieval — the structural half of the port. The
# matrix showed the seed is an excuse to stop: llama on v2 researched (2.5
# broad hops) but gpt-oss read the eight seeded chunks and answered from them
# (1.3). Dev's loop never had a seed; its model must search to see anything.
# v3 makes decide drive the first retrieval too, and adds a stop rule so the
# budget is spent on gaps, not spirals.
#
# "react" is what actually ships (s09 close-out): retrieval as a native
# function-calling TOOL, the model looping search -> read -> search until it
# answers without a tool call — transcript-lab's exact structure, which the
# matrix proved is the only shape whose research behaviour survives contact
# with every model temperament: prompts persuaded llama (v2: 2.5 hops) but
# not gpt-oss (1.3); mechanical fan-out forced hops but flooded the judge;
# the tool loop makes searching the *path of least resistance* instead of an
# instruction to obey. v1-v4 remain selectable as the measured record.
#
# v4 is the matrix's synthesis: decompose-first agentic. The model's own
# decomposition (multi's tested node) buys the research floor mechanically —
# a broad question fans out into up-to-3 sub-queries, one retrieval each,
# before the v2 decide loop chases what is still missing; a narrow question
# passes through decompose unchanged, retrieves with the raw question exactly
# like the measured A0 baseline, and keeps the plain answer contract (the
# structured Key Findings contract cost narrow questions real composite in
# both v2 and v3). Dev's protocol steps 2-3 ARE decomposition, so this is the
# closer port, not a workaround.
PROTOCOLS = ("v1", "v2", "v3", "v4", "react")
DEFAULT_PROTOCOL = "react"
PROTOCOL_MAX_HOPS = {"v1": MAX_HOPS, "v2": 6, "v3": 6, "v4": 6, "react": 8}

SYSTEM = """\
You answer questions about a corpus of YouTube transcripts covering data
engineering, Databricks, AI engineering, system design and careers.

Rules:
1. Answer ONLY from the retrieved excerpts. Never use outside knowledge.
2. Cite every claim as [video title @ MM:SS] using the excerpt's own metadata.
3. If the excerpts do not answer the question, say so plainly rather than
   guessing — an honest miss is worth more than a plausible invention.
4. Quote sparingly; summarise in your own words.
5. When speakers disagree, report the disagreement rather than picking one.
"""


class RagState(TypedDict, total=False):
    question: str
    mode: str
    queries: list[str]
    chunks: list[dict[str, Any]]
    hops: int
    # Every query already searched (v2 protocol): fed back into the decide
    # prompt so the model cannot burn its hop budget re-running one query.
    asked: list[str]
    # The react protocol's rolling conversation (OpenAI-shaped dicts).
    messages: Annotated[list[dict[str, Any]], lambda a, b: a + b]
    answer: str
    decision_log: Annotated[list[str], lambda a, b: a + b]


def format_chunks(chunks: list[dict[str, Any]]) -> str:
    """Render excerpts for the prompt, carrying the metadata a citation needs."""
    if not chunks:
        return "(no excerpts retrieved)"
    parts = []
    for chunk in chunks:
        stamp = _timestamp(chunk.get("start_seconds"))
        parts.append(
            f"[{chunk.get('title') or 'unknown'} @ {stamp}] "
            f"(channel: {chunk.get('channel_name') or 'unknown'})\n"
            f"{chunk.get('text') or ''}"
        )
    return "\n\n---\n\n".join(parts)


def _timestamp(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "00:00"
    return f"{total // 60:02d}:{total % 60:02d}"


def dedupe_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best-scoring copy of each chunk across hops.

    Multi-hop and agentic modes retrieve repeatedly and overlap heavily; without
    this the prompt fills with duplicates and crowds out real evidence.
    """
    best: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        key = chunk.get("chunk_key")
        if key is None:
            continue
        current = best.get(key)
        if current is None or (chunk.get("score") or 0) > (current.get("score") or 0):
            best[key] = chunk
    return sorted(best.values(), key=lambda c: c.get("score") or 0, reverse=True)


ANSWER_STRUCTURE_V2 = """\
Structure the answer as:
## Key Findings
A numbered list of the most important insights, each one concise sentence with
its citations. Then one short titled section per finding, expanding it from the
excerpts that support it. For a narrow factual question, a single finding with
a short expansion is fine."""


RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_transcript_chunks",
        "description": (
            "Search the indexed YouTube transcript corpus for excerpts relevant "
            "to a focused query. Call it once per distinct sub-topic; results "
            "accumulate across calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "focused, specific search query"}
            },
            "required": ["query"],
        },
    },
}

REACT_SYSTEM = (
    SYSTEM
    + """
Research protocol:
1. Start by calling retrieve_transcript_chunks with the user's question to see
   what the corpus covers.
2. From those results, identify the distinct sub-topics worth evidence of
   their own, and call the tool once per sub-topic with a focused query —
   not a paraphrase of the original question.
3. A broad question (a guide, a comparison, "what do the videos say about X")
   typically needs 4-8 calls; a narrow factual question 1-2. Stop as soon as
   the excerpts fully answer the question — extra searches past that point
   dilute the evidence.
4. Then answer WITHOUT calling the tool, from the accumulated excerpts only.
   If you researched more than one sub-topic, structure the answer as
   '## Key Findings' (numbered, cited) followed by one short titled section
   per finding; for a narrow question, plain cited prose is fine."""
)


def build_react_agent(
    tool_llm: Callable[[list[dict[str, Any]], list[dict] | None], dict[str, Any]],
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    max_iterations: int | None = None,
):
    """The shipped agentic mode: retrieval as a native function-calling tool.

    ``tool_llm`` takes (messages, tools) and returns the assistant message as
    a plain dict — ``{"content": str|None, "tool_calls": [...]}`` in the
    OpenAI-compatible shape FMAPI speaks. The model loops search -> read ->
    search until it answers without a tool call; the only hard cap is
    ``max_iterations`` tool calls, after which one final tool-less call is
    forced so the run always ends in an answer rather than a stall.
    """
    cap = max_iterations if max_iterations is not None else PROTOCOL_MAX_HOPS["react"]

    def agent(state: RagState) -> RagState:
        messages = list(state.get("messages") or [])
        if not messages:
            messages = [
                {"role": "system", "content": REACT_SYSTEM},
                {"role": "user", "content": state["question"]},
            ]
        hops = state.get("hops", 0)
        at_cap = hops >= cap
        if at_cap:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop researching. Answer now from the excerpts already "
                        "retrieved, with citations."
                    ),
                }
            )
        reply = tool_llm(messages, None if at_cap else [RETRIEVE_TOOL])
        entry = dict(reply)
        entry["role"] = "assistant"
        if at_cap:
            # No tools were offered, so no tool call is honoured — the route
            # must terminate here even against a model that emits one anyway.
            entry["tool_calls"] = None
        tool_calls = entry.get("tool_calls") or []
        log = (
            [f"agent: {len(tool_calls)} tool call(s)"]
            if tool_calls
            else [f"agent: answered after {hops} search(es)"]
        )
        if at_cap:
            log = [f"agent: hop cap ({cap}) reached, answer forced"]
        appended = messages[len(state.get("messages") or []) :] + [entry]
        return {"messages": appended, "decision_log": log}

    def tools(state: RagState) -> RagState:
        last = (state.get("messages") or [])[-1]
        found_all: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        hops = state.get("hops", 0)
        for call in last.get("tool_calls") or []:
            try:
                arguments = json.loads(call.get("function", {}).get("arguments") or "{}")
            except ValueError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            query = str(arguments.get("query") or state["question"])
            found = retrieve(query, TOP_K)
            found_all.extend(found)
            hops += 1
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": format_chunks(dedupe_chunks(found)),
                }
            )
        merged = dedupe_chunks((state.get("chunks") or []) + found_all)
        return {
            "messages": results,
            "chunks": merged,
            "hops": hops,
            "decision_log": [
                f"tool: +{len(found_all)} excerpts across {len(results)} call(s), "
                f"{len(merged)} unique total"
            ],
        }

    def route(state: RagState) -> str:
        last = (state.get("messages") or [])[-1]
        return "tools" if last.get("tool_calls") else "done"

    def finish(state: RagState) -> RagState:
        last = (state.get("messages") or [])[-1]
        answer = normalize_content(last.get("content"))
        if not answer.strip():
            answer = "I couldn't find anything in the transcript corpus that answers that."
        return {"answer": answer, "decision_log": ["answer: from the tool-use loop"]}

    graph = StateGraph(RagState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finish", finish)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "done": "finish"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finish", END)
    return graph.compile()


def build_agent(
    llm: Callable[[list[dict[str, str]]], str],
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    mode: str = DEFAULT_MODE,
    protocol: str = DEFAULT_PROTOCOL,
    max_hops: int | None = None,
):
    """Compile one mode's graph with injected LLM and retriever.

    ``protocol`` and ``max_hops`` only shape the *agentic* mode's decide loop
    and answer contract; single and multi are the eval's fixed controls and
    deliberately never change under a protocol flip.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")
    if mode == "agentic" and protocol == "react":
        # The shipped shape. ``llm`` must be the tool-capable callable here —
        # (messages, tools) -> assistant message dict; make_databricks_agent
        # wires make_fmapi_tool_llm for it.
        return build_react_agent(llm, retrieve, max_iterations=max_hops)
    hop_budget = max_hops if max_hops is not None else PROTOCOL_MAX_HOPS[protocol]
    research = mode == "agentic" and protocol in ("v2", "v3", "v4")

    # -- shared nodes ------------------------------------------------------

    def answer(state: RagState) -> RagState:
        chunks = state.get("chunks") or []
        if not chunks:
            return {
                "answer": ("I couldn't find anything in the transcript corpus that answers that."),
                "decision_log": ["answer: no excerpts retrieved"],
            }
        instruction = "Answer the question from these excerpts, with citations."
        # v4 narrow passthrough keeps A0's plain contract: Key Findings on a
        # one-query answer read as padding and cost narrow composite in v2/v3.
        researched = len(state.get("asked") or []) > 1
        if research and (protocol != "v4" or researched):
            instruction += "\n" + ANSWER_STRUCTURE_V2
        text = llm(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n\n"
                        f"Retrieved excerpts:\n{format_chunks(chunks)}\n\n"
                        f"{instruction}"
                    ),
                },
            ]
        )
        return {
            "answer": text,
            "decision_log": [f"answer: synthesised from {len(chunks)} excerpts"],
        }

    # -- single ------------------------------------------------------------

    def retrieve_once(state: RagState) -> RagState:
        chunks = retrieve(state["question"], TOP_K)
        return {
            "chunks": dedupe_chunks(chunks),
            "queries": [state["question"]],
            "asked": [state["question"]],
            "decision_log": [f"retrieve: {len(chunks)} excerpts for the question as asked"],
        }

    # -- multi -------------------------------------------------------------

    def decompose(state: RagState) -> RagState:
        if protocol == "v4":
            # Firmer passthrough than multi's shared wording: measured on the
            # matrix, llama split every narrow question 3 ways under "if it is
            # already a single simple question, return it unchanged", which
            # dragged narrow answers off the A0 baseline the no-regression
            # check protects. Naming the two cases and demanding the verbatim
            # copy is what actually holds at temperature 0.
            instruction = (
                "Decide whether this question needs multiple searches.\n"
                f"- If it spans several distinct topics (a guide, a comparison, "
                f"advice across areas), break it into at most {MAX_SUBQUESTIONS} "
                "standalone search queries — one per distinct topic, one per "
                "line, no numbering.\n"
                "- If it is one factual question about one thing, reply with "
                "the question COPIED EXACTLY as written, alone on one line. "
                "Do not rephrase it, do not split it.\n\n"
                f"Question: {state['question']}"
            )
        else:
            instruction = (
                "Break this question into at most "
                f"{MAX_SUBQUESTIONS} standalone search queries that together "
                "would answer it. Return one query per line, no numbering. "
                "If it is already a single simple question, return it unchanged.\n\n"
                f"Question: {state['question']}"
            )
        raw = llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}])
        queries = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
        queries = queries[:MAX_SUBQUESTIONS] or [state["question"]]
        return {"queries": queries, "decision_log": [f"decompose: {len(queries)} sub-question(s)"]}

    def retrieve_each(state: RagState) -> RagState:
        queries = state.get("queries") or []
        collected: list[dict[str, Any]] = []
        for query in queries:
            collected.extend(retrieve(query, TOP_K))
        deduped = dedupe_chunks(collected)
        return {
            "chunks": deduped,
            # v4's decide loop continues from here: the fan-out counts against
            # the hop budget and its queries join the no-repeat list.
            "hops": len(queries),
            "asked": list(queries),
            "decision_log": [
                f"retrieve: {len(collected)} excerpts across "
                f"{len(queries)} queries, {len(deduped)} after dedupe"
            ],
        }

    # -- agentic -----------------------------------------------------------

    def _decide_prompt(state: RagState) -> str:
        chunks = state.get("chunks") or []
        if protocol == "v1":
            return (
                f"Question: {state['question']}\n\n"
                "Excerpts retrieved so far:\n"
                f"{format_chunks(chunks)}\n\n"
                "Do you have enough to answer with citations? Reply with JSON only:\n"
                '{"action": "answer"} or {"action": "search", "query": "<what to look up>"}'
            )
        videos = len({c.get("video_id") for c in chunks if c.get("video_id")})
        asked = state.get("asked") or []
        asked_block = "\n".join(f"- {query}" for query in asked) or "- (none yet)"
        return (
            f"Question: {state['question']}\n\n"
            f"Excerpts retrieved so far ({len(chunks)} unique, from {videos} video(s)):\n"
            f"{format_chunks(chunks)}\n\n"
            f"Queries already searched — never repeat these:\n{asked_block}\n\n"
            "You are mid-research, deciding whether to search again or answer now.\n"
            "Work through this checklist:\n"
            "1. List the distinct sub-topics the question implies.\n"
            "2. Mark which sub-topics the excerpts above already cover, and from how\n"
            "   many different videos.\n"
            '3. Calibrate effort: a broad question (a guide, a comparison, a "what do\n'
            '   the videos say about X") typically needs 3-6 focused searches before\n'
            "   answering; a narrow factual question needs 1-2.\n"
            "4. If any sub-topic is uncovered — or covered by only one video when the\n"
            "   corpus likely has more — search for it with a focused query that\n"
            "   targets that sub-topic specifically. Never paraphrase the original\n"
            "   question as the query.\n\n"
            + (
                "5. Once the excerpts fully answer the question, stop and answer — "
                "searches past that point dilute evidence quality.\n\n"
                if protocol in ("v3", "v4")
                else "\n"
            )
            + "Reply with JSON only, on one line:\n"
            '{"action": "search", "query": "<focused sub-topic query>", '
            '"gaps": ["<uncovered sub-topic>", "..."]}\n'
            "or, only when every sub-topic is covered:\n"
            '{"action": "answer"}'
        )

    def decide(state: RagState) -> RagState:
        """The ReAct step: search again, or stop and answer."""
        hops = state.get("hops", 0)
        if hops >= hop_budget:
            # Clearing `queries` is what actually stops the loop — leaving the
            # previous hop's query in state would route straight back to search.
            return {
                "queries": [],
                "decision_log": [f"decide: hop cap ({hop_budget}) reached, answering"],
            }
        raw = llm(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _decide_prompt(state)},
            ]
        )
        action = _parse_action(raw)
        query = str(action.get("query") or "").strip()
        if action.get("action") == "search" and query:
            # v2 only: v1 is the measured baseline and must keep its shipped
            # behaviour bit-for-bit, repeated queries included.
            asked = {previous.strip().lower() for previous in (state.get("asked") or [])}
            if research and query.lower() in asked:
                # A repeated query would spend a hop re-reading the same rows;
                # treat it as the model having nothing new to look for.
                return {
                    "queries": [],
                    "decision_log": [f"decide: repeated query {query!r} blocked, answering"],
                }
            gaps = action.get("gaps")
            note = (
                f" (gaps: {', '.join(map(str, gaps))})" if isinstance(gaps, list) and gaps else ""
            )
            return {
                "queries": [query],
                "decision_log": [f"decide: search again for {query!r}{note}"],
            }
        return {"queries": [], "decision_log": ["decide: enough evidence, answering"]}

    def search(state: RagState) -> RagState:
        query = (state.get("queries") or [state["question"]])[0]
        found = retrieve(query, TOP_K)
        merged = dedupe_chunks((state.get("chunks") or []) + found)
        return {
            "chunks": merged,
            "hops": state.get("hops", 0) + 1,
            "asked": (state.get("asked") or []) + [query],
            "decision_log": [f"search: +{len(found)} excerpts, {len(merged)} unique total"],
        }

    def needs_search(state: RagState) -> str:
        return "search" if state.get("queries") else "answer"

    # -- wiring ------------------------------------------------------------

    graph = StateGraph(RagState)
    graph.add_node("answer", answer)

    if mode == "single":
        graph.add_node("retrieve", retrieve_once)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer")
    elif mode == "multi":
        graph.add_node("decompose", decompose)
        graph.add_node("retrieve", retrieve_each)
        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "answer")
    else:  # agentic
        graph.add_node("decide", decide)
        graph.add_node("search", search)
        if protocol == "v3":
            # No seed: the model sees nothing until it searches, the same
            # structural forcing the dev ReAct loop gets from tool calling.
            graph.set_entry_point("decide")
        elif protocol == "v4":
            # Decompose-first: the fan-out is the research floor, then the
            # decide loop chases remaining gaps.
            graph.add_node("decompose", decompose)
            graph.add_node("fanout", retrieve_each)
            graph.set_entry_point("decompose")
            graph.add_edge("decompose", "fanout")
            graph.add_edge("fanout", "decide")
        else:
            graph.add_node("seed", retrieve_once)
            graph.set_entry_point("seed")
            graph.add_edge("seed", "decide")
        graph.add_conditional_edges(
            "decide", needs_search, {"search": "search", "answer": "answer"}
        )
        graph.add_edge("search", "decide")

    graph.add_edge("answer", END)
    return graph.compile()


def _parse_action(raw: str) -> dict[str, Any]:
    """Read the ReAct step's JSON, tolerating fences and stray prose."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    # An unparseable step means stop rather than loop — a wrong answer beats an
    # endless retrieval bill.
    return {"action": "answer"}


@mlflow.trace(span_type="AGENT", name="rag_transcript_agent")
def ask(agent, question: str, tags: dict[str, str] | None = None) -> dict[str, Any]:
    """Run one question; returns the answer plus the trace the eval scores.

    Traced as the root span so the retriever and LLM spans below it land in one
    tree. Serving supplies no root of its own for a ChatAgent, and a local run
    or the benchmark certainly does not — without this each hop would be filed
    as its own orphan trace, which is what makes a multi-hop answer unreadable.

    `tags` are applied from *inside* the span on purpose. Tagging before calling
    this (the obvious place, in ChatAgent.predict) silently does nothing: there
    is no active trace until this function opens one.
    """
    if tags:
        mlflow.update_current_trace(tags=tags)
    state = agent.invoke(
        {"question": question, "hops": 0, "chunks": [], "queries": [], "decision_log": []}
    )
    chunks = state.get("chunks") or []
    return {
        "answer": state.get("answer", ""),
        "chunk_keys": [chunk.get("chunk_key") for chunk in chunks],
        "video_ids": list(dict.fromkeys(chunk.get("video_id") for chunk in chunks)),
        "decision_log": state.get("decision_log", []),
    }


def make_retriever(
    index_name: str,
    embedding_endpoint: str = "databricks-gte-large-en",
    workspace_client=None,
):
    """A retriever closing over the real Vector Search index.

    Note the two different clients: the *embedding* call goes to the serving
    endpoint, while the *index* query has to go through the typed SDK method —
    a raw api_client call lands on the control plane and is rejected, since the
    index lives behind its own data-plane host.
    """
    from databricks.sdk import WorkspaceClient

    w = workspace_client or WorkspaceClient()
    columns = [
        "chunk_key",
        "video_id",
        "chunk_index",
        "title",
        "channel_name",
        "source_url",
        "start_seconds",
        "text",
    ]

    # Traced as RETRIEVER so the endpoint's Traces tab shows what each hop
    # actually pulled back, with scores — the span type MLflow renders as
    # retrieval and the one production monitoring scores for relevance. Nothing
    # here is a LangChain object, so no autologger would capture it otherwise.
    @mlflow.trace(span_type="RETRIEVER", name="vector_search")
    def retrieve(query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        embedding = w.serving_endpoints.query(name=embedding_endpoint, input=[query]).data[0]
        response = w.vector_search_indexes.query_index(
            index_name=index_name,
            columns=columns,
            query_vector=embedding.embedding,
            num_results=k,
        )
        rows = (response.result.data_array if response.result else None) or []
        # query_index appends the similarity score as a trailing column.
        return [dict(zip(columns + ["score"], row, strict=False)) for row in rows]

    return retrieve


def normalize_content(content: Any) -> str:
    """FMAPI message content as plain text, whatever shape the model returns.

    llama endpoints return a string; the reasoning models (gpt-oss, qwen)
    return a list of typed parts where the answer lives in the ``text`` parts
    and the chain-of-thought in ``reasoning`` parts. The agent only ever wants
    the text — reasoning parts leaking into a decide step would be parsed as
    prose around the JSON, and into an answer as visible thinking.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    texts.append(str(part["text"]))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return str(content)


def make_fmapi_llm(
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    workspace_client=None,
):
    """A messages -> text callable over one FMAPI chat endpoint."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = workspace_client or WorkspaceClient()

    @mlflow.trace(span_type="LLM", name=model_endpoint)
    def llm(messages: list[dict[str, str]]) -> str:
        response = w.serving_endpoints.query(
            name=model_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"]) for m in messages
            ],
            temperature=0.0,
        )
        return normalize_content(response.choices[0].message.content)

    return llm


def make_fmapi_tool_llm(
    model_endpoint: str = "databricks-qwen35-122b-a10b",
    workspace_client=None,
):
    """(messages, tools) -> assistant message dict over one FMAPI endpoint.

    Goes through the OpenAI-compatible invocations route via the SDK's
    api_client — the typed query() surface has no ``tools`` parameter, and
    this keeps databricks-sdk the only dependency. Function calling on this
    route is proven live on Free Edition (P5 probe).
    """
    from databricks.sdk import WorkspaceClient

    w = workspace_client or WorkspaceClient()

    @mlflow.trace(span_type="LLM", name=model_endpoint)
    def tool_llm(messages: list[dict[str, Any]], tools: list[dict] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, "temperature": 0.0}
        if tools:
            body["tools"] = tools
        response = w.api_client.do(
            "POST", f"/serving-endpoints/{model_endpoint}/invocations", body=body
        )
        message = dict(response["choices"][0]["message"])
        return message

    return tool_llm


def make_databricks_agent(
    index_name: str = "workspace.rag.rag_chunks_gte",
    # qwen3.5: the matrix's model finding — tool-use-trained, it researches
    # through the react loop like dev's DeepSeek (llama made one call and
    # stopped; gpt-oss two). llama remains one env var away for comparisons.
    model_endpoint: str = "databricks-qwen35-122b-a10b",
    embedding_endpoint: str = "databricks-gte-large-en",
    mode: str = DEFAULT_MODE,
    protocol: str = DEFAULT_PROTOCOL,
    max_hops: int | None = None,
):
    """Wire one mode's graph to the real FMAPI model and Vector Search index."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    retrieve = make_retriever(index_name, embedding_endpoint, workspace_client=w)
    if mode == "agentic" and protocol == "react":
        llm = make_fmapi_tool_llm(model_endpoint, workspace_client=w)
    else:
        llm = make_fmapi_llm(model_endpoint, workspace_client=w)
    return build_agent(llm, retrieve, mode=mode, protocol=protocol, max_hops=max_hops)
