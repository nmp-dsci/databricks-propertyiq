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

from langgraph.graph import END, StateGraph

MODES = ("single", "multi", "agentic")
DEFAULT_MODE = "agentic"
TOP_K = 8
MAX_SUBQUESTIONS = 3
MAX_HOPS = 3

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


def build_agent(
    llm: Callable[[list[dict[str, str]]], str],
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    mode: str = DEFAULT_MODE,
):
    """Compile one mode's graph with injected LLM and retriever."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

    # -- shared nodes ------------------------------------------------------

    def answer(state: RagState) -> RagState:
        chunks = state.get("chunks") or []
        if not chunks:
            return {
                "answer": ("I couldn't find anything in the transcript corpus that answers that."),
                "decision_log": ["answer: no excerpts retrieved"],
            }
        text = llm(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n\n"
                        f"Retrieved excerpts:\n{format_chunks(chunks)}\n\n"
                        "Answer the question from these excerpts, with citations."
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
            "decision_log": [f"retrieve: {len(chunks)} excerpts for the question as asked"],
        }

    # -- multi -------------------------------------------------------------

    def decompose(state: RagState) -> RagState:
        raw = llm(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Break this question into at most "
                        f"{MAX_SUBQUESTIONS} standalone search queries that together "
                        "would answer it. Return one query per line, no numbering. "
                        "If it is already a single simple question, return it unchanged.\n\n"
                        f"Question: {state['question']}"
                    ),
                },
            ]
        )
        queries = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]
        queries = queries[:MAX_SUBQUESTIONS] or [state["question"]]
        return {"queries": queries, "decision_log": [f"decompose: {len(queries)} sub-question(s)"]}

    def retrieve_each(state: RagState) -> RagState:
        collected: list[dict[str, Any]] = []
        for query in state.get("queries") or []:
            collected.extend(retrieve(query, TOP_K))
        deduped = dedupe_chunks(collected)
        return {
            "chunks": deduped,
            "decision_log": [
                f"retrieve: {len(collected)} excerpts across "
                f"{len(state.get('queries') or [])} queries, {len(deduped)} after dedupe"
            ],
        }

    # -- agentic -----------------------------------------------------------

    def decide(state: RagState) -> RagState:
        """The ReAct step: search again, or stop and answer."""
        hops = state.get("hops", 0)
        if hops >= MAX_HOPS:
            # Clearing `queries` is what actually stops the loop — leaving the
            # previous hop's query in state would route straight back to search.
            return {
                "queries": [],
                "decision_log": [f"decide: hop cap ({MAX_HOPS}) reached, answering"],
            }
        raw = llm(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {state['question']}\n\n"
                        "Excerpts retrieved so far:\n"
                        f"{format_chunks(state.get('chunks') or [])}\n\n"
                        "Do you have enough to answer with citations? Reply with JSON only:\n"
                        '{"action": "answer"} or {"action": "search", "query": "<what to look up>"}'
                    ),
                },
            ]
        )
        action = _parse_action(raw)
        if action.get("action") == "search" and action.get("query"):
            return {
                "queries": [action["query"]],
                "decision_log": [f"decide: search again for {action['query']!r}"],
            }
        return {"queries": [], "decision_log": ["decide: enough evidence, answering"]}

    def search(state: RagState) -> RagState:
        query = (state.get("queries") or [state["question"]])[0]
        found = retrieve(query, TOP_K)
        merged = dedupe_chunks((state.get("chunks") or []) + found)
        return {
            "chunks": merged,
            "hops": state.get("hops", 0) + 1,
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
        graph.add_node("seed", retrieve_once)
        graph.add_node("decide", decide)
        graph.add_node("search", search)
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


def ask(agent, question: str) -> dict[str, Any]:
    """Run one question; returns the answer plus the trace the eval scores."""
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


def make_databricks_agent(
    index_name: str = "workspace.rag.rag_chunks_gte",
    model_endpoint: str = "databricks-meta-llama-3-3-70b-instruct",
    embedding_endpoint: str = "databricks-gte-large-en",
    mode: str = DEFAULT_MODE,
):
    """Wire one mode's graph to the real FMAPI model and Vector Search index."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()

    def llm(messages: list[dict[str, str]]) -> str:
        response = w.serving_endpoints.query(
            name=model_endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"]) for m in messages
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    retrieve = make_retriever(index_name, embedding_endpoint, workspace_client=w)
    return build_agent(llm, retrieve, mode=mode)
