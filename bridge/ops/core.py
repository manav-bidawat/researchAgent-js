"""
Core bridge ops: index a topic, answer a question, and the corpus tools the MCP exposes.

In:  params dicts from the TypeScript CLI, web server and MCP server.
Out: the engine's own structured dicts. Tool-level failures stay as {"error", "detail"}
     results, exactly as src/ returns them; only a crash becomes a bridge error.
"""

from __future__ import annotations

from typing import Any, Dict

from bridge import runtime
from bridge.registry import Emit, op

# The loop returns its Conversation object too; only these keys go over the wire.
ANSWER_KEYS = (
    "answer", "question", "iterations", "tool_calls", "repeated_calls", "stopped_because",
    "messages_in_context", "context_tokens", "elided_results", "run_id", "trace",
    "trace_records",
)


def _failed(result: Any) -> bool:
    return isinstance(result, dict) and "error" in result


@op("ping")
def ping(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    import sys

    return {"ok": True, "python": sys.version.split()[0]}


@op("paths")
def paths(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Resolved on-disk locations, so TS can read the manifest without a round trip."""
    cfg = runtime.config()
    return {
        "manifest": str(cfg.paths.manifest),
        "data_dir": str(cfg.paths.data),
        "chunks": str(cfg.paths.chunks),
        "traces": str(cfg.paths.traces),
        "arxiv_cooldown": str(cfg.paths.arxiv_cooldown),
        "trace_text_chars": int(cfg.agent.trace_text_chars),
        "progress_text_chars": int(cfg.agent.progress_text_chars),
    }


@op("index")
def index(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Collect, extract and index one topic. Streams a `stage` event per step."""
    from corpus.collect import search_and_fetch
    from extraction.ingest import ingest_all
    from retrieval.indexer import index_chunks

    cfg = runtime.config()
    if params.get("clear_arxiv_cooldown"):
        cfg.paths.arxiv_cooldown.unlink(missing_ok=True)
        emit({"kind": "note", "text": "cleared the arXiv rate-limit cooldown"})

    emit({"kind": "stage", "stage": "collect", "step": 1, "of": 3, "topic": params["topic"]})
    collected = search_and_fetch(
        params["topic"], max_results=params.get("max_results"),
        categories=params.get("categories"), config=cfg,
    )
    rate_limited = False
    if _failed(collected):
        partial = collected.get("partial") or {}
        if collected["error"] != "arxiv_rate_limited" or not partial.get("papers_added"):
            return {"error": collected["error"], "detail": collected.get("detail"), "stage": "collection"}
        # A 429 partway through still downloaded whole PDFs. Ingest and index them, or
        # dedup hides them from every later run and they are never chunked at all.
        emit({"kind": "note", "text": f"rate-limited partway: {collected['detail']}", "warn": True})
        collected = partial
        rate_limited = True
    emit({"kind": "collected", "result": collected})

    emit({"kind": "stage", "stage": "ingest", "step": 2, "of": 3})
    ingested = ingest_all(config=cfg, describe=not params.get("no_describe", False))
    if _failed(ingested):
        return {**ingested, "stage": "ingestion"}
    emit({"kind": "ingested", "result": ingested})

    emit({"kind": "stage", "stage": "index", "step": 3, "of": 3})
    indexed = index_chunks(config=cfg, rebuild=bool(params.get("rebuild")))
    if _failed(indexed):
        return {**indexed, "stage": "indexing"}
    runtime.invalidate_index()

    # Graph projection is opt-in: when Neo4j credentials are present, keep its
    # relationships in lockstep with indexing.
    graph_result = None
    store = runtime.graph()
    if store.configured:
        graph_result = store.sync()

    return {
        "collected": collected,
        "ingested": ingested,
        "indexed": indexed,
        "graph": graph_result,
        "rate_limited": rate_limited,
    }


@op("ask")
def ask(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Run the agent loop on one question, streaming its events with results summarised."""
    from agent.trace import summarise_result

    cfg = runtime.config()
    loop = runtime.loop()
    text_chars = int(cfg.agent.trace_text_chars)

    def on_event(event: Dict[str, Any]) -> None:
        # Full chunk text would be kilobytes per frame; the trace summariser is already
        # the project's answer to that, and it is config-truncated.
        payload = dict(event)
        if "result" in payload:
            payload["result"] = summarise_result(payload["result"], text_chars)
        emit(payload)

    loop.on_event = on_event if params.get("events", True) else None
    try:
        result = loop.run(params.get("question", ""), question_id=params.get("question_id"))
    finally:
        loop.on_event = None

    if _failed(result):
        return result
    return {key: result[key] for key in ANSWER_KEYS if key in result}


@op("retrieve_evidence")
def retrieve_evidence(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    retriever = runtime.retriever()
    # Independent callers (MCP clients) must not have their evidence suppressed by the
    # dedup state a previous caller left behind.
    if params.get("reset", True):
        retriever.reset()
    return retriever.retrieve(
        params.get("query", ""), k=params.get("k"),
        topic_filter=params.get("topic_filter"), chunk_types=params.get("chunk_types"),
    )


@op("analyze_corpus")
def analyze_corpus(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    return runtime.analyzer().analyze(
        params.get("operation", ""), topic_filter=params.get("topic_filter"),
        params=params.get("params"),
    )


@op("explore_graph")
def explore_graph(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    return runtime.graph().neighbourhood(params.get("entity_id", ""), int(params.get("depth") or 1))


@op("graph_sync")
def graph_sync(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    return runtime.graph().sync()


@op("arxiv_cooldown")
def arxiv_cooldown(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Seconds left on a recorded arXiv rate limit; clears it when asked."""
    from corpus.arxiv_fetch import read_cooldown

    cfg = runtime.config()
    if params.get("clear"):
        cfg.paths.arxiv_cooldown.unlink(missing_ok=True)
    return {"seconds_remaining": read_cooldown(cfg.paths.arxiv_cooldown)}


@op("warm")
def warm(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Build the agent (and so load every model) before the first question arrives."""
    runtime.loop()
    return {"ok": True}
