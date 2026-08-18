"""
Lazily built, process-lifetime engine objects shared by every bridge op.

In:  nothing; objects are created on first use from the loaded config.
Out: the retriever, agent loop, corpus analyzer and graph store. The models are loaded
     once per worker, which is the reason the worker is long-lived at all.
"""

from __future__ import annotations

from typing import Any, Dict

_CACHE: Dict[str, Any] = {}


def config() -> Any:
    from config import CFG

    if "paths_ensured" not in _CACHE:
        CFG.paths.ensure()
        _CACHE["paths_ensured"] = True
    return CFG


def embedder() -> Any:
    if "embedder" not in _CACHE:
        from retrieval.embedder import Embedder

        _CACHE["embedder"] = Embedder(config())
    return _CACHE["embedder"]


def reranker() -> Any:
    if "reranker" not in _CACHE:
        from retrieval.reranker import Reranker

        _CACHE["reranker"] = Reranker(config())
    return _CACHE["reranker"]


def retriever() -> Any:
    if "retriever" not in _CACHE:
        from tools.retrieve_evidence import EvidenceRetriever

        _CACHE["retriever"] = EvidenceRetriever(config(), embedder=embedder())
    return _CACHE["retriever"]


def loop() -> Any:
    if "loop" not in _CACHE:
        from agent.loop import AgentLoop
        from agent.tool_registry import build_full_registry

        cfg = config()
        _CACHE["loop"] = AgentLoop(
            build_full_registry(retriever=retriever(), config=cfg), config=cfg, retriever=retriever()
        )
    return _CACHE["loop"]


def analyzer() -> Any:
    if "analyzer" not in _CACHE:
        from tools.analyze_corpus import CorpusAnalyzer

        _CACHE["analyzer"] = CorpusAnalyzer(config())
    return _CACHE["analyzer"]


def graph() -> Any:
    if "graph" not in _CACHE:
        from graph.neo4j_store import GraphStore

        _CACHE["graph"] = GraphStore(config())
    return _CACHE["graph"]


def invalidate_index() -> None:
    """After indexing, drop the retriever's cached index so the next question sees it."""
    if "retriever" in _CACHE:
        _CACHE["retriever"].invalidate()
