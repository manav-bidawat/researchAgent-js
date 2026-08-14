"""
`search_literature`: fetch papers from arXiv and add them to the index.

In:  a natural-language topic, optional max_results and arXiv categories.
Out: {"papers_added", "papers_skipped", "chunks_added", "figures_added", "topic_tag",
     "total_indexed"} or an error dict. Wires M1 collection, M2 extraction and M3 indexing
     behind one tool call; partial success is reported, never discarded.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from config import CFG, Config
from corpus import collect
from corpus.manifest import Manifest
from extraction.ingest import ingest_paper
from llm_client import LLMClient
from retrieval.indexer import index_chunks


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


class LiteratureSearcher:
    """Fetch, extract and index in one call, holding the clients the three stages share.

    `on_corpus_change` exists because the retriever caches the index and the chunk table
    in memory for the life of a conversation. Adding papers mid-conversation would
    otherwise leave it searching the corpus as it was before this call — silently, since
    a stale index still returns plausible results. This is a callback rather than a call
    into the other tool: tools never call tools, all control flow stays in the agent.
    """

    def __init__(
        self,
        config: Config = CFG,
        client: Optional[LLMClient] = None,
        on_corpus_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.on_corpus_change = on_corpus_change

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        categories: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Search arXiv, download new papers, chunk them, and index them."""
        query = (query or "").strip()
        if not query:
            return _error("empty_query", "a non-empty topic is required")

        fetched = collect.search_and_fetch(
            query, max_results=max_results, categories=categories,
            config=self.config, llm=self.client,
        )
        rate_limited: Optional[Dict[str, Any]] = None
        if "error" in fetched:
            partial = fetched.get("partial") or {}
            if fetched["error"] != "arxiv_rate_limited" or not partial.get("papers_added"):
                return fetched
            # A 429 partway through the downloads still leaves whole PDFs on disk and
            # their papers in the manifest. Extract and index them anyway: dedup means
            # the next run skips them, so anything not ingested now is never ingested,
            # and sits in the corpus invisible to retrieval.
            rate_limited = fetched
            fetched = partial

        added = fetched.get("papers_added", [])
        topic_tag = fetched["topic_tag"]

        # Extraction, per paper. One bad PDF must not lose the others, so failures are
        # collected and reported rather than raised.
        chunks_added = 0
        figures_added = 0
        failures: List[Dict[str, str]] = list(fetched.get("failures", []))
        indexed_ok: List[Dict[str, Any]] = []

        for paper in added:
            result = ingest_paper(paper["paper_id"], client=self.client, config=self.config)
            if "error" in result:
                if result["error"] != "already_ingested":
                    failures.append({"paper_id": paper["paper_id"], "detail": result["detail"]})
                    continue
                indexed_ok.append(paper)
                continue
            chunks_added += result["chunks_added"]
            figures_added += result["figures_added"]
            indexed_ok.append(paper)

        indexing = index_chunks(self.config)
        if "error" in indexing:
            return _error(
                "indexing_failed", indexing["detail"],
                partial={
                    "papers_added": indexed_ok, "chunks_added": chunks_added,
                    "figures_added": figures_added, "topic_tag": topic_tag,
                },
            )

        if self.on_corpus_change is not None:
            self.on_corpus_change()

        manifest = Manifest.load(self.config, strict_model_check=False)
        result = {
            "papers_added": [
                {"arxiv_id": p["arxiv_id"], "paper_id": p["paper_id"], "title": p["title"],
                 "year": p["year"], "abstract_snippet": p["abstract_snippet"]}
                for p in indexed_ok
            ],
            "papers_skipped": fetched.get("papers_skipped", 0),
            "chunks_added": chunks_added,
            "figures_added": figures_added,
            "topic_tag": topic_tag,
            "total_indexed": len(manifest.papers),
            "arxiv_query": fetched.get("arxiv_query", ""),
        }
        # Neo4j is an optional projection, but when it is configured the corpus may
        # have changed during this tool call, so refresh it before exposing new papers
        # to an MCP graph traversal. A graph outage does not discard successful FAISS
        # indexing; the result names it for the caller to retry.
        from graph.neo4j_store import GraphStore
        graph = GraphStore(self.config)
        if graph.configured:
            result["graph_sync"] = graph.sync()
        if failures:
            # Three good papers out of five is a valid result, not a failure.
            result["failures"] = failures
            result["note"] = (
                f"{len(failures)} paper(s) could not be processed; the rest were indexed"
            )
        if rate_limited is not None:
            # Still an error, so the agent does not read this as a complete search and
            # immediately ask for more — but it carries everything that did get indexed.
            return _error("arxiv_rate_limited", rate_limited["detail"], partial=result)
        return result


SEARCH_LITERATURE_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The topic to search arXiv for, in natural language.",
        },
        "max_results": {
            "type": "integer",
            "description": "How many papers to fetch. Defaults to 8.",
        },
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "arXiv categories to restrict the search to, e.g. ['cs.LG','cs.CL'].",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}
