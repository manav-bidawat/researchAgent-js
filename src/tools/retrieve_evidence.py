"""
`retrieve_evidence`: the core RAG tool — bi-encoder, rerank, gate, expand, truncate.

In:  a query, optional k, topic_filter and chunk_types.
Out: {"chunks", "sufficient_evidence", "note", "n_candidates_considered"} or an error
     dict. Never raises, never fetches literature — the agent decides what to do next.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Set

from common.tokenization import TokenCounter
from config import CFG, Config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest, ManifestError
from retrieval.embedder import Embedder, EmbeddingError
from retrieval.reranker import RerankError, Reranker
from retrieval.vector_index import IndexError_, VectorIndex

VALID_CHUNK_TYPES = ("text", "figure", "table")


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


def _trim_head(text: str, budget: int) -> str:
    """Last `budget` characters of `text`, started at a sentence boundary if one is near.

    Neighbour context that opens mid-word reads as corruption rather than as context.
    """
    if budget <= 0 or not text:
        return ""
    clipped = text[-budget:]
    for marker in (". ", "? ", "! "):
        position = clipped.find(marker)
        if 0 <= position < len(clipped) // 2:
            return clipped[position + len(marker):].lstrip()
    return clipped.lstrip()


def _trim_tail(text: str, budget: int) -> str:
    """First `budget` characters of `text`, ended at a sentence boundary if one is near."""
    if budget <= 0 or not text:
        return ""
    clipped = text[:budget]
    for marker in (". ", "? ", "! "):
        position = clipped.rfind(marker)
        if position > len(clipped) // 2:
            return clipped[: position + 1].rstrip()
    return clipped.rstrip()


class EvidenceRetriever:
    """Holds the models and the index so repeated calls in one conversation are cheap.

    Also holds the set of chunk_ids already returned, which is what stops a re-query
    from stacking the same passages into the context window a second time. The agent
    loop owns one instance per conversation and calls reset() between conversations.
    """

    def __init__(
        self,
        config: Config = CFG,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.config = config
        self.counter = TokenCounter(config)
        self.embedder = embedder if embedder is not None else Embedder(config)
        self.reranker = reranker if reranker is not None else Reranker(config, self.counter)
        self._returned: Set[str] = set()
        self._index: Optional[VectorIndex] = None
        self._chunks: Optional[Dict[str, Dict[str, Any]]] = None
        self._by_position: Optional[Dict[str, Dict[int, str]]] = None
        self.last_rerank_ms = 0

    # ---- conversation state ----------------------------------------------------

    def reset(self) -> None:
        """Forget which chunks were already returned. Called at the start of a question."""
        self._returned.clear()

    def invalidate(self) -> None:
        """Drop the cached index and chunk table so the next call re-reads from disk.

        Called after search_literature adds papers. Without it the retriever keeps
        searching the corpus as it stood when this conversation began, and newly fetched
        papers are invisible for the rest of the run — with no error, because a stale
        index still returns plausible results.
        """
        self._index = None
        self._chunks = None
        self._by_position = None

    @property
    def returned(self) -> Set[str]:
        return set(self._returned)

    # ---- corpus access ---------------------------------------------------------

    def _load(self) -> Optional[Dict[str, Any]]:
        """Load index and chunks once per instance. Returns an error dict on failure."""
        if self._index is not None and self._chunks is not None:
            return None
        try:
            manifest = Manifest.load(self.config, strict_model_check=True)
        except ManifestError as exc:
            return _error("index_stale", str(exc))
        try:
            self._index = VectorIndex.load(manifest, self.config)
        except IndexError_ as exc:
            return _error("index_unreadable", str(exc))

        self._chunks = {chunk["chunk_id"]: chunk for chunk in ChunkStore(config=self.config)}
        by_position: Dict[str, Dict[int, str]] = {}
        for chunk in self._chunks.values():
            by_position.setdefault(chunk["paper_id"], {})[int(chunk["position"])] = chunk["chunk_id"]
        self._by_position = by_position
        return None

    def _allowed(
        self, topic_filter: Optional[str], chunk_types: Optional[Sequence[str]]
    ) -> Optional[Set[str]]:
        """chunk_ids passing the metadata filter, or None when no filter was asked for.

        topic_filter is a membership test against the chunk's topic_tags list, not an
        equality test — a chunk reached through two topics carries both tags.
        """
        if not topic_filter and not chunk_types:
            return None
        wanted_types = {str(t).lower() for t in (chunk_types or [])}
        allowed = set()
        for chunk_id, chunk in (self._chunks or {}).items():
            if topic_filter and topic_filter not in (chunk.get("topic_tags") or []):
                continue
            if wanted_types and str(chunk.get("chunk_type", "")).lower() not in wanted_types:
                continue
            allowed.add(chunk_id)
        return allowed

    # ---- assembly --------------------------------------------------------------

    def _expand(self, chunk: Dict[str, Any]) -> str:
        """Attach the adjacent chunks from the same paper, within the character budget.

        Recovers a claim severed at a chunk boundary without raising k. The citation
        still belongs to the middle chunk, so the neighbours are context, not sources.
        """
        span = int(self.config.retrieval.neighbour_expansion)
        budget = int(self.config.retrieval.max_chunk_chars)
        body = chunk["text"]
        if span <= 0:
            return body[:budget]

        positions = (self._by_position or {}).get(chunk["paper_id"], {})
        position = int(chunk["position"])
        before, after = [], []
        for offset in range(1, span + 1):
            previous = (self._chunks or {}).get(positions.get(position - offset, ""))
            following = (self._chunks or {}).get(positions.get(position + offset, ""))
            if previous:
                before.insert(0, previous["text"])
            if following:
                after.append(following["text"])

        remaining = budget - len(body)
        if remaining <= 0:
            return body[:budget]

        # Split what is left evenly, and give the *tail* of the preceding chunk rather
        # than its head — that is the part actually adjacent to this one.
        share = remaining // 2 if (before and after) else remaining
        head = _trim_head(" ".join(before), share) if before else ""
        tail = _trim_tail(" ".join(after), share) if after else ""
        return "\n".join(part for part in (head, body, tail) if part)[:budget]

    def _present(self, chunk: Dict[str, Any], score: float) -> Dict[str, Any]:
        record = {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk["paper_id"],
            "paper_title": chunk["paper_title"],
            "page": chunk["page"],
            "section": chunk.get("section"),
            "chunk_type": chunk["chunk_type"],
            "score": round(float(score), 4),
            "text": self._expand(chunk),
        }
        if chunk.get("figure_id"):
            record["figure_id"] = chunk["figure_id"]
        return record

    # ---- the tool --------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        topic_filter: Optional[str] = None,
        chunk_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve ranked evidence for `query`. See docs/TOOLS.md section 2."""
        query = (query or "").strip()
        if not query:
            return _error("empty_query", "a non-empty query is required")

        if chunk_types:
            bad = [t for t in chunk_types if str(t).lower() not in VALID_CHUNK_TYPES]
            if bad:
                return _error(
                    "bad_chunk_types",
                    f"unknown chunk_types {bad}; expected a subset of {list(VALID_CHUNK_TYPES)}",
                )

        failure = self._load()
        if failure is not None:
            return failure
        if self._index is None or len(self._index) == 0:
            return _error("empty_index", "no chunks are indexed; fetch literature first")

        k_final = int(k if k is not None else self.config.retrieval.k_final)
        if k_final <= 0:
            return _error("bad_k", f"k must be positive, got {k_final}")

        allowed = self._allowed(topic_filter, chunk_types)
        if allowed is not None and not allowed:
            return {
                "chunks": [],
                "sufficient_evidence": False,
                "note": f"no chunks match the filter (topic_filter={topic_filter!r}, "
                        f"chunk_types={list(chunk_types or [])})",
                "n_candidates_considered": 0,
            }

        try:
            query_vector = self.embedder.encode_query(query)
        except EmbeddingError as exc:
            return _error("embedding_failed", str(exc))

        k_retrieve = int(self.config.retrieval.k_retrieve)
        hits = self._index.search(query_vector, k_retrieve, allowed=allowed)
        if not hits:
            return {
                "chunks": [],
                "sufficient_evidence": False,
                "note": "the index returned no candidates for this query",
                "n_candidates_considered": 0,
            }

        candidates = [
            {**self._chunks[chunk_id], "bi_score": score}
            for chunk_id, score in hits
            if chunk_id in (self._chunks or {})
        ]

        rerank_enabled = bool(self.config.retrieval.rerank_enabled)
        if rerank_enabled:
            try:
                ranked = self.reranker.rerank(query, candidates)
            except RerankError as exc:
                # Reranking is precision, not correctness. Losing it degrades ordering;
                # refusing to answer would be worse, so fall back to the bi-encoder.
                ranked = [{**c, "rerank_score": c["bi_score"]} for c in candidates]
                rerank_enabled = False
                self.last_rerank_ms = 0
                rerank_note = f" (rerank unavailable: {exc})"
            else:
                self.last_rerank_ms = self.reranker.last_latency_ms
                rerank_note = ""
        else:
            ranked = [{**c, "rerank_score": c["bi_score"]} for c in candidates]
            self.last_rerank_ms = 0
            rerank_note = ""

        # The gate is applied per chunk, not only to the best one. Gating on the top
        # score alone still hands back everything below it, which is the thing
        # docs/ARCHITECTURE.md section 7 says not to do: a query whose best hit scrapes
        # over the line would drag four weak chunks into the context window with it.
        # The threshold has to match whatever produced the score. Cross-encoder output is
        # an uncalibrated logit spanning roughly -11..+11; bi-encoder output is a cosine
        # in 0..1. Applying the cross-encoder's threshold to cosine scores lets every
        # candidate through, which does not fail loudly — it silently removes abstention
        # and answers absent-topic questions from whatever ranked first. That is the
        # failure mode here, so the threshold is chosen by which scorer actually ran,
        # including when reranking was requested but fell back above.
        threshold = float(
            self.config.retrieval.relevance_threshold if rerank_enabled
            else self.config.retrieval.bi_encoder_relevance_threshold
        )
        top_score = ranked[0]["rerank_score"] if ranked else float("-inf")
        passing = [item for item in ranked if item["rerank_score"] >= threshold]

        if not passing:
            # Abstention as a mechanism: naive top-k always returns something, so without
            # this the model is handed weak chunks and invited to answer from them.
            return {
                "chunks": [],
                "sufficient_evidence": False,
                "note": f"best relevance score {top_score:.3f} is below the threshold "
                        f"{threshold:.3f}; the index has nothing good for this query"
                        + rerank_note,
                "n_candidates_considered": len(candidates),
            }

        selected: List[Dict[str, Any]] = []
        suppressed = 0
        for item in passing:
            if len(selected) >= k_final:
                break
            if item["chunk_id"] in self._returned:
                suppressed += 1
                continue
            selected.append(item)

        weak_dropped = len(ranked) - len(passing)

        for item in selected:
            self._returned.add(item["chunk_id"])

        notes: List[str] = []
        if suppressed:
            notes.append(f"{suppressed} chunk(s) were already returned earlier and were not repeated")
        if weak_dropped and len(selected) < k_final:
            notes.append(f"{weak_dropped} candidate(s) scored below the relevance threshold "
                         f"and were dropped, so fewer than k chunks are returned")
        if not selected:
            notes = ["every chunk above the relevance threshold was already returned earlier "
                     "in this conversation; rephrase the query or work with what you have"]
        note = "; ".join(notes)

        return {
            "chunks": [self._present(item, item["rerank_score"]) for item in selected],
            "sufficient_evidence": bool(selected),
            "note": note + rerank_note,
            "n_candidates_considered": len(candidates),
            "reranked": rerank_enabled,
            "rerank_ms": self.last_rerank_ms,
        }
