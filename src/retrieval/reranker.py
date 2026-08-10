"""
Cross-encoder reranking of a candidate shortlist — stage two of retrieval, never stage one.

In:  a query and the chunk records the bi-encoder shortlisted (30-50 of them).
Out: the same records with a "rerank_score", ordered best first. Scores are raw logits:
     useful for ranking, not calibrated, and never comparable across queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.device import resolve_device
from common.tokenization import TokenCounter
from config import CFG, Config


class RerankError(RuntimeError):
    """The cross-encoder could not be loaded."""


class Reranker:
    """Scores (query, chunk) pairs with full cross-attention in a single pass.

    A bi-encoder embeds the two sides separately and compares vectors, so the model
    never sees them together. A cross-encoder does, which is far more accurate but
    cannot be precomputed — hence it runs only on the shortlist. Running it over the
    index would be quadratic in corpus size and is the thing this class must never do.
    """

    def __init__(self, config: Config = CFG, counter: Optional[TokenCounter] = None) -> None:
        self.model_name = str(config.retrieval.reranker_model)
        self.max_query_tokens = int(config.retrieval.max_query_tokens)
        self.max_chunk_tokens = int(config.chunking.max_tokens)
        self.counter = counter if counter is not None else TokenCounter(config)
        self._model = None
        self.last_latency_ms = 0
        self.last_pair_count = 0
        self.device = resolve_device(config)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankError(f"sentence-transformers is not installed: {exc}") from None
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                raise RerankError(
                    f"could not load {self.model_name} on {self.device}: {exc}"
                ) from None
        return self._model

    def _pair(self, query: str, text: str) -> Tuple[str, str]:
        """Trim a (query, chunk) pair to fit the model's single 512-token window.

        The window is shared between the two sides, so the query is cut to its budget
        and the chunk to what remains. Chunks are already sized for this at index time;
        this is the guard for anything that slipped through.
        """
        return (
            self.counter.truncate(query, self.max_query_tokens),
            self.counter.truncate(text, self.max_chunk_tokens),
        )

    def rerank(
        self, query: str, candidates: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score and reorder `candidates`. Returns a new list, best first.

        The input is the bi-encoder's shortlist. This method is deliberately given no
        access to the index, so it cannot be pointed at the whole corpus by accident.
        """
        if not candidates:
            self.last_latency_ms, self.last_pair_count = 0, 0
            return []

        pairs = [self._pair(query, str(candidate.get("text", ""))) for candidate in candidates]
        started = time.perf_counter()
        scores = self.model.predict(pairs, show_progress_bar=False)
        self.last_latency_ms = int((time.perf_counter() - started) * 1000)
        self.last_pair_count = len(pairs)

        scored = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored
