"""
Bi-encoder embedding of chunk text and queries, with a content-hash vector cache.

In:  passage texts (as stored) or a query string; the model name comes from config.
Out: float32 numpy arrays, L2-normalised so inner product equals cosine similarity.
     Queries get BGE's instruction prefix; passages never do.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import CFG, Config
from common.device import resolve_device
from common.records import content_hash
from common.storage import read_json, write_json_atomic


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or produced the wrong shape."""


class EmbeddingCache:
    """content_hash -> vector, persisted as JSON.

    Keyed by the hash of the text, never by chunk_id, so identical text appearing in two
    papers is embedded once and reused — while both papers keep their own chunk record
    (docs/DATA_SCHEMA.md section 2).
    """

    def __init__(self, config: Config = CFG) -> None:
        self.path = config.paths.embedding_cache
        self.dim = int(config.embedding.dim)
        raw = read_json(self.path, default={}) or {}
        self._data: Dict[str, List[float]] = {
            key: value for key, value in raw.items()
            if isinstance(value, list) and len(value) == self.dim
        }
        self._dirty = False

    def get(self, digest: str) -> Optional[np.ndarray]:
        vector = self._data.get(digest)
        return np.asarray(vector, dtype=np.float32) if vector is not None else None

    def put(self, digest: str, vector: np.ndarray) -> None:
        self._data[digest] = [float(x) for x in vector]
        self._dirty = True

    def save(self) -> None:
        if self._dirty:
            write_json_atomic(self.path, self._data, indent=None)
            self._dirty = False

    def __len__(self) -> int:
        return len(self._data)


class Embedder:
    """Wraps the sentence-transformers bi-encoder named in config.

    The model is loaded lazily on first use, because importing this module happens in
    places that never embed anything.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.model_name = str(config.embedding.model)
        self.dim = int(config.embedding.dim)
        self.batch_size = int(config.embedding.batch_size)
        self.normalize = bool(config.embedding.normalize)
        self.query_prefix = str(config.embedding.query_prefix)
        self.passage_prefix = str(config.embedding.passage_prefix)
        self.device = resolve_device(config)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(f"sentence-transformers is not installed: {exc}") from None
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise EmbeddingError(
                    f"could not load {self.model_name} on {self.device}: {exc}"
                ) from None
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.dim:
            raise EmbeddingError(
                f"{self.model_name} produced dim {array.shape[1]}, config says {self.dim}"
            )
        return array

    def encode_passages(
        self, texts: Sequence[str], cache: Optional[EmbeddingCache] = None
    ) -> np.ndarray:
        """Embed stored chunk text. No instruction prefix — BGE is asymmetric.

        Uses the cache for text already embedded, and encodes only the misses.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: List[Optional[np.ndarray]] = [None] * len(texts)
        pending: List[int] = []
        digests = [content_hash(text) for text in texts]

        if cache is not None:
            for index, digest in enumerate(digests):
                hit = cache.get(digest)
                if hit is not None:
                    out[index] = hit
                else:
                    pending.append(index)
        else:
            pending = list(range(len(texts)))

        if pending:
            fresh = self._encode([f"{self.passage_prefix}{texts[i]}" for i in pending])
            for slot, index in enumerate(pending):
                out[index] = fresh[slot]
                if cache is not None:
                    cache.put(digests[index], fresh[slot])

        return np.vstack([vector for vector in out]).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a search query, with BGE's instruction prefix attached.

        The prefix goes on the query and never on the passage. Applying it to both
        sides, or to neither, measurably degrades retrieval — the model was trained
        asymmetrically (docs/ARCHITECTURE.md section 5).
        """
        return self._encode([f"{self.query_prefix}{query}"])[0]

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode([f"{self.query_prefix}{query}" for query in queries])
