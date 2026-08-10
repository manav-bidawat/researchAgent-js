"""
FAISS flat index over chunk vectors, with its chunk_id mapping kept in lockstep.

In:  chunk_ids plus their embedding vectors; a query vector to search with.
Out: ranked (chunk_id, score) pairs. Persistence writes faiss.index, embeddings.npy and
     the manifest's faiss_id_map together — the map exists nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import CFG, Config
from corpus.manifest import Manifest


class IndexError_(RuntimeError):
    """The index on disk is unusable or inconsistent with its chunk_id mapping."""


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise IndexError_(f"faiss is not installed: {exc}") from None
    return faiss


class VectorIndex:
    """A flat inner-product index plus the positional chunk_id map it depends on.

    Vectors are L2-normalised before they arrive, so inner product is cosine similarity
    and a flat index is exact. Flat is correct at this corpus size; IndexIVFPQ is the
    documented upgrade path and changes nothing above this class.
    """

    def __init__(self, dim: int, config: Config = CFG) -> None:
        self.dim = int(dim)
        self.config = config
        self._faiss = _faiss()
        self.index = self._faiss.IndexFlatIP(self.dim)
        self.chunk_ids: List[str] = []
        # Positional, like chunk_ids: the content_hash of the text actually embedded at
        # each FAISS slot. Without it, a chunk whose text changed under an unchanged
        # chunk_id keeps its stale vector forever and nothing downstream can tell.
        self.content_hashes: List[str] = []
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)

    # ---- lifecycle -------------------------------------------------------------

    @classmethod
    def load(cls, manifest: Manifest, config: Config = CFG) -> "VectorIndex":
        """Load the index and its mapping, or return an empty one when absent.

        Refuses to load a mapping whose length disagrees with the index or the vector
        array. FAISS ids are positional, so a mismatch means every lookup past the first
        divergence silently returns the wrong chunk — worse than an empty index.
        """
        index = cls(int(manifest.data.get("embedding_dim") or config.embedding.dim), config)
        path = config.paths.faiss_index
        chunk_ids = list(manifest.data.get("faiss_id_map") or [])
        content_hashes = list(manifest.data.get("indexed_hashes") or [])

        if not path.is_file():
            if chunk_ids:
                raise IndexError_(
                    f"manifest lists {len(chunk_ids)} indexed chunks but {path} is missing; "
                    "the index must be rebuilt"
                )
            return index

        try:
            loaded = index._faiss.read_index(str(path))
        except Exception as exc:
            raise IndexError_(f"could not read {path}: {exc}") from None

        vectors = np.zeros((0, index.dim), dtype=np.float32)
        if config.paths.embeddings.is_file():
            try:
                vectors = np.load(config.paths.embeddings).astype(np.float32)
            except Exception as exc:
                raise IndexError_(f"could not read {config.paths.embeddings}: {exc}") from None

        if loaded.ntotal != len(chunk_ids):
            raise IndexError_(
                f"faiss index holds {loaded.ntotal} vectors but faiss_id_map has "
                f"{len(chunk_ids)} entries; the mapping is unrecoverable, rebuild the index"
            )
        if vectors.shape[0] != loaded.ntotal:
            raise IndexError_(
                f"embeddings.npy has {vectors.shape[0]} rows but the index holds "
                f"{loaded.ntotal}; they are appended in lockstep and must match"
            )
        if content_hashes and len(content_hashes) != loaded.ntotal:
            raise IndexError_(
                f"indexed_hashes has {len(content_hashes)} entries but the index holds "
                f"{loaded.ntotal}; they are appended in lockstep and must match"
            )

        index.index = loaded
        index.chunk_ids = chunk_ids
        # An index written before indexed_hashes existed has none. Treat those vectors as
        # of unknown provenance rather than as current, so the next run re-embeds them.
        index.content_hashes = content_hashes or [""] * loaded.ntotal
        index.vectors = vectors
        return index

    def save(self, manifest: Manifest) -> None:
        """Persist index, vectors and mapping together.

        The manifest is written last and is the only record of which chunk each FAISS
        position holds. Writing one without the others leaves an index that resolves to
        the wrong text, which no later stage can detect.
        """
        self.config.paths.index.mkdir(parents=True, exist_ok=True)
        self._write_atomic(self.config.paths.faiss_index,
                           lambda tmp: self._faiss.write_index(self.index, str(tmp)))
        self._write_atomic(self.config.paths.embeddings,
                           lambda tmp: np.save(str(tmp), self.vectors, allow_pickle=False))
        manifest.data["faiss_id_map"] = list(self.chunk_ids)
        manifest.data["indexed_hashes"] = list(self.content_hashes)
        manifest.data["embedding_model"] = str(self.config.embedding.model)
        manifest.data["embedding_dim"] = self.dim
        manifest.save()

    @staticmethod
    def _write_atomic(path: Path, writer) -> None:
        """Write via a sibling temp file and rename, so a crash cannot truncate a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp")
        # np.save appends .npy unless the name already ends in it.
        target = temp if path.suffix != ".npy" else temp.with_suffix(".tmp.npy")
        try:
            writer(temp if path.suffix != ".npy" else temp)
            written = target if target.is_file() else temp
            os.replace(written, path)
        except BaseException:
            for leftover in (temp, target):
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)
            raise

    # ---- contents --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @property
    def indexed(self) -> set:
        """The chunk_ids already in the index — what makes appending incremental."""
        return set(self.chunk_ids)

    @property
    def indexed_hashes(self) -> Dict[str, str]:
        """chunk_id -> the content_hash whose text is actually embedded for it."""
        return dict(zip(self.chunk_ids, self.content_hashes))

    def is_stale(self, chunk_id: str, content_hash: str) -> bool:
        """True when this chunk is indexed but its text has changed since.

        chunk_id is positional within a paper, so it survives a re-ingest unchanged
        while the text under it does not — figure chunks gaining a description is
        exactly that case. Presence of the id alone therefore proves nothing.
        """
        current = self.indexed_hashes.get(chunk_id)
        return current is not None and current != content_hash

    def add(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        content_hashes: Optional[Sequence[str]] = None,
    ) -> int:
        """Append new vectors. Returns how many were added.

        Appending is the only write path here; replacing a vector in place is not
        possible in a flat index without renumbering, so changed content is handled by
        a rebuild in the indexer rather than by mutating this structure.
        """
        if len(chunk_ids) != vectors.shape[0]:
            raise IndexError_(
                f"{len(chunk_ids)} chunk_ids but {vectors.shape[0]} vectors; "
                "they are appended in lockstep and must match"
            )
        if content_hashes is not None and len(content_hashes) != len(chunk_ids):
            raise IndexError_(
                f"{len(chunk_ids)} chunk_ids but {len(content_hashes)} content_hashes; "
                "they are appended in lockstep and must match"
            )
        if vectors.shape[0] == 0:
            return 0
        if vectors.shape[1] != self.dim:
            raise IndexError_(f"vectors have dim {vectors.shape[1]}, index expects {self.dim}")

        array = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.add(array)
        self.chunk_ids.extend(str(chunk_id) for chunk_id in chunk_ids)
        self.content_hashes.extend(
            str(digest) for digest in (content_hashes or [""] * len(chunk_ids))
        )
        self.vectors = np.vstack([self.vectors, array]) if self.vectors.size else array
        return len(chunk_ids)

    def search(
        self, query_vector: np.ndarray, k: int, allowed: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        """Top-k (chunk_id, score) for a query vector, highest score first.

        `allowed` post-filters to a chunk_id subset; the search widens automatically so a
        filter cannot silently return fewer results than asked for.
        """
        if len(self) == 0 or k <= 0:
            return []

        query = np.ascontiguousarray(
            np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        )
        depth = min(len(self), k if allowed is None else max(k * 4, k + 50))
        scores, positions = self.index.search(query, depth)

        results: List[Tuple[str, float]] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[position]
            if allowed is not None and chunk_id not in allowed:
                continue
            results.append((chunk_id, float(score)))
            if len(results) >= k:
                break
        return results

    def vector_for(self, chunk_id: str) -> Optional[np.ndarray]:
        """The stored vector for a chunk, or None. Used by analyze_corpus clustering."""
        try:
            position = self.chunk_ids.index(chunk_id)
        except ValueError:
            return None
        return self.vectors[position] if position < self.vectors.shape[0] else None
