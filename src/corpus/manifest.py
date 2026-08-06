"""
Read/write wrapper over data/index/manifest.json, the corpus's index of record.

In:  paper records, figure records, topic entries, and the FAISS id map.
Out: a persisted manifest matching docs/DATA_SCHEMA.md section 4, plus lookups
     (has_paper, papers_for_topic) and the manifest_hash caches are keyed on.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CFG, Config
from common.records import SCHEMA_VERSION, RecordError, add_topic_tag, utc_now, validate_topic_tags
from common.storage import read_json, write_json_atomic


class ManifestError(RuntimeError):
    """The manifest on disk is unusable — for example a changed embedding model."""


def empty_manifest(embedding_model: str, embedding_dim: int) -> Dict[str, Any]:
    """A manifest with no papers, stamped with the embedding model it belongs to."""
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "embedding_dim": int(embedding_dim),
        "created_at": now,
        "updated_at": now,
        "papers": {},
        "figures": {},
        "topics": {},
        "faiss_id_map": [],
        "content_hashes": {},
        "manifest_hash": "",
    }


class Manifest:
    """The manifest, loaded fully into memory. It is small enough that this is fine.

    Nothing is written until `save()`; `save()` is atomic, so a crash mid-write leaves
    the previous manifest intact rather than a truncated one.
    """

    def __init__(self, data: Dict[str, Any], path: Path) -> None:
        self.data = data
        self.path = path

    # ---- lifecycle -------------------------------------------------------------

    @classmethod
    def load(cls, config: Config = CFG, strict_model_check: bool = True) -> "Manifest":
        """Load the manifest, creating an empty one when absent.

        A different `embedding_model` than the stored one means the existing vectors
        are incomparable with any new ones, so this raises instead of appending — the
        mismatch check is mandatory, not advisory (docs/DATA_SCHEMA.md section 4).
        """
        path = config.paths.manifest
        data = read_json(path)
        if not isinstance(data, dict):
            return cls(empty_manifest(config.embedding.model, config.embedding.dim), path)

        if strict_model_check and data.get("papers"):
            stored_model = data.get("embedding_model")
            stored_dim = data.get("embedding_dim")
            if stored_model != config.embedding.model or stored_dim != config.embedding.dim:
                raise ManifestError(
                    f"manifest was built with {stored_model} (dim {stored_dim}) but config "
                    f"says {config.embedding.model} (dim {config.embedding.dim}). Appending "
                    "vectors from a different model silently corrupts retrieval; the index "
                    "must be rebuilt."
                )
        return cls(data, path)

    def save(self) -> None:
        """Persist atomically, refreshing `updated_at` and `manifest_hash`."""
        self.data["updated_at"] = utc_now()
        self.data["manifest_hash"] = self.compute_hash()
        write_json_atomic(self.path, self.data)

    # ---- papers ----------------------------------------------------------------

    @property
    def papers(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("papers", {})

    def has_paper(self, paper_id: str) -> bool:
        return paper_id in self.papers

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.papers.get(paper_id)

    def add_paper(self, record: Dict[str, Any]) -> None:
        """Insert a new paper record. Use `tag_paper` for one that already exists."""
        paper_id = record["paper_id"]
        if paper_id in self.papers:
            raise RecordError(f"paper {paper_id} already in the manifest; use tag_paper")
        validate_topic_tags(record.get("topic_tags", []), f"papers.{paper_id}.topic_tags")
        self.papers[paper_id] = record

    def tag_paper(self, paper_id: str, tag: str) -> bool:
        """Append a topic tag to an existing paper record. True when it was new.

        The paper record is only half the job: its chunk records carry their own copy
        of `topic_tags`, and the caller must backfill those in the same operation.
        """
        record = self.papers.get(paper_id)
        if record is None:
            raise RecordError(f"paper {paper_id} is not in the manifest")
        return add_topic_tag(record, tag)

    # ---- topics ----------------------------------------------------------------

    @property
    def topics(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("topics", {})

    def record_topic(self, topic_tag: str, query: str, arxiv_query: str, paper_ids: List[str]) -> None:
        """Create or update one topic entry, merging paper ids and touching last_accessed."""
        now = utc_now()
        entry = self.topics.get(topic_tag)
        if entry is None:
            self.topics[topic_tag] = {
                "query": query,
                "arxiv_query": arxiv_query,
                "paper_ids": list(dict.fromkeys(paper_ids)),
                "created_at": now,
                "last_accessed": now,
            }
            return
        entry["arxiv_query"] = arxiv_query
        entry["paper_ids"] = list(dict.fromkeys([*entry.get("paper_ids", []), *paper_ids]))
        entry["last_accessed"] = now

    def papers_for_topic(self, topic_tag: str) -> List[str]:
        return list(self.topics.get(topic_tag, {}).get("paper_ids", []))

    # ---- figures ---------------------------------------------------------------

    @property
    def figures(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("figures", {})

    # ---- derived ---------------------------------------------------------------

    def total_chunks(self) -> int:
        return sum(int(paper.get("n_chunks", 0)) for paper in self.papers.values())

    def compute_hash(self) -> str:
        """sha256 over sorted paper ids plus the chunk count — the cluster cache key."""
        payload = "|".join(sorted(self.papers)) + f"#{self.total_chunks()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> Dict[str, Any]:
        """Small dict for logging and tool returns."""
        return {
            "papers": len(self.papers),
            "chunks": self.total_chunks(),
            "figures": len(self.figures),
            "topics": sorted(self.topics),
        }
