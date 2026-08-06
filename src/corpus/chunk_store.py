"""
Read/append/rewrite access to data/index/chunks.jsonl, one chunk record per line.

In:  chunk records to append, or a paper_id plus a tag to backfill.
Out: iteration over stored records, and atomic whole-file rewrites for in-place edits.
     Line order carries no meaning; `chunk_id` is the identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from config import CFG, Config
from common.records import add_topic_tag
from common.storage import append_jsonl, read_jsonl, write_jsonl_atomic


class ChunkStore:
    """chunks.jsonl. Appends are cheap; edits rewrite the whole file, atomically.

    Rewriting is acceptable at this corpus size (~1k lines) and is what keeps a
    partially-applied backfill impossible.
    """

    def __init__(self, path: Optional[Path] = None, config: Config = CFG) -> None:
        self.path = path if path is not None else config.paths.chunks

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return read_jsonl(self.path)

    def all(self) -> List[Dict[str, Any]]:
        """Every chunk record, in file order."""
        return list(read_jsonl(self.path))

    def count(self) -> int:
        return sum(1 for _ in read_jsonl(self.path))

    def for_paper(self, paper_id: str) -> List[Dict[str, Any]]:
        """This paper's chunks, sorted by `position` so neighbour walks are meaningful."""
        chunks = [c for c in read_jsonl(self.path) if c.get("paper_id") == paper_id]
        return sorted(chunks, key=lambda c: c.get("position", 0))

    def append(self, records: Sequence[Dict[str, Any]]) -> int:
        """Append new chunk records. Returns how many were written."""
        return append_jsonl(self.path, records)

    def rewrite(self, records: Sequence[Dict[str, Any]]) -> int:
        """Replace the whole file atomically. Returns how many were written."""
        return write_jsonl_atomic(self.path, records)

    def update_where(
        self, predicate: Callable[[Dict[str, Any]], bool], mutate: Callable[[Dict[str, Any]], bool]
    ) -> int:
        """Apply `mutate` to every record matching `predicate`; rewrite if any changed.

        Returns the number of records actually modified. The file is left untouched
        when nothing changed, so a no-op backfill costs one read and no write.
        """
        records = self.all()
        if not records:
            return 0
        changed = 0
        for record in records:
            if predicate(record) and mutate(record):
                changed += 1
        if changed:
            self.rewrite(records)
        return changed

    def add_topic_tag_to_paper(self, paper_id: str, tag: str) -> int:
        """Backfill `tag` onto every chunk of `paper_id`. Returns how many gained it.

        This is the other half of `Manifest.tag_paper`. Chunk records denormalise
        `topic_tags` so retrieval needs no join; the cost is that a tag added to a
        paper must reach its chunks in the same operation, or `topic_filter` silently
        returns nothing for that paper (docs/DATA_SCHEMA.md, Topic naming).
        """
        return self.update_where(
            lambda record: record.get("paper_id") == paper_id,
            lambda record: add_topic_tag(record, tag),
        )
