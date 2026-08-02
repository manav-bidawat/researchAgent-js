"""
Constructors and validators for the record shapes in docs/DATA_SCHEMA.md.

In:  raw field values — an arXiv result's fields, a topic tag, extracted chunk text.
Out: dicts matching the documented schema exactly, or a RecordError naming the field
     that is wrong. Every record in the system is built here, never by hand.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

SCHEMA_VERSION = 1

PARSE_STATUSES = ("ok", "partial", "failed")
CHUNK_TYPES = ("text", "figure", "table")

# arXiv ids contain dots and (for pre-2007 ids) slashes; both are unsafe in filenames.
_UNSAFE_IN_ID = re.compile(r"[./\\]")


class RecordError(ValueError):
    """A record was built with a field that violates docs/DATA_SCHEMA.md."""


def utc_now() -> str:
    """Current time as an ISO 8601 UTC string, the timestamp format used everywhere."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    """sha256 of chunk text. An embedding-cache key — never a cross-paper dedup key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def topic_hash(topic: str) -> str:
    """sha256 of a natural-language topic, used to cache its planned arXiv query."""
    return hashlib.sha256(topic.strip().lower().encode("utf-8")).hexdigest()


def normalise_paper_id(arxiv_id: str) -> str:
    """'2103.14030v2' -> '2103_14030v2'. Dots and slashes are unsafe in filenames."""
    if not arxiv_id or not arxiv_id.strip():
        raise RecordError("arxiv_id is empty")
    return _UNSAFE_IN_ID.sub("_", arxiv_id.strip())


def validate_topic_tags(topic_tags: Any, field: str = "topic_tags") -> List[str]:
    """Return `topic_tags` as a list of non-empty strings, or raise.

    A bare string must not pass. It survives every `in` test character-by-character
    ("cs" in "cs.LG" is True), so a scalar that slips through does not fail loudly —
    it silently turns `topic_filter` into a substring match.
    """
    if isinstance(topic_tags, str):
        raise RecordError(f"{field} must be list[str], got a bare string {topic_tags!r}")
    if not isinstance(topic_tags, (list, tuple)):
        raise RecordError(f"{field} must be list[str], got {type(topic_tags).__name__}")
    tags: List[str] = []
    for tag in topic_tags:
        if not isinstance(tag, str) or not tag.strip():
            raise RecordError(f"{field} contains a non-string or empty tag: {tag!r}")
        if tag not in tags:
            tags.append(tag)
    return tags


def paper_record(
    *,
    arxiv_id: str,
    title: str,
    authors: Sequence[str],
    abstract: str,
    published: str,
    year: int,
    categories: Sequence[str],
    pdf_url: str,
    pdf_path: str,
    topic_tags: Sequence[str],
    n_chunks: int = 0,
    n_figures: int = 0,
    parse_status: str = "ok",
    parse_note: str = "",
    indexed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a paper record per docs/DATA_SCHEMA.md section 1."""
    if parse_status not in PARSE_STATUSES:
        raise RecordError(f"parse_status must be one of {PARSE_STATUSES}, got {parse_status!r}")
    if not title or not title.strip():
        raise RecordError("title is empty")

    return {
        "paper_id": normalise_paper_id(arxiv_id),
        "arxiv_id": arxiv_id.strip(),
        "title": " ".join(title.split()),
        "authors": [str(author) for author in authors],
        "abstract": " ".join(abstract.split()),
        "year": int(year),
        "published": published,
        "categories": [str(category) for category in categories],
        "pdf_url": pdf_url,
        "pdf_path": pdf_path,
        "topic_tags": validate_topic_tags(topic_tags),
        "n_chunks": int(n_chunks),
        "n_figures": int(n_figures),
        "indexed_at": indexed_at or utc_now(),
        "parse_status": parse_status,
        "parse_note": parse_note,
    }


def chunk_id(paper_id: str, position: int) -> str:
    """'{paper_id}__c{NNNN}'. `position` is the post-dedup ordinal within the paper."""
    return f"{paper_id}__c{position:04d}"


def figure_id(paper_id: str, ordinal: int) -> str:
    """'{paper_id}__f{NN}'."""
    return f"{paper_id}__f{ordinal:02d}"


def chunk_record(
    *,
    paper_id: str,
    paper_title: str,
    topic_tags: Sequence[str],
    chunk_type: str,
    text: str,
    page: int,
    position: int,
    n_tokens: int,
    section: Optional[str] = None,
    figure_id_value: Optional[str] = None,
    image_path: Optional[str] = None,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a chunk record per docs/DATA_SCHEMA.md section 2.

    `position` must already be the dense post-dedup ordinal; this does not assign it.
    """
    if chunk_type not in CHUNK_TYPES:
        raise RecordError(f"chunk_type must be one of {CHUNK_TYPES}, got {chunk_type!r}")
    if chunk_type != "text" and not figure_id_value:
        raise RecordError(f"chunk_type {chunk_type!r} requires a figure_id")
    if position < 0:
        raise RecordError(f"position must be non-negative, got {position}")

    return {
        "chunk_id": chunk_id(paper_id, position),
        "paper_id": paper_id,
        "paper_title": paper_title,
        "topic_tags": validate_topic_tags(topic_tags),
        "chunk_type": chunk_type,
        "text": text,
        "page": int(page),
        "section": section,
        "position": int(position),
        "n_tokens": int(n_tokens),
        "content_hash": content_hash(text),
        "figure_id": figure_id_value,
        "image_path": image_path,
        "caption": caption,
    }


def add_topic_tag(record: Dict[str, Any], tag: str) -> bool:
    """Append `tag` to a record's topic_tags in place. Returns True if it was new.

    Used on a dedup hit. The caller must apply this to the paper record *and* every
    chunk record that paper produced — chunks denormalise topic_tags, so patching only
    the manifest leaves `topic_filter` silently returning nothing for that paper.
    """
    if not isinstance(tag, str) or not tag.strip():
        raise RecordError(f"tag must be a non-empty string, got {tag!r}")
    tags = validate_topic_tags(record.get("topic_tags", []))
    if tag in tags:
        record["topic_tags"] = tags
        return False
    record["topic_tags"] = [*tags, tag]
    return True
