"""
Turns a downloaded paper into chunk records: the M2 pipeline, end to end.

In:  a paper_id already present in the manifest, plus an LLMClient for figure description.
Out: {"chunks_added", "figures_added", "parse_status", ...}; chunk records appended to
     chunks.jsonl and figure records written to the manifest. Never raises.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from corpus.chunk_store import ChunkStore
from extraction.chunker import Chunker
from config import CFG, Config
from extraction.describe import describe_all, figure_chunk_text
from extraction.figures import extract_figures
from llm_client import LLMClient
from corpus.manifest import Manifest
from extraction.pdf_text import extract_pages
from common.records import RecordError, chunk_record, content_hash, utc_now
from common.tokenization import TokenCounter


def _ordered(
    text_chunks: List[Dict[str, Any]], figure_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Interleave figure chunks into the text in reading order, by page.

    `position` has to be contiguous *and* meaningful, because neighbour expansion walks
    it: a figure appended after the last text chunk would make `position ± 1` step from
    the end of the conclusion into a figure on page 2. Placing each figure after the
    text of its own page keeps the walk local.
    """
    items: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for order, chunk in enumerate(text_chunks):
        # Rank 0 puts a page's prose before that page's figures.
        items.append((int(chunk.get("page", 1)), 0, order, chunk))
    for order, chunk in enumerate(figure_chunks):
        items.append((int(chunk.get("page", 1)), 1, order, chunk))
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in items]


def _dedupe_within_paper(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Drop repeats of identical text *within this paper*, keeping the first occurrence.

    Only within the paper. Two papers sharing text keep both chunk records — the hash is
    an embedding-cache key, not an identity (docs/DATA_SCHEMA.md §2).
    """
    seen: set = set()
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for item in items:
        digest = content_hash(item["text"])
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(item)
    return kept, dropped


def ingest_paper(
    paper_id: str,
    manifest: Optional[Manifest] = None,
    store: Optional[ChunkStore] = None,
    client: Optional[LLMClient] = None,
    config: Config = CFG,
    describe: bool = True,
) -> Dict[str, Any]:
    """Extract, chunk, describe figures, and append chunk records for one paper.

    Returns an error dict when the paper is unknown or its text cannot be extracted.
    A paper whose text works but whose figures fail is kept as parse_status 'partial'.
    """
    owns_manifest = manifest is None
    manifest = manifest if manifest is not None else Manifest.load(config)
    store = store if store is not None else ChunkStore(config=config)

    paper = manifest.get_paper(paper_id)
    if paper is None:
        return {"error": "unknown_paper", "detail": f"{paper_id} is not in the manifest",
                "partial": None}

    if store.for_paper(paper_id):
        return {"error": "already_ingested",
                "detail": f"{paper_id} already has chunks; nothing to do", "partial": None}

    extracted = extract_pages(config.paths.paper_pdf(paper_id), config)
    if extracted["status"] == "failed":
        paper["parse_status"] = "failed"
        paper["parse_note"] = extracted["note"]
        manifest.save()
        return {"error": "parse_failed", "detail": extracted["note"], "partial": None}

    counter = TokenCounter(config)
    text_chunks = Chunker(config, counter).chunk_pages(extracted["pages"])

    figure_result = extract_figures(config.paths.paper_pdf(paper_id), paper_id, config)
    figures = figure_result.get("figures", [])
    if describe and figures:
        figures = describe_all(figures, client=client, config=config)
    else:
        figures = [{**figure, "description": "", "description_cached": False} for figure in figures]

    figure_chunks: List[Dict[str, Any]] = []
    for figure in figures:
        text = figure_chunk_text(figure.get("caption", ""), figure.get("description", ""))
        if not text.strip():
            continue  # no caption and no description: nothing retrievable to embed
        figure_chunks.append(
            {
                "text": text,
                "page": figure["page"],
                "section": None,
                "n_tokens": counter.count(text),
                "figure_id": figure["figure_id"],
                "image_path": figure["image_path"],
                "caption": figure.get("caption", ""),
                "chunk_type": figure.get("kind", "figure"),
            }
        )

    ordered = _ordered(text_chunks, figure_chunks)
    deduped, dropped = _dedupe_within_paper(ordered)

    records: List[Dict[str, Any]] = []
    try:
        # position is assigned only now, after dedup, so it is dense: 0..n-1 with no holes.
        for position, item in enumerate(deduped):
            records.append(
                chunk_record(
                    paper_id=paper_id,
                    paper_title=paper["title"],
                    topic_tags=paper["topic_tags"],
                    chunk_type=item.get("chunk_type", "text"),
                    text=item["text"],
                    page=item["page"],
                    position=position,
                    n_tokens=item["n_tokens"],
                    section=item.get("section"),
                    figure_id_value=item.get("figure_id"),
                    image_path=item.get("image_path"),
                    caption=item.get("caption"),
                )
            )
    except RecordError as exc:
        return {"error": "bad_chunk_record", "detail": str(exc), "partial": None}

    if not records:
        paper["parse_status"] = "failed"
        paper["parse_note"] = "no usable chunks were produced"
        manifest.save()
        return {"error": "no_chunks", "detail": "no usable chunks were produced", "partial": None}

    store.append(records)

    by_figure_id = {record["figure_id"]: record["chunk_id"] for record in records
                    if record.get("figure_id")}
    for figure in figures:
        identifier = figure["figure_id"]
        if identifier not in by_figure_id:
            continue
        manifest.figures[identifier] = {
            "figure_id": identifier,
            "paper_id": paper_id,
            "kind": figure.get("kind", "figure"),
            "page": figure["page"],
            "image_path": figure["image_path"],
            "image_hash": figure["image_hash"],
            "caption": figure.get("caption", ""),
            "description": figure.get("description", ""),
            "width": figure["width"],
            "height": figure["height"],
            "chunk_id": by_figure_id[identifier],
        }

    figures_failed = figure_result.get("status") == "failed" or any(
        figure.get("description_error") for figure in figures
    )
    paper["n_chunks"] = len(records)
    paper["n_figures"] = len(by_figure_id)
    paper["parse_status"] = "partial" if figures_failed else "ok"
    paper["parse_note"] = figure_result.get("note", "") or (
        "some figure descriptions failed" if figures_failed else ""
    )
    paper["indexed_at"] = utc_now()
    manifest.save()

    return {
        "paper_id": paper_id,
        "chunks_added": len(records),
        "text_chunks": sum(1 for r in records if r["chunk_type"] == "text"),
        "figures_added": len(by_figure_id),
        "duplicates_dropped": dropped,
        "vision_calls": sum(1 for f in figures if not f.get("description_cached")
                            and f.get("description")),
        "parse_status": paper["parse_status"],
        "pages": extracted["n_pages"],
    }


def ingest_all(
    client: Optional[LLMClient] = None, config: Config = CFG, describe: bool = True
) -> Dict[str, Any]:
    """Ingest every manifest paper that has no chunks yet."""
    manifest = Manifest.load(config)
    store = ChunkStore(config=config)
    results = []
    for paper_id in list(manifest.papers):
        result = ingest_paper(paper_id, manifest=manifest, store=store, client=client,
                              config=config, describe=describe)
        results.append({"paper_id": paper_id, **result})
    return {
        "papers": results,
        "chunks_total": store.count(),
        "figures_total": len(manifest.figures),
    }
