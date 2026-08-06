"""
PyMuPDF text extraction: a PDF into clean, page-numbered, section-labelled lines.

In:  a path to a PDF.
Out: {"pages": [{"page", "lines": [{"text", "section"}]}], "status", "note"} — repeated
     headers/footers and the arXiv margin stamp removed, headings detected where found.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from config import CFG, Config

# The vertical stamp arXiv prints down the left margin of the first page.
_ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}v\d+", re.IGNORECASE)
# A line that is only a page number, or "3 of 12".
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/|of)?\s*\d*\s*$", re.IGNORECASE)

# Canonical section names, matched with or without a leading number.
_CANONICAL = (
    "abstract", "introduction", "background", "related work", "preliminaries",
    "method", "methods", "methodology", "approach", "model", "architecture",
    "experiments", "experimental setup", "experimental results", "setup",
    "results", "evaluation", "analysis", "ablation", "ablations", "ablation study",
    "discussion", "limitations", "conclusion", "conclusions", "future work",
    "references", "acknowledgements", "acknowledgments", "appendix",
    "broader impact", "related works",
)
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^.]{2,60})\s*$")
_APPENDIX_HEADING = re.compile(r"^\s*(appendix\s+[A-Z0-9]+)\b[.:]?\s*(.{0,60})$", re.IGNORECASE)
_LETTERS = re.compile(r"[A-Za-z]")


def _normalise_section(text: str) -> str:
    """A heading reduced to its bare name: '5. References' -> 'references'."""
    return re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text).strip().lower().rstrip(":.")


def _is_canonical(text: str) -> bool:
    stripped = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", text).strip().lower().rstrip(":.")
    return stripped in _CANONICAL


def detect_heading(
    text: str, size: float, body_size: float, bold: bool, size_delta: float = 1.5
) -> Optional[str]:
    """Return a normalised heading for `text`, or None if it is body text.

    Font size carries the decision. Typography separates the three classes cleanly on
    real papers — headings sit well above the body size, numbered contribution lists sit
    just above it, and table cells sit at or below it — whereas font *name* does not:
    plenty of arXiv PDFs embed fonts whose names never say "bold".
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 80:
        return None
    # Kills table cells like "500M", "2297M", "N/A" that typography alone would promote.
    if len(_LETTERS.findall(stripped)) < 3:
        return None
    if stripped.endswith((".", ",", ";")) and not _is_canonical(stripped):
        return None

    prominent = bold or size >= body_size + size_delta

    if _is_canonical(stripped):
        # A canonical name in a table cell is still body text, so it must not be smaller
        # than the surrounding prose.
        return stripped.rstrip(":.").strip() if size >= body_size else None

    if _APPENDIX_HEADING.match(stripped):
        return stripped.rstrip(":.").strip()

    if _NUMBERED_HEADING.match(stripped):
        # Numbered contribution lists ("1. We propose ...") look identical to numbered
        # headings apart from their size, so prominence is what separates them.
        return stripped.rstrip(":.").strip() if prominent else None

    # A non-canonical heading is at least two words. One prominent word is far more
    # often a table header ("PSPNet", "BDD100K") than a section title, and those were
    # leaking through as sections.
    words = len(stripped.split())
    if words < 2:
        return None

    # All-caps short lines are headings in several arXiv styles.
    if stripped.isupper() and 3 <= len(stripped) <= 40:
        return stripped.title()

    if prominent and words <= 8:
        return stripped.rstrip(":.").strip()

    return None


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def body_font_size(pages: List[List[Tuple[str, float, bool]]], default: float = 10.0) -> float:
    """The dominant body font size, as the character-weighted mode of all line sizes.

    Weighting by characters rather than counting lines is what makes this robust: body
    prose is a minority of *lines* in a paper full of short caption, table and reference
    lines, so a median over lines lands too low and real body text then reads as
    prominent enough to be a heading.
    """
    weight: Counter = Counter()
    for lines in pages:
        for text, size, _ in lines:
            if size > 0:
                weight[round(size * 2) / 2] += len(text)
    if not weight:
        return default
    return weight.most_common(1)[0][0]


def _raw_lines(document: "fitz.Document") -> List[List[Tuple[str, float, bool]]]:
    """Per page, the (text, max font size, any-bold) of each line, in reading order."""
    pages: List[List[Tuple[str, float, bool]]] = []
    for page in document:
        lines: List[Tuple[str, float, bool]] = []
        data = page.get_text("dict")
        for block in data.get("blocks", []):
            if block.get("type") != 0:  # 0 is text; 1 is an image
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                size = max((span.get("size", 0.0) for span in spans), default=0.0)
                bold = any("bold" in str(span.get("font", "")).lower() for span in spans)
                lines.append((text, size, bold))
        pages.append(lines)
    return pages


def _repeated_lines(pages: List[List[Tuple[str, float, bool]]], threshold: float = 0.5) -> set:
    """Lines appearing near the top or bottom of at least `threshold` of pages.

    Running heads and footers repeat; body text does not. Only the first and last few
    lines of each page are considered, so a genuinely repeated sentence in the body is
    not stripped.
    """
    if len(pages) < 3:
        return set()
    counts: Counter = Counter()
    for lines in pages:
        edge = {text for text, _, _ in lines[:2]} | {text for text, _, _ in lines[-2:]}
        counts.update(edge)
    minimum = max(2, int(len(pages) * threshold))
    return {text for text, count in counts.items() if count >= minimum}


def extract_pages(pdf_path: Path, config: Config = CFG) -> Dict[str, Any]:
    """Extract a PDF into per-page lines tagged with the section they fall under.

    Returns status 'ok', 'partial' (some pages failed) or 'failed', never raising —
    a corrupt PDF is a normal outcome that the caller records and moves past.
    """
    path = Path(pdf_path)
    if not path.is_file():
        return {"pages": [], "status": "failed", "note": f"no file at {path}", "n_pages": 0}

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        return {"pages": [], "status": "failed", "note": f"could not open: {exc}", "n_pages": 0}

    try:
        raw = _raw_lines(document)
        n_pages = document.page_count
    except Exception as exc:
        document.close()
        return {"pages": [], "status": "failed", "note": f"text extraction failed: {exc}", "n_pages": 0}
    finally:
        if not document.is_closed:
            document.close()

    body_size = body_font_size(raw)
    size_delta = float(config.extraction.heading_size_delta)
    boilerplate = _repeated_lines(raw)

    drop_sections = {str(name).strip().lower()
                     for name in (config.extraction.get("drop_sections") or [])}

    pages: List[Dict[str, Any]] = []
    section: Optional[str] = None
    dropping = False
    kept = 0

    for index, lines in enumerate(raw, start=1):
        page_lines: List[Dict[str, Any]] = []
        for text, size, bold in lines:
            if text in boilerplate or _PAGE_NUMBER.match(text) or _ARXIV_STAMP.search(text):
                continue
            heading = detect_heading(text, size, body_size, bold, size_delta)

            if dropping:
                # Inside a dropped section, only an appendix heading can end the skip.
                # Nothing else is trusted: reference entries throw off heading detection
                # badly, which is how "[Iccv, 2021. 2]" ends up looking like a section.
                if heading and _APPENDIX_HEADING.match(heading):
                    dropping, section = False, heading
                continue

            if heading:
                if _normalise_section(heading) in drop_sections:
                    dropping = True
                    continue
                section = heading
                continue  # the heading labels what follows; it is not body text

            page_lines.append({"text": text, "section": section})
            kept += 1
        pages.append({"page": index, "lines": page_lines})

    if kept == 0:
        return {
            "pages": pages,
            "status": "failed",
            "note": "no extractable text — the PDF is probably scanned images",
            "n_pages": n_pages,
        }

    status = "ok" if len(pages) == n_pages else "partial"
    return {"pages": pages, "status": status, "note": "", "n_pages": n_pages}
