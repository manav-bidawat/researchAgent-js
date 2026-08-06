"""
Extracts figure and table images from a PDF, with the caption printed next to them.

In:  a path to a PDF and the extraction caps from config.
Out: [{"figure_id", "kind", "page", "image_path", "image_hash", "caption", "width",
     "height"}] — decorative images filtered out, duplicates collapsed by image hash.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from config import CFG, Config
from common.records import figure_id as make_figure_id

# "Figure 3:", "Fig. 3.", "Table 2 —" — the label that opens a caption.
#
# The separator is mandatory. Without it this also matches prose that merely *refers* to
# a figure ("Figure 1 shows the similarity matrix..."), and since that sentence usually
# appears before the figure itself, the reference claims the label and the real caption
# is then discarded as a duplicate.
_CAPTION_START = re.compile(
    r"^\s*(?P<kind>figure|fig\.|table|tab\.)\s*(?P<number>\d+[a-z]?)\s*(?P<sep>[.:—–])\s+",
    re.IGNORECASE,
)


def _kind_of(label: str) -> str:
    return "table" if label.lower().startswith(("table", "tab")) else "figure"


def image_hash(payload: bytes) -> str:
    """sha256 of the raw image bytes — the cache key for the generated description."""
    return hashlib.sha256(payload).hexdigest()


def _caption_blocks(page: "fitz.Page") -> List[Tuple[float, float, str]]:
    """(top, bottom, text) for every text block on the page that opens like a caption."""
    found: List[Tuple[float, float, str]] = []
    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], str(block[4])
        cleaned = " ".join(text.split())
        if _CAPTION_START.match(cleaned):
            found.append((y0, y1, cleaned))
    return found


def find_caption(
    page: "fitz.Page", rect: "fitz.Rect", search_chars: int
) -> Tuple[str, Optional[str]]:
    """The caption belonging to the image at `rect`, and its kind.

    Figures are captioned below and tables above, so the nearest caption block in either
    direction is taken, preferring the one below on a tie. Returns ("", None) when the
    image has no caption near it.
    """
    candidates = _caption_blocks(page)
    if not candidates:
        return "", None

    best: Optional[Tuple[float, str]] = None
    for top, bottom, text in candidates:
        if top >= rect.y1:
            distance = top - rect.y1          # caption sits below the image
        elif bottom <= rect.y0:
            distance = (rect.y0 - bottom) + 1  # above: same distance ranks slightly worse
        else:
            continue                           # overlapping the image itself
        if best is None or distance < best[0]:
            best = (distance, text)

    if best is None:
        return "", None

    caption = best[1][:search_chars].strip()
    match = _CAPTION_START.match(caption)
    return caption, _kind_of(match.group("kind")) if match else None


# A figure's own panel labels and axis ticks are text blocks too. Only a block with at
# least this many words is treated as prose that bounds the figure region; below it, the
# block is assumed to live inside the figure. Without this, "(a) Dense ViT" printed under
# a panel collapses the region to nothing and the figure is lost entirely.
_BOUNDARY_MIN_WORDS = 12


def _text_blocks(page: "fitz.Page") -> List[Tuple[float, float, float, float, str]]:
    """(x0, y0, x1, y1, text) for every text block on the page."""
    out = []
    for block in page.get_text("blocks"):
        if len(block) >= 5 and str(block[4]).strip():
            out.append((block[0], block[1], block[2], block[3], " ".join(str(block[4]).split())))
    return out


def _boundary_blocks(page: "fitz.Page") -> List[Tuple[float, float, float, float, str]]:
    """Text blocks substantial enough to mark where a figure stops."""
    return [b for b in _text_blocks(page) if len(b[4].split()) >= _BOUNDARY_MIN_WORDS]


def figure_region(
    page: "fitz.Page", caption: Tuple[float, float, float, float, str], kind: str
) -> Optional["fitz.Rect"]:
    """The rectangle the figure occupies, given its caption block.

    Driving this from the caption rather than from embedded images is what makes vector
    figures work: most arXiv figures are drawn as vector graphics with no raster to
    extract, and a multi-panel figure that *is* raster arrives as several separate
    images sharing one caption. The region covers both cases identically.
    """
    cx0, cy0, cx1, cy1, _ = caption
    page_rect = page.rect
    # Stay inside the caption's column so a two-column layout does not capture its neighbour.
    x0, x1 = max(page_rect.x0, cx0 - 8), min(page_rect.x1, cx1 + 8)

    if kind == "table":
        # Table captions sit above their table, figures below theirs.
        boundary = page_rect.y1 - 20
        for bx0, by0, bx1, by1, _ in _boundary_blocks(page):
            if by0 > cy1 and bx1 > x0 and bx0 < x1:
                boundary = min(boundary, by0)
        top, bottom = cy1 + 2, boundary - 2
    else:
        boundary = page_rect.y0 + 20
        for bx0, by0, bx1, by1, _ in _boundary_blocks(page):
            if by1 < cy0 and bx1 > x0 and bx0 < x1:
                boundary = max(boundary, by1)
        top, bottom = boundary + 2, cy0 - 2

    # Running heads sit in the top margin and are too short to count as boundary prose,
    # so clamp below them rather than rendering the header into the figure.
    top = max(top, page_rect.y0 + 0.06 * page_rect.height)

    if bottom - top < 20 or x1 - x0 < 20:
        return None
    return fitz.Rect(x0, top, x1, bottom)


def extract_figures(
    pdf_path: Path, paper_id: str, config: Config = CFG
) -> Dict[str, Any]:
    """Extract captioned figure and table regions from a PDF, rendered to PNG.

    One image per caption, so a multi-panel figure is one record rather than one per
    panel. Regions below the configured minimum size are skipped. Never raises.
    """
    path = Path(pdf_path)
    if not path.is_file():
        return {"figures": [], "status": "failed", "note": f"no file at {path}"}

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        return {"figures": [], "status": "failed", "note": f"could not open: {exc}"}

    min_width = int(config.extraction.min_figure_width)
    min_height = int(config.extraction.min_figure_height)
    limit = int(config.extraction.max_figures_per_paper)
    search_chars = int(config.extraction.caption_search_chars)
    zoom = float(config.extraction.render_zoom)
    out_dir = config.paths.figure_dir(paper_id)

    figures: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    seen_labels: set = set()
    skipped_small = 0
    note = ""

    try:
        for page_index, page in enumerate(document, start=1):
            if len(figures) >= limit:
                break
            for block in _text_blocks(page):
                if len(figures) >= limit:
                    break
                text = block[4]
                match = _CAPTION_START.match(text)
                if not match:
                    continue

                kind = _kind_of(match.group("kind"))
                label = f"{kind}-{match.group('number').lower()}"
                if label in seen_labels:
                    continue  # a caption repeated across panels or continued overleaf
                region = figure_region(page, block, kind)
                if region is None:
                    continue

                try:
                    pixmap = page.get_pixmap(clip=region, matrix=fitz.Matrix(zoom, zoom))
                except Exception:
                    continue
                if pixmap.width < min_width or pixmap.height < min_height:
                    skipped_small += 1
                    continue

                payload = pixmap.tobytes("png")
                digest = image_hash(payload)
                if digest in seen_hashes:
                    continue

                identifier = make_figure_id(paper_id, len(figures) + 1)
                out_dir.mkdir(parents=True, exist_ok=True)
                image_path = out_dir / f"{identifier}.png"
                try:
                    image_path.write_bytes(payload)
                except OSError as exc:
                    note = f"could not write {image_path.name}: {exc}"
                    continue

                seen_hashes.add(digest)
                seen_labels.add(label)
                figures.append(
                    {
                        "figure_id": identifier,
                        "paper_id": paper_id,
                        "kind": kind,
                        "page": page_index,
                        "image_path": str(image_path),
                        "image_hash": digest,
                        "caption": text[:search_chars].strip(),
                        "width": pixmap.width,
                        "height": pixmap.height,
                    }
                )
    except Exception as exc:
        document.close()
        return {
            "figures": figures,
            "status": "partial" if figures else "failed",
            "note": f"figure extraction failed partway: {exc}",
        }

    document.close()
    return {"figures": figures, "status": "ok", "note": note, "skipped_small": skipped_small}
