"""
`inspect_figure`: hand the agent the real image of a figure, table, or user file.

In:  a figure_id (or paper_id + figure_id, or chunk_id), or a user_image path.
Out: {"image_path", "mime_type", "caption", "stored_description", "paper_id", "page",
     "figure_id"} or an error dict. The image itself is attached by the agent loop on a
     follow-up user turn — see docs/ARCHITECTURE.md section 9 for why it cannot ride here.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

from config import CFG, Config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest, ManifestError

# Only these reach a vision model; anything else is a mistake worth naming.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


class FigureInspector:
    """Resolves a figure reference to an image on disk plus its stored metadata.

    Returns a path, never image bytes. The OpenAI-compatible surface only accepts image
    parts on a user message, so the loop attaches the file on a following turn; returning
    base64 here would also make this the one tool handing back a raw blob instead of a
    structured dict.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.config = config

    def inspect(
        self,
        paper_id: Optional[str] = None,
        figure_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        user_image: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve a figure or a user-supplied image to something the loop can attach."""
        if user_image:
            return self._user_image(user_image, question)

        if not figure_id and not chunk_id:
            return _error(
                "missing_reference",
                "give a figure_id from a retrieve_evidence result, a chunk_id, or a "
                "user_image path",
            )

        try:
            manifest = Manifest.load(self.config, strict_model_check=False)
        except ManifestError as exc:
            return _error("manifest_unreadable", str(exc))

        if not figure_id and chunk_id:
            chunk = next(
                (c for c in ChunkStore(config=self.config) if c["chunk_id"] == chunk_id), None
            )
            if chunk is None:
                return _error("unknown_chunk", f"no chunk with id {chunk_id!r}")
            figure_id = chunk.get("figure_id")
            if not figure_id:
                return _error(
                    "not_a_figure",
                    f"chunk {chunk_id!r} is a {chunk.get('chunk_type')} chunk with no image; "
                    "only figure and table chunks can be inspected",
                )

        record = manifest.figures.get(figure_id)
        if record is None:
            available = [f for f in manifest.figures if not paper_id or f.startswith(paper_id)]
            return _error(
                "unknown_figure",
                f"no figure with id {figure_id!r}"
                + (f". Figures for this paper: {available[:8]}" if available else ""),
            )
        if paper_id and record["paper_id"] != paper_id:
            return _error(
                "figure_paper_mismatch",
                f"figure {figure_id!r} belongs to {record['paper_id']}, not {paper_id!r}",
            )

        path = Path(record["image_path"])
        if not path.is_file():
            return _error("image_missing", f"the image file for {figure_id!r} is gone: {path}")

        return {
            "image_path": str(path),
            "mime_type": mimetypes.guess_type(path.name)[0] or "image/png",
            "caption": record.get("caption", ""),
            "stored_description": record.get("description", ""),
            "paper_id": record["paper_id"],
            "page": record["page"],
            "figure_id": figure_id,
            "kind": record.get("kind", "figure"),
            "question": question or "",
        }

    def _user_image(self, user_image: str, question: Optional[str]) -> Dict[str, Any]:
        """A user-supplied file. No caption or stored description exists for these."""
        path = Path(user_image).expanduser()
        if not path.is_file():
            return _error("image_missing", f"no file at {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            return _error(
                "not_an_image",
                f"{path.name} is not an image ({sorted(IMAGE_SUFFIXES)})",
            )
        return {
            "image_path": str(path),
            "mime_type": mimetypes.guess_type(path.name)[0] or "image/png",
            "source": "user",
            "question": question or "",
        }


INSPECT_FIGURE_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "figure_id": {
            "type": "string",
            "description": "The figure_id from a retrieve_evidence result, e.g. '2603_11114v1__f01'.",
        },
        "paper_id": {
            "type": "string",
            "description": "Optional paper the figure belongs to; checked against the figure.",
        },
        "chunk_id": {
            "type": "string",
            "description": "A figure or table chunk_id, as an alternative to figure_id.",
        },
        "user_image": {
            "type": "string",
            "description": "Path to an image the user supplied, instead of an indexed figure.",
        },
        "question": {
            "type": "string",
            "description": "What specifically to look for in the image.",
        },
    },
    "required": [],
    "additionalProperties": False,
}
