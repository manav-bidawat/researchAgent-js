"""
Generates a text description of a figure or table image with one vision-LLM call.

In:  figure records from figures.extract_figures, and an LLMClient.
Out: the same records with a "description" field, in input order, cached by image_hash in
     data/cache/descriptions.json so a re-index costs no vision calls at all.
Calls are issued concurrently (extraction.describe_workers) — they are network-bound.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import CFG, Config
from llm_client import LLMClient, LLMError
from common.storage import read_json, write_json_atomic

# The model is told to emit this when the image is not a readable figure.
UNREADABLE = "UNREADABLE"


class DescriptionCache:
    """image_hash -> description, persisted as JSON.

    Keyed by content hash rather than by path, so the same image costs one vision call
    however many papers or re-indexes it appears in.

    Every method takes a lock: describe_all fans its calls out across threads, and they
    all share one cache instance so the file is still written exactly once.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.path = config.paths.descriptions
        self._data: Dict[str, str] = read_json(self.path, default={}) or {}
        self._dirty = False
        self._lock = threading.Lock()

    def get(self, image_hash: str) -> Optional[str]:
        with self._lock:
            value = self._data.get(image_hash)
        return value if isinstance(value, str) and value.strip() else None

    def put(self, image_hash: str, description: str) -> None:
        with self._lock:
            self._data[image_hash] = description
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            snapshot = dict(self._data)
            self._dirty = False
        write_json_atomic(self.path, snapshot)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _build_messages(caption: str, kind: str, config: Config) -> List[Dict[str, Any]]:
    caption_line = caption.strip() or "(no caption was found near this image)"
    return [
        {"role": "system", "content": config.prompt("figure_description")},
        {"role": "user", "content": f"This is a {kind} from a scientific paper.\nCaption: {caption_line}"},
    ]


def describe_figure(
    figure: Dict[str, Any],
    client: Optional[LLMClient] = None,
    cache: Optional[DescriptionCache] = None,
    config: Config = CFG,
) -> Dict[str, Any]:
    """Describe one figure, using the cache when the same image was seen before.

    Returns the figure record with "description" and "description_cached" set. A failed
    call yields an empty description rather than raising: the caption alone still makes
    the figure findable, so a vision outage degrades quality without losing the record.
    """
    owns_cache = cache is None
    cache = cache if cache is not None else DescriptionCache(config)
    image_hash = figure.get("image_hash", "")

    cached = cache.get(image_hash)
    if cached is not None:
        return {**figure, "description": cached, "description_cached": True}

    llm = client if client is not None else LLMClient(config=config)
    try:
        response = llm.complete_vision(
            _build_messages(figure.get("caption", ""), figure.get("kind", "figure"), config),
            [figure["image_path"]],
            role="vision",
        )
        description = " ".join((response.text or "").split())
    except (LLMError, KeyError, OSError) as exc:
        return {**figure, "description": "", "description_cached": False,
                "description_error": str(exc)}

    if not description or description.strip().upper().startswith(UNREADABLE):
        # Cache the verdict too: re-asking about the same unreadable image is pure cost.
        cache.put(image_hash, "")
        if owns_cache:
            cache.save()
        return {**figure, "description": "", "description_cached": False,
                "description_error": "model reported the image as unreadable"}

    cache.put(image_hash, description)
    if owns_cache:
        cache.save()
    return {**figure, "description": description, "description_cached": False}


def _thread_client(local: threading.local, config: Config) -> LLMClient:
    """One LLMClient per worker thread, reused across that thread's calls.

    Not one shared client: an LLMClient owns a requests.Session, which is not safe to
    use from several threads at once. Not one client per call either — that would open a
    fresh connection pool for every figure and throw away TLS session reuse.
    """
    client = getattr(local, "client", None)
    if client is None:
        client = LLMClient(config=config)
        local.client = client
    return client


def describe_all(
    figures: Sequence[Dict[str, Any]],
    client: Optional[LLMClient] = None,
    config: Config = CFG,
) -> List[Dict[str, Any]]:
    """Describe every figure, sharing one cache so the file is written once.

    The calls go out concurrently: each one is a vision request that spends its time
    waiting on the network, so they overlap rather than queue. Results keep input order
    regardless of which finishes first — ingest assigns `position` from this list.
    """
    cache = DescriptionCache(config)
    workers = max(1, int(config.extraction.describe_workers))

    def describe(figure: Dict[str, Any]) -> Dict[str, Any]:
        return describe_figure(figure, client=client, cache=cache, config=config)

    if workers == 1 or len(figures) <= 1 or client is not None:
        # A caller-supplied client is used as given — this function cannot know whether
        # it is safe to share, and a test stub generally is not.
        described = [describe(figure) for figure in figures]
    else:
        local = threading.local()

        def describe_threaded(figure: Dict[str, Any]) -> Dict[str, Any]:
            return describe_figure(
                figure, client=_thread_client(local, config), cache=cache, config=config
            )

        with ThreadPoolExecutor(max_workers=min(workers, len(figures))) as pool:
            described = list(pool.map(describe_threaded, figures))

    cache.save()
    return described


def figure_chunk_text(caption: str, description: str) -> str:
    """The text that gets embedded for a figure chunk: caption AND description.

    Both, concatenated, never the description alone. The caption carries the authors'
    exact terminology, which a generated description tends to paraphrase away — and that
    terminology is what a query is most likely to match.
    """
    parts = [part.strip() for part in (caption, description) if part and part.strip()]
    return "\n".join(parts)
