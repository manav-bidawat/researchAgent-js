"""
Bridge ops for building the eval corpus: the eval topic spec, and search_literature run
standalone (fetch, extract and index in one call).

In:  nothing, or one topic's query settings from eval/topics.yaml.
Out: the parsed topic spec, or the tool's own structured result. Topics stay in eval/;
     these ops only carry them, so nothing topic-specific lands in src/ or bridge/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from bridge import runtime
from bridge.registry import Emit, op

ROOT = Path(__file__).resolve().parents[2]


@op("eval_topics")
def eval_topics(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """eval/topics.yaml, parsed. Read here because the YAML parser lives on this side."""
    import yaml

    path = ROOT / "eval" / "topics.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@op("search_literature")
def search_literature(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """LiteratureSearcher.search: collect, ingest and index one query in a single call."""
    from tools.search_literature import LiteratureSearcher

    searcher = LiteratureSearcher(runtime.config(), on_corpus_change=runtime.invalidate_index)
    return searcher.search(
        params.get("query", ""),
        max_results=params.get("max_results"),
        categories=params.get("categories"),
    )
