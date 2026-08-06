"""
Turns a natural-language topic into an arXiv query string via one LLM call.

In:  a topic string, plus an optional LLMClient (the caller's, so failures surface there).
Out: {"query", "categories", "reasoning", "cached", "fallback"} — never raises. On an
     LLM failure it degrades to a keyword query and says so via `fallback`.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from config import CFG, Config
from llm_client import LLMClient, LLMError
from common.records import topic_hash
from common.storage import read_json, write_json_atomic

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
# Words that add nothing to a search and cost an AND term if kept.
_STOPWORDS = frozenset(
    "a an the of for on in to and or how do does what why papers paper research "
    "study studies about into using use with best recent new".split()
)


def fallback_query(topic: str) -> str:
    """A keyword query built without an LLM, for when the planning call fails.

    Deliberately broad: over-retrieving is recoverable by reranking, whereas a query
    that returns nothing ends the search.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9+\-]+", topic.lower()) if w not in _STOPWORDS]
    terms = words[:6] or [topic.strip() or "science"]
    return "abs:(" + " OR ".join(f'"{term}"' if " " in term else term for term in terms) + ")"


def _parse_plan(text: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON object out of a model reply, tolerating code fences and prose."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
        return None
    if not payload["query"].strip():
        return None
    categories = payload.get("categories")
    return {
        "query": payload["query"].strip(),
        "categories": [str(c) for c in categories] if isinstance(categories, list) else [],
        "reasoning": str(payload.get("reasoning") or ""),
    }


def apply_categories(query: str, categories: Optional[Sequence[str]]) -> str:
    """AND a category filter onto a planned query. No categories leaves it unchanged."""
    tags = [str(c).strip() for c in (categories or []) if str(c).strip()]
    if not tags:
        return query
    clause = " OR ".join(f"cat:{tag}" for tag in tags)
    return f"({query}) AND ({clause})" if query else f"({clause})"


def plan_query(
    topic: str,
    client: Optional[LLMClient] = None,
    config: Config = CFG,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Plan an arXiv query for `topic`, caching the result by sha256(topic).

    The cache is keyed on the topic alone, per docs/DATA_SCHEMA.md section 5. Explicit
    `categories` from the caller are applied at search time rather than baked in here,
    so a cached query stays valid whatever categories a later call passes.
    """
    topic = (topic or "").strip()
    if not topic:
        return {
            "query": "",
            "categories": [],
            "reasoning": "",
            "cached": False,
            "fallback": True,
            "error": "empty_topic",
        }

    cache_path = config.paths.arxiv_queries
    cache: Dict[str, str] = read_json(cache_path, default={}) or {}
    key = topic_hash(topic)

    if use_cache and isinstance(cache.get(key), str) and cache[key].strip():
        # The cached string is the COMPLETE query, category clause included. Caching the
        # bare query and dropping the planner's categories made a cache hit search a
        # broader query than the miss did, so the "same topic returns the same papers"
        # guarantee that dedup relies on quietly failed.
        return {
            "query": cache[key],
            "categories": [],
            "reasoning": "",
            "cached": True,
            "fallback": False,
        }

    llm = client if client is not None else LLMClient(config=config)
    messages = [
        {"role": "system", "content": config.prompt("arxiv_query")},
        {"role": "user", "content": topic},
    ]

    # Retry an unparseable reply rather than falling back immediately. A fallback query
    # is not cached, so a topic that fell back once and planned successfully the next
    # time would be searched with two different queries — and dedup, which assumes a
    # topic maps to a stable query, would silently stop matching.
    attempts = max(1, int(config.llm.planning_retries))
    budget = float(config.llm.planning_budget_s)
    started = time.monotonic()
    plan, note = None, ""
    for attempt in range(attempts):
        if time.monotonic() - started > budget:
            note = f"planning budget of {budget:.0f}s exhausted after {attempt} attempts"
            break
        try:
            # max_retries=1 because this loop is already the retry: letting the client
            # retry too would multiply the two budgets together.
            response = llm.complete(
                messages,
                role="utility",
                timeout=float(config.llm.planning_timeout_s),
                max_retries=1,
            )
        except LLMError as exc:
            note = f"{exc.code}: {exc.detail}"
            continue
        plan = _parse_plan(response.text)
        if plan is not None:
            break
        note = (
            f"reply {attempt + 1}/{attempts} had no usable JSON object "
            f"(model={response.model!r}, text={' '.join((response.text or '').split())[:80]!r})"
        )

    if plan is None:
        return {
            "query": fallback_query(topic),
            "categories": [],
            "reasoning": "planning call failed; using a keyword query",
            "cached": False,
            "fallback": True,
            "error": note,
        }

    # Fold the planner's own category suggestion into the query before caching, so the
    # cached string is self-contained and a hit reproduces the miss exactly.
    plan["query"] = apply_categories(plan["query"], plan["categories"])
    cache[key] = plan["query"]
    write_json_atomic(cache_path, cache)
    return {**plan, "cached": False, "fallback": False}
