"""
Fetches papers from arXiv and records them: the fetch half of `search_literature`.

In:  a natural-language topic, a topic_tag, and caps (max_results, categories).
Out: {"papers_added", "papers_skipped", "papers_tagged", "topic_tag", "arxiv_query", ...}
     or an error dict. Never raises, and never indexes — chunking is M2's job.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

import arxiv

from corpus.arxiv_fetch import (
    FetchError,
    RateLimited,
    cooldown_detail,
    download_pdf,
    prepare_client,
    read_cooldown,
    write_cooldown,
)
from corpus.arxiv_query import apply_categories, plan_query
from corpus.chunk_store import ChunkStore
from config import CFG, Config
from llm_client import LLMClient
from corpus.manifest import Manifest, ManifestError
from common.records import RecordError, normalise_paper_id, paper_record

_SORT_CRITERIA = {
    "relevance": arxiv.SortCriterion.Relevance,
    "submitted": arxiv.SortCriterion.SubmittedDate,
    "updated": arxiv.SortCriterion.LastUpdatedDate,
}


def slugify_topic(topic: str) -> str:
    """'Graph Neural Networks!' -> 'graph_neural_networks'. Used as the default tag."""
    slug = re.sub(r"[^a-z0-9]+", "_", (topic or "").strip().lower()).strip("_")
    return slug[:60] or "untagged"


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


def _record_rate_limit(exc: RateLimited, config: Config) -> str:
    """Persist the cooldown and build the detail string. Both 429 paths go through here."""
    seconds = write_cooldown(
        config.paths.arxiv_cooldown,
        exc.retry_after,
        float(config.collection.rate_limit_cooldown_s),
    )
    return f"{exc} Not retrying for {seconds:.0f}s."


# One shared client for the whole process. Rebuilt only when the policy it was built
# with changes.
_CLIENT: Optional[arxiv.Client] = None
_CLIENT_POLICY: Optional[tuple] = None


def _build_client(config: Config) -> arxiv.Client:
    """arXiv client with the delay and retry policy from config, not library defaults.

    Deliberately shared rather than constructed per search. `arxiv.Client` enforces
    `delay_seconds` by comparing against `_last_request_dt`, which lives on the
    *instance* — so a fresh client per call resets that clock and two searches in quick
    succession reach arXiv with no gap at all. That is exactly the pattern the agent
    produces when it calls `search_literature` twice, and that a gate produces when it
    loops. Reusing the instance is what makes the configured delay real.

    The same instance also carries the session PDF downloads borrow, so the delay spans
    searches *and* downloads rather than searches alone.
    """
    global _CLIENT, _CLIENT_POLICY
    policy = (
        float(config.collection.request_delay_s),
        int(config.collection.max_retries),
        int(config.collection.page_size),
    )
    if _CLIENT is None or _CLIENT_POLICY != policy:
        delay, retries, page_size = policy
        _CLIENT = arxiv.Client(page_size=page_size, delay_seconds=delay, num_retries=retries)
        _CLIENT_POLICY = policy
    return prepare_client(_CLIENT, str(config.collection.user_agent))


def _to_paper_record(result: Any, pdf_path: str, topic_tag: str) -> Dict[str, Any]:
    """One arxiv.Result -> a paper record. Kept separate so it is testable with a stub."""
    published = getattr(result, "published", None)
    return paper_record(
        arxiv_id=result.get_short_id(),
        title=result.title or "",
        authors=[getattr(a, "name", str(a)) for a in (result.authors or [])],
        abstract=result.summary or "",
        published=published.isoformat() if published is not None else "",
        year=published.year if published is not None else 0,
        categories=list(result.categories or []),
        pdf_url=result.pdf_url or "",
        pdf_path=pdf_path,
        topic_tags=[topic_tag],
        parse_status="ok",
    )


def search_and_fetch(
    topic: str,
    topic_tag: Optional[str] = None,
    max_results: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    config: Config = CFG,
    llm: Optional[LLMClient] = None,
    download: bool = True,
) -> Dict[str, Any]:
    """Plan a query, search arXiv, dedup against the manifest, and download new PDFs.

    Papers already in the manifest are not re-downloaded; they gain `topic_tag` on
    both their paper record and every chunk record they already produced.
    """
    topic = (topic or "").strip()
    if not topic:
        return _error("empty_topic", "a non-empty topic is required")

    tag = (topic_tag or slugify_topic(topic)).strip()
    config.paths.ensure()

    # Checked before anything else costs money or a request: a live cooldown means every
    # arXiv query would answer 429, so planning one is wasted LLM spend too.
    remaining = read_cooldown(config.paths.arxiv_cooldown)
    if remaining > 0:
        return _error(
            "arxiv_rate_limited",
            cooldown_detail(remaining, config.paths.arxiv_cooldown),
            partial={"topic_tag": tag, "retry_after_s": round(remaining)},
        )

    try:
        manifest = Manifest.load(config)
    except ManifestError as exc:
        return _error("manifest_model_mismatch", str(exc))

    cap = int(config.collection.max_papers_per_topic)
    requested = int(max_results if max_results is not None else config.collection.max_results_default)
    limit = max(1, min(requested, cap))
    already = len(manifest.papers_for_topic(tag))
    if already >= cap:
        return _error(
            "topic_cap_reached",
            f"topic '{tag}' already holds {already} papers, at the cap of {cap}",
            partial={"topic_tag": tag, "papers_skipped": already},
        )

    plan = plan_query(topic, client=llm, config=config)
    # Only the caller's explicit categories are applied here. The planner's own
    # suggestions are already folded into plan["query"] before it is cached, so applying
    # them again would build "((q) AND cat) AND cat" on a cache miss and "(q) AND cat" on
    # a hit. Those are logically equivalent but arXiv scores the two strings differently
    # and returns a different ordering, which silently defeats dedup on the second run.
    query = apply_categories(plan["query"], categories)
    if not query:
        return _error("query_planning_failed", plan.get("error") or "no query could be built")

    sort_by = _SORT_CRITERIA.get(str(config.collection.sort_by).lower(), arxiv.SortCriterion.Relevance)
    search = arxiv.Search(query=query, max_results=limit, sort_by=sort_by)

    client = _build_client(config)
    try:
        results = list(client.results(search))
    except RateLimited as exc:
        # Caught ahead of ArxivError on purpose: 429 is not "arXiv is down", it is
        # "this IP is blocked for minutes, on every query". Retrying is what deepens it.
        return _error(
            "arxiv_rate_limited",
            _record_rate_limit(exc, config),
            partial={"arxiv_query": query, "topic_tag": tag, "retry_after": exc.retry_after},
        )
    except (arxiv.ArxivError, arxiv.HTTPError, arxiv.UnexpectedEmptyPageError) as exc:
        return _error("arxiv_unavailable", f"arXiv search failed: {exc}", partial={"arxiv_query": query})
    except Exception as exc:  # network stack, DNS, TLS — anything urllib raises underneath
        return _error("arxiv_unavailable", f"arXiv search failed: {exc}", partial={"arxiv_query": query})

    if not results:
        return _error(
            "no_results",
            f"arXiv returned nothing for {query!r}",
            partial={"arxiv_query": query, "topic_tag": tag},
        )

    store = ChunkStore(config=config)
    added: List[Dict[str, Any]] = []
    tagged: List[str] = []
    skipped: List[str] = []
    failures: List[Dict[str, str]] = []
    touched: List[str] = []
    rate_limited: Optional[RateLimited] = None

    for result in results:
        paper_id = normalise_paper_id(result.get_short_id())

        if manifest.has_paper(paper_id):
            touched.append(paper_id)
            skipped.append(paper_id)
            # Dedup hit: the paper stays, but the new tag must reach the paper record
            # AND its chunk records, or topic_filter silently misses it.
            if manifest.tag_paper(paper_id, tag):
                store.add_topic_tag_to_paper(paper_id, tag)
                tagged.append(paper_id)
            continue

        pdf_path = config.paths.paper_pdf(paper_id)
        if download and not pdf_path.is_file():
            try:
                download_pdf(
                    client,
                    getattr(result, "pdf_url", "") or "",
                    pdf_path,
                    float(config.collection.download_timeout_s),
                )
            except RateLimited as exc:
                # Stop the whole loop. Every remaining result would be one more request
                # into a wall that answers 429 for all of them, and each one extends the
                # cooling-off. Papers already added below are kept and saved.
                rate_limited = exc
                break
            except Exception as exc:  # FetchError, OSError, and anything requests raises
                failures.append({"paper_id": paper_id, "detail": f"pdf download failed: {exc}"})
                continue

        try:
            record = _to_paper_record(result, str(pdf_path), tag)
        except RecordError as exc:
            failures.append({"paper_id": paper_id, "detail": f"bad record: {exc}"})
            continue

        manifest.add_paper(record)
        # Only now is the paper really part of the topic. Recording it before the
        # download and record build could succeed left phantom ids in topics.paper_ids,
        # which then counted against max_papers_per_topic.
        touched.append(paper_id)
        added.append(
            {
                "arxiv_id": record["arxiv_id"],
                "paper_id": record["paper_id"],
                "title": record["title"],
                "year": record["year"],
                "abstract_snippet": record["abstract"][:200],
            }
        )

    # Runs even after a rate-limit break: the papers that did land are already on disk
    # and in the manifest, and dropping them would make a 429 halfway through a total
    # loss that the next run has to redo — more arXiv traffic, not less.
    manifest.record_topic(tag, topic, query, touched)
    manifest.save()

    if rate_limited is not None:
        return _error(
            "arxiv_rate_limited",
            _record_rate_limit(rate_limited, config),
            partial={
                "arxiv_query": query,
                "topic_tag": tag,
                "retry_after": rate_limited.retry_after,
                "papers_added": added,
                "papers_skipped": len(skipped),
                "papers_tagged": len(tagged),
                "failures": failures,
            },
        )

    return {
        "papers_added": added,
        "papers_skipped": len(skipped),
        "papers_tagged": len(tagged),
        "topic_tag": tag,
        "arxiv_query": query,
        "query_was_cached": plan.get("cached", False),
        "query_was_fallback": plan.get("fallback", False),
        "total_indexed": len(manifest.papers),
        "failures": failures,
    }
