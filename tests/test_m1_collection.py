"""
Offline tests for M1: records, manifest, chunk backfill, query planning, collection.

In:  a tmp_path-scoped config, a stub arxiv.Client, and a fake LLM — no network.
Out: assertions on record shapes, dedup behaviour, tag backfill, and error dicts.
"""

import time
from datetime import datetime, timezone

import arxiv
import pytest
import requests

from corpus import arxiv_fetch
from corpus import arxiv_query
from corpus import collect
from common import records
from corpus.chunk_store import ChunkStore
from config import load_config
from corpus.manifest import Manifest, ManifestError
from common.records import RecordError, chunk_record, paper_record


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """A Config whose paths all live under tmp_path, so tests touch no real data."""
    import config as config_module

    repo_root = config_module.REPO_ROOT  # capture before patching it away
    source = (repo_root / "config.yaml").read_text(encoding="utf-8")
    local = tmp_path / "config.yaml"
    local.write_text(source, encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    # Prompts are not copied into tmp_path; read them from the real tree.
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


class StubResult:
    """Mimics the surface of arxiv.Result that collect.py actually touches."""

    def __init__(self, short_id, title="A Paper", year=2021, categories=("cs.LG",)):
        self._short_id = short_id
        self.title = title
        self.authors = [type("A", (), {"name": "Ada Lovelace"})()]
        self.summary = "We show a thing about a method."
        self.published = datetime(year, 3, 25, tzinfo=timezone.utc)
        self.categories = list(categories)
        self.pdf_url = f"https://arxiv.org/pdf/{short_id}"
        self.downloads = 0
        self.download_error = None   # set to an exception to make this PDF fail

    def get_short_id(self):
        return self._short_id


def stub_arxiv(monkeypatch, results, fail_with=None):
    """Replace collect's arxiv client and PDF fetch so no request leaves the machine.

    Downloads are stubbed at `collect.download_pdf` rather than on the result, because
    that is where the real fetch now lives: `arxiv.Result.download_pdf` bypasses the
    shared rate-limit clock and is deliberately no longer called.
    """
    by_url = {r.pdf_url: r for r in results}

    class StubClient:
        delay_seconds = 4.0
        _last_request_dt = None

        def results(self, search):
            if fail_with is not None:
                raise fail_with
            return list(results)

    def fake_download(client, pdf_url, dest, timeout_s):
        result = by_url[pdf_url]
        result.downloads += 1
        if result.download_error is not None:
            raise result.download_error
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4 stub")
        return dest

    monkeypatch.setattr(collect, "_build_client", lambda config: StubClient())
    monkeypatch.setattr(collect, "download_pdf", fake_download)


@pytest.fixture()
def planned(monkeypatch):
    """Skip the LLM: planning always returns a fixed query."""
    monkeypatch.setattr(
        arxiv_query,
        "plan_query",
        lambda *a, **k: {"query": "abs:(x)", "categories": [], "reasoning": "", "cached": False, "fallback": False},
    )
    monkeypatch.setattr(
        collect,
        "plan_query",
        lambda *a, **k: {"query": "abs:(x)", "categories": [], "reasoning": "", "cached": False, "fallback": False},
    )


# ----------------------------------------------------------------------------- records

def test_paper_record_matches_the_schema():
    record = paper_record(
        arxiv_id="2103.14030v2", title="Swin   Transformer", authors=["A"], abstract="a  b",
        published="2021-03-25", year=2021, categories=["cs.CV"], pdf_url="u", pdf_path="p",
        topic_tags=["vision"],
    )
    assert record["paper_id"] == "2103_14030v2"
    assert record["arxiv_id"] == "2103.14030v2"
    assert record["title"] == "Swin Transformer", "whitespace should be collapsed"
    assert record["topic_tags"] == ["vision"]
    assert set(record) == {
        "paper_id", "arxiv_id", "title", "authors", "abstract", "year", "published",
        "categories", "pdf_url", "pdf_path", "topic_tags", "n_chunks", "n_figures",
        "indexed_at", "parse_status", "parse_note",
    }


def test_old_style_arxiv_id_is_filename_safe():
    assert records.normalise_paper_id("math/0309136v1") == "math_0309136v1"


@pytest.mark.parametrize("bad", ["vision", 42, None, ["ok", ""], ["ok", 3]])
def test_bare_string_and_junk_topic_tags_are_rejected(bad):
    with pytest.raises(RecordError):
        records.validate_topic_tags(bad)


def test_duplicate_tags_are_collapsed():
    assert records.validate_topic_tags(["a", "b", "a"]) == ["a", "b"]


def test_chunk_record_requires_a_figure_id_for_figures():
    with pytest.raises(RecordError, match="requires a figure_id"):
        chunk_record(paper_id="p", paper_title="t", topic_tags=["x"], chunk_type="figure",
                     text="c", page=1, position=0, n_tokens=1)


def test_content_hash_is_text_only():
    a = chunk_record(paper_id="p1", paper_title="t", topic_tags=["x"], chunk_type="text",
                     text="same words", page=1, position=0, n_tokens=2)
    b = chunk_record(paper_id="p2", paper_title="u", topic_tags=["y"], chunk_type="text",
                     text="same words", page=9, position=5, n_tokens=2)
    assert a["content_hash"] == b["content_hash"], "identical text hashes identically"
    assert a["chunk_id"] != b["chunk_id"], "but identity is per-paper, so both records live"


# ---------------------------------------------------------------------------- manifest

def test_manifest_round_trips(cfg):
    manifest = Manifest.load(cfg)
    manifest.add_paper(paper_record(
        arxiv_id="1.1v1", title="T", authors=[], abstract="", published="2020-01-01", year=2020,
        categories=[], pdf_url="", pdf_path="", topic_tags=["alpha"]))
    manifest.record_topic("alpha", "a topic", "abs:(x)", ["1_1v1"])
    manifest.save()

    reloaded = Manifest.load(cfg)
    assert reloaded.has_paper("1_1v1")
    assert reloaded.papers_for_topic("alpha") == ["1_1v1"]
    assert reloaded.data["manifest_hash"], "hash must be written for the cluster cache"


def test_embedding_model_mismatch_refuses_to_load(cfg):
    manifest = Manifest.load(cfg)
    manifest.add_paper(paper_record(
        arxiv_id="1.1v1", title="T", authors=[], abstract="", published="2020-01-01", year=2020,
        categories=[], pdf_url="", pdf_path="", topic_tags=["alpha"]))
    manifest.data["embedding_model"] = "some/other-model"
    manifest.save()

    with pytest.raises(ManifestError, match="must be rebuilt"):
        Manifest.load(cfg)


def test_manifest_hash_tracks_papers(cfg):
    manifest = Manifest.load(cfg)
    before = manifest.compute_hash()
    manifest.add_paper(paper_record(
        arxiv_id="9.9v1", title="T", authors=[], abstract="", published="2020-01-01", year=2020,
        categories=[], pdf_url="", pdf_path="", topic_tags=["z"]))
    assert manifest.compute_hash() != before


# ------------------------------------------------------------------------- chunk store

def test_backfill_touches_only_the_named_paper(cfg):
    store = ChunkStore(config=cfg)
    store.append([
        chunk_record(paper_id="p1", paper_title="T", topic_tags=["a"], chunk_type="text",
                     text=f"body {i}", page=1, position=i, n_tokens=3) for i in range(3)
    ])
    store.append([chunk_record(paper_id="p2", paper_title="U", topic_tags=["b"], chunk_type="text",
                               text="other", page=1, position=0, n_tokens=3)])

    assert store.add_topic_tag_to_paper("p1", "b") == 3
    assert all(c["topic_tags"] == ["a", "b"] for c in store.for_paper("p1"))
    assert store.for_paper("p2")[0]["topic_tags"] == ["b"], "other papers untouched"
    assert store.add_topic_tag_to_paper("p1", "b") == 0, "backfill is idempotent"


def test_malformed_line_is_skipped_not_raised(cfg):
    store = ChunkStore(config=cfg)
    store.append([chunk_record(paper_id="p1", paper_title="T", topic_tags=["a"], chunk_type="text",
                               text="good", page=1, position=0, n_tokens=1)])
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write('{"truncated": \n')
    assert store.count() == 1


# ------------------------------------------------------------------- query planning

def test_fallback_query_drops_stopwords():
    query = arxiv_query.fallback_query("How do the papers use graph neural networks?")
    assert "graph" in query and "networks" in query
    assert "the" not in query.split() and query.startswith("abs:(")


def test_planning_retries_past_an_unusable_reply(cfg):
    """openrouter/free can route to a moderation classifier that answers everything with
    'User Safety: safe'. A retry re-rolls the model, so one bad reply must not force the
    fallback — an uncached fallback would make the same topic search two different
    queries across calls, and dedup assumes a topic maps to a stable query."""
    replies = [
        "User Safety: safe",
        "User Safety: safe",
        '{"query": "abs:(moe)", "categories": [], "reasoning": "r"}',
    ]

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def complete(self, *a, **k):
            text = replies[self.calls]
            self.calls += 1
            return type("R", (), {"text": text, "model": "stub"})()

    client = FlakyClient()
    plan = arxiv_query.plan_query("mixture of experts", client=client, config=cfg)

    assert plan["fallback"] is False, "should have retried instead of falling back"
    assert plan["query"] == "abs:(moe)"
    assert client.calls == 3


def test_planning_gives_up_after_the_retry_budget(cfg):
    class AlwaysBad:
        def __init__(self):
            self.calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            return type("R", (), {"text": "User Safety: safe", "model": "stub"})()

    client = AlwaysBad()
    plan = arxiv_query.plan_query("mixture of experts", client=client, config=cfg)

    assert client.calls == int(cfg.llm.planning_retries)
    assert plan["fallback"] is True
    assert "User Safety" in plan["error"], "the unusable reply should be reported"


def test_planning_stops_at_its_wall_clock_budget(cfg, monkeypatch):
    """Two retry layers must not multiply. Planning retries wrap the client's own
    transport retries, so without a wall-clock ceiling one topic could stall for
    planning_retries x max_retries x timeout_s seconds."""
    clock = {"t": 0.0}
    monkeypatch.setattr(arxiv_query.time, "monotonic", lambda: clock["t"])

    class SlowClient:
        def __init__(self):
            self.calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            clock["t"] += 40.0  # each attempt burns 40s of the budget
            return type("R", (), {"text": "User Safety: safe", "model": "stub"})()

    client = SlowClient()
    plan = arxiv_query.plan_query("a topic", client=client, config=cfg)

    assert client.calls == 2, "budget of 75s allows two 40s attempts, not three"
    assert plan["fallback"] is True
    assert "budget" in plan["error"]


def test_planning_passes_a_short_timeout_and_no_transport_retry(cfg):
    """The client must not retry underneath a caller that is already retrying."""
    seen = {}

    class RecordingClient:
        def complete(self, messages, **kwargs):
            seen.update(kwargs)
            return type("R", (), {"text": '{"query": "abs:(x)", "categories": []}', "model": "s"})()

    arxiv_query.plan_query("a topic", client=RecordingClient(), config=cfg)
    assert seen["timeout"] == cfg.llm.planning_timeout_s
    assert seen["max_retries"] == 1


def test_plan_query_falls_back_when_the_llm_fails(cfg, monkeypatch):
    from llm_client import LLMError

    class DeadClient:
        def complete(self, *a, **k):
            raise LLMError("llm_unreachable", "no network")

    plan = plan = arxiv_query.plan_query("robot grasping", client=DeadClient(), config=cfg)
    assert plan["fallback"] is True
    assert plan["query"].startswith("abs:(")
    assert "llm_unreachable" in plan["error"]


def test_cached_query_reproduces_the_uncached_one_exactly(cfg):
    """A cache hit must search the same query as the miss, category clause included.

    Caching the bare query and dropping the planner's categories made the second
    search on a topic broader than the first, so it returned different papers and
    dedup never fired.
    """
    class Planner:
        def complete(self, *a, **k):
            return type("R", (), {"text": '{"query": "abs:(moe)", "categories": ["cs.LG"], "reasoning": ""}'})()

    miss = arxiv_query.plan_query("routing", client=Planner(), config=cfg)
    hit = arxiv_query.plan_query("routing", client=Planner(), config=cfg)

    assert miss["cached"] is False and hit["cached"] is True
    assert hit["query"] == miss["query"]
    assert "cat:cs.LG" in hit["query"], "the planner's categories must survive caching"


def test_collect_builds_an_identical_query_on_miss_and_hit(cfg, monkeypatch):
    """The query string must be byte-identical whether planning was cached or not.

    arXiv is deterministic for a given query string, so any difference between the two
    paths changes the result ordering and dedup stops matching on the second run.
    """
    class Planner:
        def complete(self, *a, **k):
            return type("R", (), {
                "text": '{"query": "abs:(moe)", "categories": ["cs.LG"], "reasoning": ""}',
                "model": "stub",
            })()

    monkeypatch.setattr(collect, "LLMClient", lambda config=None: Planner())
    seen = []

    class RecordingClient:
        def results(self, search):
            seen.append(search.query)
            return []

    monkeypatch.setattr(collect, "_build_client", lambda config: RecordingClient())

    collect.search_and_fetch("a topic", topic_tag="t", config=cfg, llm=Planner())
    collect.search_and_fetch("a topic", topic_tag="t", config=cfg, llm=Planner())

    assert len(seen) == 2
    assert seen[0] == seen[1], f"cache miss searched {seen[0]!r} but the hit searched {seen[1]!r}"
    assert seen[0].count("cat:cs.LG") == 1, "the category clause must appear exactly once"


def test_plan_query_caches_by_topic_hash(cfg):
    calls = []

    class OnceClient:
        def complete(self, *a, **k):
            calls.append(1)
            return type("R", (), {"text": '{"query": "abs:(planned)", "categories": [], "reasoning": "r"}'})()

    first = arxiv_query.plan_query("some topic", client=OnceClient(), config=cfg)
    second = arxiv_query.plan_query("some topic", client=OnceClient(), config=cfg)
    assert first["query"] == second["query"] == "abs:(planned)"
    assert len(calls) == 1, "second call must hit the cache"
    assert second["cached"] is True


# ------------------------------------------------------------------------- collection

def test_fetch_then_refetch_downloads_once(cfg, monkeypatch, planned):
    results = [StubResult("2101.00001v1"), StubResult("2101.00002v1"), StubResult("2101.00003v1")]
    stub_arxiv(monkeypatch, results)

    first = collect.search_and_fetch("a topic", config=cfg, max_results=3)
    assert len(first["papers_added"]) == 3
    assert first["papers_skipped"] == 0
    assert sum(r.downloads for r in results) == 3

    second = collect.search_and_fetch("a topic", config=cfg, max_results=3)
    assert second["papers_added"] == []
    assert second["papers_skipped"] == 3
    assert sum(r.downloads for r in results) == 3, "no paper is downloaded twice"


def test_second_topic_backfills_tags_onto_chunks(cfg, monkeypatch, planned):
    shared = StubResult("2101.00001v1")
    stub_arxiv(monkeypatch, [shared])
    collect.search_and_fetch("first topic", topic_tag="alpha", config=cfg)

    # Pretend M2 has already chunked it.
    store = ChunkStore(config=cfg)
    store.append([
        chunk_record(paper_id="2101_00001v1", paper_title="A Paper", topic_tags=["alpha"],
                     chunk_type="text", text=f"t{i}", page=1, position=i, n_tokens=2)
        for i in range(4)
    ])

    stub_arxiv(monkeypatch, [shared])
    result = collect.search_and_fetch("second topic", topic_tag="beta", config=cfg)

    assert result["papers_skipped"] == 1
    assert result["papers_tagged"] == 1
    manifest = Manifest.load(cfg)
    assert manifest.get_paper("2101_00001v1")["topic_tags"] == ["alpha", "beta"]
    assert all(c["topic_tags"] == ["alpha", "beta"] for c in store.for_paper("2101_00001v1")), \
        "chunk records must be backfilled, not just the manifest"


def test_arxiv_failure_returns_an_error_dict(cfg, monkeypatch, planned):
    stub_arxiv(monkeypatch, [], fail_with=ConnectionError("network is unreachable"))
    result = collect.search_and_fetch("a topic", config=cfg)
    assert result["error"] == "arxiv_unavailable"
    assert "unreachable" in result["detail"]
    assert "papers_added" not in result


def test_zero_results_is_an_error_dict_not_an_exception(cfg, monkeypatch, planned):
    stub_arxiv(monkeypatch, [])
    result = collect.search_and_fetch("a topic", config=cfg)
    assert result["error"] == "no_results"


def test_empty_topic_is_rejected(cfg):
    assert collect.search_and_fetch("   ", config=cfg)["error"] == "empty_topic"


def test_download_failure_is_partial_not_fatal(cfg, monkeypatch, planned):
    good, bad = StubResult("2101.00001v1"), StubResult("2101.00002v1")
    bad.download_error = OSError("disk full")
    stub_arxiv(monkeypatch, [good, bad])

    result = collect.search_and_fetch("a topic", config=cfg)
    assert len(result["papers_added"]) == 1, "the good paper still lands"
    assert len(result["failures"]) == 1
    assert "disk full" in result["failures"][0]["detail"]


def test_failed_download_leaves_no_phantom_id_in_the_topic(cfg, monkeypatch, planned):
    """topics.paper_ids must list only papers that actually made it into the manifest.

    A phantom id counts against max_papers_per_topic and makes papers_for_topic
    return an id that resolves to nothing.
    """
    good, bad = StubResult("2101.00001v1"), StubResult("2101.00002v1")
    bad.download_error = OSError("disk full")
    stub_arxiv(monkeypatch, [good, bad])

    collect.search_and_fetch("a topic", topic_tag="alpha", config=cfg)
    manifest = Manifest.load(cfg)
    recorded = manifest.papers_for_topic("alpha")

    assert recorded == ["2101_00001v1"]
    assert all(manifest.get_paper(pid) is not None for pid in recorded), "no phantom ids"


def test_same_papers_under_a_new_tag_are_tagged_not_re_added(cfg, monkeypatch, planned):
    """The deterministic backfill case: identical results, different topic_tag."""
    results = [StubResult("2101.00001v1"), StubResult("2101.00002v1")]
    stub_arxiv(monkeypatch, results)
    collect.search_and_fetch("a topic", topic_tag="alpha", config=cfg)

    store = ChunkStore(config=cfg)
    store.append([
        chunk_record(paper_id="2101_00001v1", paper_title="A Paper", topic_tags=["alpha"],
                     chunk_type="text", text=f"t{i}", page=1, position=i, n_tokens=2)
        for i in range(3)
    ])

    stub_arxiv(monkeypatch, results)
    second = collect.search_and_fetch("a topic", topic_tag="beta", config=cfg)

    assert second["papers_added"] == []
    assert second["papers_skipped"] == 2
    assert second["papers_tagged"] == 2
    assert sum(r.downloads for r in results) == 2, "nothing re-downloaded"
    assert all(c["topic_tags"] == ["alpha", "beta"] for c in store.for_paper("2101_00001v1"))


def test_topic_cap_is_enforced(cfg, monkeypatch, planned):
    cap = int(cfg.collection.max_papers_per_topic)
    stub_arxiv(monkeypatch, [StubResult(f"2101.{i:05d}v1") for i in range(cap)])
    collect.search_and_fetch("a topic", topic_tag="alpha", config=cfg, max_results=cap)

    stub_arxiv(monkeypatch, [StubResult("2102.00001v1")])
    result = collect.search_and_fetch("a topic", topic_tag="alpha", config=cfg)
    assert result["error"] == "topic_cap_reached"


# --- arXiv request pacing -------------------------------------------------------


class _PacingResponse:
    """A response that always fails, so the client retries and we can time the gaps."""

    status_code = 503
    content = b""


def _timed_client(monkeypatch):
    """The shared client, with its HTTP layer replaced by a timestamp recorder.

    Stubbing `_session.get` rather than `_parse_feed` or `__try_parse_feed` is the whole
    point: the delay is enforced *inside* `__try_parse_feed`, so replacing either of
    those bypasses the behaviour under test and the assertion passes vacuously.
    """
    import arxiv

    from config import CFG
    from corpus import collect

    collect._CLIENT = None  # ignore any client a previous test built
    collect._CLIENT_POLICY = None
    client = collect._build_client(CFG)
    stamps: list = []

    def record(url, headers=None, **kwargs):
        stamps.append(time.monotonic())
        return _PacingResponse()

    # `hooks` and `headers` are part of the requests.Session surface prepare_client
    # touches, so the stub has to carry them or _build_client fails on the next call.
    stub_session = type(
        "S", (), {"get": staticmethod(record), "hooks": {"response": []}, "headers": {}}
    )()
    monkeypatch.setattr(client, "_session", stub_session)
    return collect, client, stamps


def test_arxiv_requests_are_spaced_by_the_configured_delay(monkeypatch):
    """arXiv asks for a gap between requests; the client must actually leave one."""
    import arxiv

    from config import CFG

    collect, client, stamps = _timed_client(monkeypatch)
    client.num_retries = 2  # one initial request plus two retries

    with pytest.raises(Exception):
        list(client.results(arxiv.Search(query="pacing", max_results=1)))

    gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
    assert len(stamps) == 3, f"expected 3 requests, got {len(stamps)}"
    delay = float(CFG.collection.request_delay_s)
    assert delay > 3, "arXiv asks for 3s; the configured delay must exceed it, not sit on it"
    assert all(gap >= delay - 0.2 for gap in gaps), f"gaps too tight: {gaps}"


def test_the_arxiv_client_is_shared_so_the_delay_spans_separate_searches(monkeypatch):
    """A fresh client per search would reset the delay clock and defeat the pacing.

    `arxiv.Client` tracks `_last_request_dt` per instance, so two searches in quick
    succession — the agent calling search_literature twice — would reach arXiv with no
    gap at all if each built its own client.
    """
    import arxiv

    from config import CFG

    collect, client, stamps = _timed_client(monkeypatch)
    assert collect._build_client(CFG) is client, "client must be reused across calls"

    client.num_retries = 0
    for _ in range(2):  # two separate 'search_and_fetch' calls
        with pytest.raises(Exception):
            list(collect._build_client(CFG).results(arxiv.Search(query="q", max_results=1)))

    gap = stamps[1] - stamps[0]
    assert gap >= float(CFG.collection.request_delay_s) - 0.2, f"no gap between searches: {gap}"


def test_arxiv_endpoints_are_https():
    """Queries and PDF fetches must not travel in the clear."""
    import arxiv

    assert arxiv.Client.query_url_format.startswith("https://")


# ------------------------------------------------------------------- rate limiting

def test_rate_limited_download_stops_the_loop_after_one_attempt(cfg, monkeypatch, planned):
    """A 429 is per-IP and applies to every arXiv request, so the burst must stop.

    The old code recorded the 429 as a per-paper failure and continued, turning one
    blocked request into N more that deepen the cooling-off.
    """
    results = [StubResult(f"2101.{i:05d}v1") for i in range(5)]
    for r in results[1:]:
        r.download_error = arxiv_fetch.RateLimited(r.pdf_url, retry_after="120")
    stub_arxiv(monkeypatch, results)

    result = collect.search_and_fetch("a topic", topic_tag="alpha", config=cfg, max_results=5)

    assert result["error"] == "arxiv_rate_limited"
    assert result["partial"]["retry_after"] == "120"
    attempted = sum(r.downloads for r in results)
    assert attempted == 2, f"stopped at the first 429, not {attempted} requests into the wall"


def test_rate_limit_keeps_the_papers_that_did_land(cfg, monkeypatch, planned):
    """Papers fetched before the 429 stay in the manifest; a 429 is not a rollback."""
    results = [StubResult(f"2101.{i:05d}v1") for i in range(4)]
    results[2].download_error = arxiv_fetch.RateLimited(results[2].pdf_url)
    stub_arxiv(monkeypatch, results)

    result = collect.search_and_fetch("a topic", topic_tag="alpha", config=cfg, max_results=4)

    assert result["error"] == "arxiv_rate_limited"
    assert len(result["partial"]["papers_added"]) == 2
    manifest = Manifest.load(cfg)
    assert manifest.papers_for_topic("alpha") == ["2101_00000v1", "2101_00001v1"]


def test_rate_limited_search_is_not_reported_as_arxiv_unavailable(cfg, monkeypatch, planned):
    """The search path distinguishes 'blocked for minutes' from 'arXiv is down'."""
    stub_arxiv(monkeypatch, [], fail_with=arxiv_fetch.RateLimited("https://export.arxiv.org/api/query"))
    result = collect.search_and_fetch("a topic", config=cfg)
    assert result["error"] == "arxiv_rate_limited"


def test_rate_limited_is_not_retried_by_the_arxiv_client():
    """`RateLimited` must fall outside the exception set `_parse_feed` retries on.

    If it were an `arxiv.HTTPError`, the library would send `num_retries` more requests
    at a moment when arXiv is answering 429 to everything.
    """
    retried = (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError, requests.exceptions.ConnectionError)
    assert not issubclass(arxiv_fetch.RateLimited, retried)


def test_downloads_wait_for_the_shared_rate_limit_slot(monkeypatch, tmp_path):
    """Downloads read and write the same clock the library's feed path uses.

    Without this, a search and every PDF after it leave together: the delay is enforced
    only between feed requests, and `urlretrieve` touches none of that bookkeeping.
    """
    slept = []
    monkeypatch.setattr(arxiv_fetch.time, "sleep", lambda s: slept.append(s))

    class Recording:
        delay_seconds = 4.0
        _last_request_dt = None

        def __init__(self):
            self._session = self

        def get(self, url, stream=False, timeout=None):
            return _PdfResponse()

    client = Recording()
    arxiv_fetch.download_pdf(client, "https://arxiv.org/pdf/x", tmp_path / "a.pdf", 60.0)
    assert slept == [], "nothing to wait for on the first request"
    assert client._last_request_dt is not None, "the clock must be stamped for the next caller"

    arxiv_fetch.download_pdf(client, "https://arxiv.org/pdf/y", tmp_path / "b.pdf", 60.0)
    assert len(slept) == 1 and slept[0] > 3.0, f"expected a ~4s wait, got {slept}"


class _PdfResponse:
    status_code = 200
    headers: dict = {}

    def iter_content(self, chunk_size=0):
        yield b"%PDF-1.4 stub body"

    def close(self):
        pass


def test_a_truncated_download_never_becomes_a_cache_hit(monkeypatch, tmp_path):
    """A body that dies mid-write must leave no file at the final path.

    `urlretrieve` wrote straight to the destination, so a partial PDF looked like a
    completed download forever after — `not pdf_path.is_file()` is the only re-fetch gate.
    """
    monkeypatch.setattr(arxiv_fetch.time, "sleep", lambda s: None)

    class Dying:
        status_code = 200
        headers: dict = {}

        def iter_content(self, chunk_size=0):
            yield b"%PDF-1.4 part"
            raise requests.exceptions.ChunkedEncodingError("connection reset")

        def close(self):
            pass

    class Client:
        delay_seconds = 0.0
        _last_request_dt = None

        def __init__(self):
            self._session = self

        def get(self, url, stream=False, timeout=None):
            return Dying()

    dest = tmp_path / "a.pdf"
    with pytest.raises(arxiv_fetch.FetchError):
        arxiv_fetch.download_pdf(Client(), "https://arxiv.org/pdf/x", dest, 60.0)
    assert not dest.exists()
    assert list(tmp_path.glob("*.part")) == [], "the partial file is cleaned up too"


def test_an_html_error_page_with_status_200_is_not_accepted_as_a_pdf(monkeypatch, tmp_path):
    """A size check passes on an error page; the magic bytes are what catch it."""
    monkeypatch.setattr(arxiv_fetch.time, "sleep", lambda s: None)

    class Html:
        status_code = 200
        headers: dict = {}

        def iter_content(self, chunk_size=0):
            yield b"<!DOCTYPE html><html><body>Too many requests</body></html>"

        def close(self):
            pass

    class Client:
        delay_seconds = 0.0
        _last_request_dt = None

        def __init__(self):
            self._session = self

        def get(self, url, stream=False, timeout=None):
            return Html()

    dest = tmp_path / "a.pdf"
    with pytest.raises(arxiv_fetch.FetchError):
        arxiv_fetch.download_pdf(Client(), "https://arxiv.org/pdf/x", dest, 60.0)
    assert not dest.exists()


def test_the_429_hook_is_installed_once_however_often_the_client_is_prepared():
    """`_build_client` calls prepare_client on every search; hooks must not stack."""
    client = arxiv.Client(page_size=1, delay_seconds=0, num_retries=0)
    for _ in range(3):
        arxiv_fetch.prepare_client(client, "researchAgent/test")
    assert client._session.hooks["response"].count(arxiv_fetch._raise_on_429) == 1
    assert client._session.headers["User-Agent"] == "researchAgent/test"


def test_a_429_writes_a_cooldown_that_blocks_the_next_call_without_a_request(cfg, monkeypatch, planned):
    """The agent can vary its wording and call again; the memo makes that cost 0 requests.

    Loop detection only catches verbatim repeats, and `eval/build_corpus.ts` moves to its
    next topic on error — so without the memo one 429 becomes one per remaining topic.
    """
    results = [StubResult("2101.00001v1")]
    results[0].download_error = arxiv_fetch.RateLimited(results[0].pdf_url, retry_after="300")
    stub_arxiv(monkeypatch, results)

    first = collect.search_and_fetch("a topic", config=cfg, max_results=1)
    assert first["error"] == "arxiv_rate_limited"
    assert cfg.paths.arxiv_cooldown.is_file()

    searched = []
    monkeypatch.setattr(
        collect, "_build_client",
        lambda config: searched.append(1) or (_ for _ in ()).throw(AssertionError("requested")),
    )
    second = collect.search_and_fetch("a different topic", config=cfg, max_results=1)

    assert second["error"] == "arxiv_rate_limited"
    assert searched == [], "no client was even built, so no request left the machine"
    assert "Delete" in second["detail"], "the message must say how to clear the memo"


def test_an_expired_cooldown_does_not_block(cfg, monkeypatch, planned):
    """A stale memo must never become a silent permanent lockout."""
    arxiv_fetch.write_cooldown(cfg.paths.arxiv_cooldown, retry_after="0", default_s=900)
    assert arxiv_fetch.read_cooldown(cfg.paths.arxiv_cooldown) == 0.0

    stub_arxiv(monkeypatch, [StubResult("2101.00001v1")])
    assert len(collect.search_and_fetch("a topic", config=cfg)["papers_added"]) == 1


def test_a_corrupt_cooldown_memo_reads_as_no_cooldown(cfg):
    cfg.paths.arxiv_cooldown.write_text("{not json", encoding="utf-8")
    assert arxiv_fetch.read_cooldown(cfg.paths.arxiv_cooldown) == 0.0


def test_retry_after_is_honoured_over_the_configured_default(cfg):
    seconds = arxiv_fetch.write_cooldown(cfg.paths.arxiv_cooldown, retry_after="42", default_s=900)
    assert seconds == 42
    # An HTTP-date Retry-After is not parsed; the default must apply rather than crash.
    assert arxiv_fetch.write_cooldown(
        cfg.paths.arxiv_cooldown, retry_after="Wed, 21 Oct 2026 07:28:00 GMT", default_s=900
    ) == 900
