"""
Offline tests for M4: the relevance gate, filters, expansion, and cross-call dedup.

In:  a tmp_path corpus with a stub embedder and stub reranker — no models, no network.
Out: assertions that weak chunks are withheld, filters are membership tests, neighbours
     attach within budget, and repeated queries do not restack the context window.
"""

import numpy as np
import pytest

from common.records import chunk_record, paper_record
from config import load_config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest
from retrieval.embedder import Embedder
from retrieval.indexer import index_chunks
from tools.retrieve_evidence import EvidenceRetriever


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    import config as config_module

    repo_root = config_module.REPO_ROOT
    local = tmp_path / "config.yaml"
    local.write_text((repo_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    built = load_config(local)
    built.paths.ensure()
    built.paths.prompts = repo_root / "src" / "prompts"
    return built


def config_with_retrieval(base, **overrides):
    """A copy of `base` with retrieval settings overridden.

    Written through the YAML because config sections are read-only by design: a test
    that pokes them in place would assert against a state the real system cannot reach.
    """
    import yaml

    raw = yaml.safe_load(base.source_path.read_text(encoding="utf-8"))
    raw["retrieval"].update(overrides)
    scratch = base.paths.root / f"retrieval_override_{abs(hash(tuple(sorted(overrides.items()))))}.yaml"
    scratch.write_text(yaml.safe_dump(raw), encoding="utf-8")
    built = load_config(scratch)
    built.paths.prompts = base.paths.prompts
    return built


class StubEmbedder(Embedder):
    """Deterministic unit vectors; never downloads a model."""

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, abs(hash(text)) % self.dim] = 1.0
        return out


class StubReranker:
    """Scores by a caller-supplied table, and records what it was asked to score."""

    def __init__(self, scores=None, default=1.0):
        self.scores = scores or {}
        self.default = default
        self.batch_sizes = []
        self.last_latency_ms = 7

    def rerank(self, query, candidates):
        self.batch_sizes.append(len(candidates))
        scored = [{**c, "rerank_score": self.scores.get(c["chunk_id"], self.default)}
                  for c in candidates]
        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored


def build_corpus(cfg, papers):
    """papers: {paper_id: (topic_tags, [(text, chunk_type), ...])}"""
    manifest = Manifest.load(cfg, strict_model_check=False)
    store = ChunkStore(config=cfg)
    for paper_id, (tags, items) in papers.items():
        manifest.add_paper(paper_record(
            arxiv_id=paper_id, title=f"Title {paper_id}", authors=[], abstract="",
            published="2024-01-01", year=2024, categories=[], pdf_url="", pdf_path="",
            topic_tags=tags))
        records = []
        for position, (text, chunk_type) in enumerate(items):
            records.append(chunk_record(
                paper_id=paper_id, paper_title=f"Title {paper_id}", topic_tags=tags,
                chunk_type=chunk_type, text=text, page=1, position=position, n_tokens=10,
                section="S", figure_id_value=(f"{paper_id}__f01" if chunk_type != "text" else None),
                image_path=("/tmp/x.png" if chunk_type != "text" else None),
                caption=("Figure 1: cap" if chunk_type != "text" else None)))
        store.append(records)
    manifest.save()
    index_chunks(cfg, embedder=StubEmbedder(cfg))


@pytest.fixture()
def corpus(cfg):
    build_corpus(cfg, {
        "pA": (["alpha"], [(f"alpha chunk {i} about routing", "text") for i in range(5)]),
        "pB": (["beta"], [("beta chunk about baking bread", "text"),
                          ("Figure 1: cap\na described figure", "figure")]),
    })
    return cfg


def retriever(cfg, scores=None, default=1.0):
    return EvidenceRetriever(cfg, embedder=StubEmbedder(cfg),
                             reranker=StubReranker(scores, default))


# --------------------------------------------------------------------- input handling

def test_empty_query_is_an_error_dict(corpus):
    assert retriever(corpus).retrieve("")["error"] == "empty_query"


def test_unknown_chunk_types_is_an_error_dict(corpus):
    result = retriever(corpus).retrieve("x", chunk_types=["diagram"])
    assert result["error"] == "bad_chunk_types"
    assert "diagram" in result["detail"]


def test_non_positive_k_is_an_error_dict(corpus):
    assert retriever(corpus).retrieve("x", k=0)["error"] == "bad_k"


def test_empty_index_is_an_error_dict(cfg):
    assert retriever(cfg).retrieve("anything")["error"] == "empty_index"


# ------------------------------------------------------------------- the relevance gate

def test_gate_fires_when_everything_scores_below_threshold(corpus):
    result = retriever(corpus, default=-5.0).retrieve("anything at all")
    assert result["sufficient_evidence"] is False
    assert result["chunks"] == []
    assert "below the threshold" in result["note"]
    assert result["n_candidates_considered"] > 0, "the gate reports what it considered"


def test_weak_chunks_are_withheld_even_when_the_best_one_passes(corpus):
    """Gating on the top score alone still hands back everything beneath it, which is
    what docs/ARCHITECTURE.md section 7 forbids.

    Scores are placed relative to the configured threshold rather than hardcoded. The
    threshold is a tuned value that moved from 0.0 to -3.0 in M8, and a test pinned to
    one number tests the number rather than the behaviour.
    """
    threshold = float(corpus.retrieval.relevance_threshold)
    scores = {"pA__c0000": threshold + 5.0, "pA__c0001": threshold - 1.0,
              "pA__c0002": threshold - 2.0, "pA__c0003": threshold - 3.0,
              "pA__c0004": threshold - 4.0}
    result = retriever(corpus, scores=scores, default=threshold - 9.0).retrieve("routing", k=5)

    assert result["sufficient_evidence"] is True
    assert [c["chunk_id"] for c in result["chunks"]] == ["pA__c0000"]
    assert "below the relevance threshold" in result["note"]


def test_gate_reports_scores_it_used(corpus):
    result = retriever(corpus, default=3.0).retrieve("routing", k=2)
    assert all(c["score"] == 3.0 for c in result["chunks"])


# ------------------------------------------------------------------------- the filters

def test_topic_filter_is_a_membership_test_not_a_substring_match(cfg):
    """A bare string would match character-by-character; tags are a list for this reason."""
    build_corpus(cfg, {"pA": (["alpha"], [("alpha text", "text")]),
                       "pB": (["alphabet"], [("alphabet text", "text")])})
    result = retriever(cfg).retrieve("text", topic_filter="alpha", k=5)
    ids = {c["chunk_id"] for c in result["chunks"]}
    assert ids == {"pA__c0000"}, "'alpha' must not match the paper tagged 'alphabet'"


def test_chunk_types_filter_restricts_the_pool(corpus):
    result = retriever(corpus).retrieve("anything", chunk_types=["figure"], k=5)
    assert result["chunks"]
    assert all(c["chunk_type"] == "figure" for c in result["chunks"])
    assert all("figure_id" in c for c in result["chunks"])


def test_unmatched_filter_fails_the_gate_rather_than_being_ignored(corpus):
    result = retriever(corpus).retrieve("anything", topic_filter="nope")
    assert result["sufficient_evidence"] is False
    assert result["chunks"] == []
    assert "no chunks match the filter" in result["note"]


# -------------------------------------------------------------- neighbours and budget

def test_neighbour_expansion_attaches_adjacent_chunks(corpus):
    result = retriever(corpus, scores={"pA__c0002": 9.0}, default=-9.0).retrieve("routing", k=1)
    text = result["chunks"][0]["text"]
    assert "alpha chunk 2" in text
    assert "alpha chunk 1" in text and "alpha chunk 3" in text, "position +/- 1 attaches"
    assert "alpha chunk 0" not in text, "expansion is one step, not the whole paper"


def test_expansion_never_crosses_into_another_paper(corpus):
    """position is per-paper, so a neighbour lookup must stay inside its own paper."""
    result = retriever(corpus, scores={"pB__c0000": 9.0}, default=-9.0).retrieve("bread", k=1)
    assert "alpha chunk" not in result["chunks"][0]["text"]


def test_expanded_text_respects_the_character_cap(cfg):
    build_corpus(cfg, {"pA": (["alpha"], [("x" * 3000, "text") for _ in range(3)])})
    result = retriever(cfg, scores={"pA__c0001": 9.0}, default=-9.0).retrieve("x", k=1)
    assert len(result["chunks"][0]["text"]) <= cfg.retrieval.max_chunk_chars


# ------------------------------------------------------------------- conversation state

def test_repeated_query_does_not_restack_the_same_chunks(corpus):
    agent = retriever(corpus, default=2.0)
    first = {c["chunk_id"] for c in agent.retrieve("routing", k=2)["chunks"]}
    second = {c["chunk_id"] for c in agent.retrieve("routing", k=2)["chunks"]}
    assert first and second
    assert not (first & second), "context must not accumulate duplicates across iterations"


def test_reset_clears_the_dedup_state(corpus):
    agent = retriever(corpus, default=2.0)
    first = {c["chunk_id"] for c in agent.retrieve("routing", k=2)["chunks"]}
    agent.reset()
    assert {c["chunk_id"] for c in agent.retrieve("routing", k=2)["chunks"]} == first


def test_exhausting_the_corpus_reports_why(corpus):
    agent = retriever(corpus, default=2.0)
    for _ in range(6):
        result = agent.retrieve("routing", k=5)
    assert result["sufficient_evidence"] is False
    assert "already returned earlier" in result["note"]


# ---------------------------------------------------------------------------- reranking

def test_reranker_sees_the_shortlist_never_the_whole_index(cfg):
    """Rerank is stage two, on the candidate pool. Running it over the index is the
    documented mistake this asserts against."""
    build_corpus(cfg, {"pA": (["alpha"], [(f"chunk {i}", "text") for i in range(90)])})
    reranker = StubReranker(default=1.0)
    agent = EvidenceRetriever(cfg, embedder=StubEmbedder(cfg), reranker=reranker)
    agent.retrieve("anything", k=5)

    assert reranker.batch_sizes
    assert max(reranker.batch_sizes) <= cfg.retrieval.k_retrieve
    assert max(reranker.batch_sizes) < ChunkStore(config=cfg).count()


def test_rerank_failure_falls_back_to_bi_encoder_order(corpus):
    """Rerank is precision, not correctness. Losing it should degrade ranking, not the
    ability to answer at all."""
    from retrieval.reranker import RerankError

    class DeadReranker:
        last_latency_ms = 0

        def rerank(self, query, candidates):
            raise RerankError("model checkpoint missing")

    # The gate is not what this test is about: an explicit floor of 0.0 admits whatever
    # the stub scores, so a failure here means the fallback ordering broke, not the gate.
    permissive = config_with_retrieval(corpus, bi_encoder_relevance_threshold=0.0)
    agent = EvidenceRetriever(permissive, embedder=StubEmbedder(permissive),
                              reranker=DeadReranker())
    result = agent.retrieve("routing", k=3)
    assert result["sufficient_evidence"] is True
    assert result["chunks"]
    assert "rerank unavailable" in result["note"]
    assert result["reranked"] is False


def test_the_gate_threshold_matches_whichever_scorer_ran(corpus):
    """A cross-encoder threshold applied to cosine scores lets everything through.

    Cross-encoder output is an uncalibrated logit spanning roughly -11..+11; bi-encoder
    output is a cosine in 0..1. Every cosine clears a threshold of -3.0, so reusing the
    cross-encoder's number on the fallback path does not fail loudly — it removes
    abstention and answers absent-topic questions from whatever ranked first.
    """
    from retrieval.reranker import RerankError

    class DeadReranker:
        last_latency_ms = 0

        def rerank(self, query, candidates):
            raise RerankError("model checkpoint missing")

    # A bi-encoder gate above anything the stub can score must withhold everything.
    strict = config_with_retrieval(corpus, bi_encoder_relevance_threshold=0.99)
    agent = EvidenceRetriever(strict, embedder=StubEmbedder(strict), reranker=DeadReranker())
    result = agent.retrieve("routing", k=3)
    assert result["sufficient_evidence"] is False, (
        "fallback path gated on the cross-encoder threshold, so the bi-encoder gate "
        "never applied and abstention was silently disabled"
    )
    assert result["chunks"] == []
