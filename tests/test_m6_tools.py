"""
Offline smoke tests for M6: the four remaining tools, return shapes and error paths.

In:  a tmp_path corpus, stub NLI and stub LLM — no models, no network, no arXiv.
Out: assertions that each tool returns its documented shape and an error dict on bad
     input rather than raising, per docs/TOOLS.md.
"""

import json

import numpy as np
import pytest

from agent.tool_registry import ToolRegistry, build_full_registry
from common.records import chunk_record, paper_record
from config import load_config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest
from tools.analyze_corpus import CorpusAnalyzer
from tools.check_evidence_consistency import ConsistencyChecker
from tools.inspect_figure import FigureInspector


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


def seed(cfg, papers=2, chunks_per=4):
    manifest = Manifest.load(cfg, strict_model_check=False)
    store = ChunkStore(config=cfg)
    for index in range(papers):
        paper_id = f"p{index}"
        tag = f"topic{index}"
        manifest.add_paper(paper_record(
            arxiv_id=paper_id, title=f"Paper {index}", authors=["A"], abstract="x",
            published=f"202{index}-01-01", year=2020 + index, categories=["cs.LG"],
            pdf_url="", pdf_path="", topic_tags=[tag]))
        records = [
            chunk_record(paper_id=paper_id, paper_title=f"Paper {index}", topic_tags=[tag],
                         chunk_type="text",
                         text=f"Paper {index} routes each token to {index + 1} experts "
                              f"using a learned gating network. It reports strong results.",
                         page=1, position=position, n_tokens=20)
            for position in range(chunks_per)
        ]
        store.append(records)
    manifest.save()
    return manifest, store


def seed_figure(cfg, tmp_path):
    manifest, store = seed(cfg, papers=1, chunks_per=1)
    image = tmp_path / "fig.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest.figures["p0__f01"] = {
        "figure_id": "p0__f01", "paper_id": "p0", "kind": "figure", "page": 2,
        "image_path": str(image), "image_hash": "abc", "caption": "Figure 1: A plot.",
        "description": "A line chart rising from 71 to 74.", "width": 400, "height": 300,
        "chunk_id": "p0__c0001",
    }
    store.append([chunk_record(
        paper_id="p0", paper_title="Paper 0", topic_tags=["topic0"], chunk_type="figure",
        text="Figure 1: A plot.\nA line chart rising from 71 to 74.", page=2, position=1,
        n_tokens=12, figure_id_value="p0__f01", image_path=str(image),
        caption="Figure 1: A plot.")])
    manifest.save()
    return manifest


class StubNLI:
    """Returns a scripted label for every pair, so tests need no model download."""

    def __init__(self, label="neutral", confidence=0.9):
        self.label = label
        self.confidence = confidence
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        scores = {"contradiction": 0.05, "entailment": 0.05, "neutral": 0.05}
        scores[self.label] = self.confidence
        return [{"label": self.label, "confidence": self.confidence, "scores": dict(scores)}
                for _ in pairs]


class StubLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return type("R", (), {"text": self.text, "model": "stub"})()


# ------------------------------------------------------------------- analyze_corpus

def test_stats_returns_the_documented_shape(cfg):
    seed(cfg)
    result = CorpusAnalyzer(cfg).analyze("stats")
    assert set(result) == {"operation", "result", "summary"}
    assert result["operation"] == "stats"
    assert result["result"]["n_papers"] == 2
    assert result["result"]["n_chunks"] == 8
    assert isinstance(result["summary"], str) and result["summary"]


def test_summary_is_templated_from_the_result_not_generated(cfg):
    """No LLM call, so the prose cannot disagree with the numbers it describes."""
    seed(cfg)
    result = CorpusAnalyzer(cfg).analyze("stats")
    assert str(result["result"]["n_papers"]) in result["summary"]
    assert str(result["result"]["n_chunks"]) in result["summary"]


def test_timeline_reports_years(cfg):
    seed(cfg)
    result = CorpusAnalyzer(cfg).analyze("timeline")
    assert result["result"]["oldest_year"] == 2020
    assert result["result"]["newest_year"] == 2021
    assert set(result["result"]["per_topic"]) == {"topic0", "topic1"}


def test_topic_filter_narrows_the_corpus(cfg):
    seed(cfg)
    result = CorpusAnalyzer(cfg).analyze("stats", topic_filter="topic0")
    assert result["result"]["n_papers"] == 1


def test_unknown_operation_is_an_error_dict(cfg):
    seed(cfg)
    assert CorpusAnalyzer(cfg).analyze("nonsense")["error"] == "unknown_operation"


def test_empty_corpus_is_an_error_dict(cfg):
    assert CorpusAnalyzer(cfg).analyze("stats")["error"] == "empty_corpus"


def test_unknown_topic_filter_is_an_error_dict(cfg):
    seed(cfg)
    assert CorpusAnalyzer(cfg).analyze("stats", topic_filter="nope")["error"] == "empty_corpus"


def test_compare_topics_needs_two_topics(cfg):
    seed(cfg, papers=1)
    assert CorpusAnalyzer(cfg).analyze("compare_topics")["error"] == "too_few_topics"


def test_cluster_without_an_index_is_an_error_dict(cfg):
    seed(cfg)
    assert CorpusAnalyzer(cfg).analyze("cluster")["error"] == "no_index"


# -------------------------------------------------------------------- inspect_figure

def test_inspect_returns_a_path_never_image_bytes(cfg, tmp_path):
    """The OpenAI-compatible schema takes image parts only on a user turn, so the tool
    hands back a path and the loop attaches the file."""
    seed_figure(cfg, tmp_path)
    result = FigureInspector(cfg).inspect(figure_id="p0__f01")
    assert set(result) >= {"image_path", "mime_type", "caption", "stored_description",
                           "paper_id", "page", "figure_id"}
    assert result["mime_type"] == "image/png"
    assert "A line chart" in result["stored_description"]
    assert not any(isinstance(v, bytes) for v in result.values())


def test_inspect_resolves_a_chunk_id(cfg, tmp_path):
    seed_figure(cfg, tmp_path)
    assert FigureInspector(cfg).inspect(chunk_id="p0__c0001")["figure_id"] == "p0__f01"


def test_inspect_rejects_a_text_chunk(cfg, tmp_path):
    seed_figure(cfg, tmp_path)
    assert FigureInspector(cfg).inspect(chunk_id="p0__c0000")["error"] == "not_a_figure"


def test_inspect_needs_some_reference(cfg):
    assert FigureInspector(cfg).inspect()["error"] == "missing_reference"


def test_inspect_reports_a_paper_mismatch(cfg, tmp_path):
    seed_figure(cfg, tmp_path)
    result = FigureInspector(cfg).inspect(figure_id="p0__f01", paper_id="pX")
    assert result["error"] == "figure_paper_mismatch"


def test_inspect_reports_a_missing_image_file(cfg, tmp_path):
    manifest = seed_figure(cfg, tmp_path)
    (tmp_path / "fig.png").unlink()
    assert FigureInspector(cfg).inspect(figure_id="p0__f01")["error"] == "image_missing"


def test_user_image_needs_to_be_an_image(cfg, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not an image")
    assert FigureInspector(cfg).inspect(user_image=str(notes))["error"] == "not_an_image"


def test_user_image_has_no_caption_or_stored_description(cfg, tmp_path):
    image = tmp_path / "user.png"
    image.write_bytes(b"\x89PNG")
    result = FigureInspector(cfg).inspect(user_image=str(image), question="what is shown?")
    assert result["source"] == "user"
    assert "caption" not in result and "stored_description" not in result


# ------------------------------------------------------- check_evidence_consistency

def test_contradiction_returns_the_documented_shape(cfg):
    seed(cfg)
    checker = ConsistencyChecker(cfg, nli=StubNLI("contradiction", 0.95))
    result = checker.check("contradiction", ["p0__c0000", "p1__c0000"])
    assert set(result) >= {"found", "conflicting_pairs"}
    assert result["found"] is True
    pair = result["conflicting_pairs"][0]
    assert set(pair) == {"chunk_a", "chunk_b", "snippet_a", "snippet_b", "confidence"}


def test_contradiction_below_threshold_reports_nothing(cfg):
    seed(cfg)
    checker = ConsistencyChecker(cfg, nli=StubNLI("neutral", 0.99))
    result = checker.check("contradiction", ["p0__c0000", "p1__c0000"])
    assert result["found"] is False
    assert result["conflicting_pairs"] == []


def test_contradiction_needs_two_chunks(cfg):
    seed(cfg)
    checker = ConsistencyChecker(cfg, nli=StubNLI())
    assert checker.check("contradiction", ["p0__c0000"])["error"] == "too_few_chunks"


def test_groundedness_returns_the_documented_shape(cfg):
    seed(cfg)
    checker = ConsistencyChecker(
        cfg, client=StubLLM('["Paper 0 routes each token to one expert."]'),
        nli=StubNLI("entailment", 0.9))
    result = checker.check("groundedness", ["p0__c0000"], answer_text="Some drafted answer.")

    assert set(result) >= {"claims", "grounded_ratio", "unsupported_claims"}
    assert result["grounded_ratio"] == 1.0
    claim = result["claims"][0]
    assert set(claim) == {"claim", "label", "best_supporting_chunk", "confidence"}
    assert claim["label"] == "entailed"


def test_unsupported_claims_are_listed_explicitly(cfg):
    seed(cfg)
    checker = ConsistencyChecker(
        cfg, client=StubLLM('["An unsupported assertion.", "Another one."]'),
        nli=StubNLI("neutral", 0.9))
    result = checker.check("groundedness", ["p0__c0000"], answer_text="draft")

    assert result["grounded_ratio"] == 0.0
    assert len(result["unsupported_claims"]) == 2
    assert all(c["label"] == "neutral" for c in result["claims"])


def test_claims_are_extracted_inside_the_tool(cfg):
    """If the agent supplied its own claim list it would be grading a list it chose."""
    seed(cfg)
    llm = StubLLM('["A claim."]')
    ConsistencyChecker(cfg, client=llm, nli=StubNLI()).check(
        "groundedness", ["p0__c0000"], answer_text="Some answer text here.")
    assert llm.calls == 1, "the tool makes its own extraction call"


def test_citations_are_stripped_before_claim_extraction(cfg):
    seed(cfg)

    class Recorder(StubLLM):
        def __init__(self):
            super().__init__('["x"]')
            self.seen = ""

        def complete(self, messages, **kwargs):
            self.seen = messages[-1]["content"]
            return super().complete(messages, **kwargs)

    recorder = Recorder()
    ConsistencyChecker(cfg, client=recorder, nli=StubNLI()).check(
        "groundedness", ["p0__c0000"], answer_text="Routing works [p0__c0000] well.")
    assert "[p0__c0000]" not in recorder.seen


def test_groundedness_needs_answer_text(cfg):
    seed(cfg)
    checker = ConsistencyChecker(cfg, nli=StubNLI())
    assert checker.check("groundedness", ["p0__c0000"])["error"] == "missing_answer_text"


def test_unknown_mode_is_an_error_dict(cfg):
    seed(cfg)
    assert ConsistencyChecker(cfg, nli=StubNLI()).check("vibes", ["p0__c0000"])["error"] \
        == "unknown_mode"


def test_unknown_chunk_ids_are_reported(cfg):
    seed(cfg)
    result = ConsistencyChecker(cfg, nli=StubNLI()).check("groundedness", ["ghost"],
                                                          answer_text="x")
    assert result["error"] == "no_chunks"


def test_unrelated_sentences_are_not_compared_at_all(cfg):
    """MNLI models score topically unrelated pairs as confident contradictions, so pairs
    below the similarity floor never reach the NLI model. Measured: a real conflict sits
    at ~0.85 cosine, an unrelated same-paper pair at ~0.58."""
    manifest = Manifest.load(cfg, strict_model_check=False)
    store = ChunkStore(config=cfg)
    for paper_id, text in (
        ("pA", "Existing work has largely focused on training stability and scaling."),
        ("pB", "Story prompts emphasize open-ended generation of narrative text."),
    ):
        manifest.add_paper(paper_record(
            arxiv_id=paper_id, title="T", authors=[], abstract="", published="2024-01-01",
            year=2024, categories=[], pdf_url="", pdf_path="", topic_tags=["x"]))
        store.append([chunk_record(
            paper_id=paper_id, paper_title="T", topic_tags=["x"], chunk_type="text",
            text=text, page=1, position=0, n_tokens=15)])
    manifest.save()

    nli = StubNLI("contradiction", 0.99)
    result = ConsistencyChecker(cfg, nli=nli).check("contradiction", ["pA__c0000", "pB__c0000"])

    assert result["n_pairs_scored"] == 0, "unrelated sentences must not reach the NLI model"
    assert result["found"] is False
    assert nli.pairs == [], "the NLI model was never called"


def test_groundedness_scores_only_the_most_similar_premises(cfg):
    """One claim is checked against premises_per_claim sentences, not every sentence.
    Scoring all of them and taking the strongest verdict lets a spurious contradiction
    from an unrelated sentence win."""
    seed(cfg, papers=1, chunks_per=6)
    nli = StubNLI("entailment", 0.9)
    ConsistencyChecker(cfg, client=StubLLM('["A claim about routing experts."]'),
                       nli=nli).check("groundedness", [f"p0__c000{i}" for i in range(6)],
                                      answer_text="draft")
    assert len(nli.pairs) <= int(cfg.nli.premises_per_claim)


# ------------------------------------------------------------------------- registry

def test_all_five_tools_are_registered(cfg):
    seed(cfg)
    registry = build_full_registry(config=cfg)
    assert registry.names == sorted([
        "analyze_corpus", "check_evidence_consistency", "inspect_figure",
        "retrieve_evidence", "search_literature",
    ])


def test_every_tool_has_a_description_from_a_file(cfg):
    seed(cfg)
    for schema in build_full_registry(config=cfg).schemas():
        function = schema["function"]
        assert len(function["description"]) > 80, f"{function['name']} has no real description"
        assert function["parameters"]["type"] == "object"


def test_search_literature_invalidates_the_retriever_cache(cfg):
    """Papers added mid-conversation are invisible to a retriever holding a cached index."""
    from tools.retrieve_evidence import EvidenceRetriever

    retriever = EvidenceRetriever(cfg)
    retriever._index = "stale"
    retriever._chunks = {"old": "view"}
    retriever.invalidate()
    assert retriever._index is None and retriever._chunks is None
