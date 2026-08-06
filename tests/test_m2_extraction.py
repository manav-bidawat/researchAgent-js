"""
Offline tests for M2: tokenization, extraction, chunking, figures, description, ingest.

In:  synthetic PDFs built with PyMuPDF, a tmp_path config, and a stub vision client.
Out: assertions on token budgets, hyphen repair, heading rules, caption matching,
     description caching, dense positions, and within-paper-only dedup. No network.
"""

import fitz
import pytest

from extraction import describe as describe_module
from extraction import figures as figures_module
from extraction.chunker import Chunker, _join_lines, hyphenated_vocabulary
from corpus.chunk_store import ChunkStore
from config import load_config
from extraction.describe import DescriptionCache, describe_figure, figure_chunk_text
from extraction.ingest import _dedupe_within_paper, _ordered, ingest_paper
from corpus.manifest import Manifest
from extraction.pdf_text import body_font_size, detect_heading, extract_pages
from common.records import paper_record
from common.tokenization import TokenCounter, split_sentences


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


def make_pdf(path, pages):
    """Build a PDF. `pages` is a list of [(text, x, y, size), ...] per page."""
    document = fitz.open()
    for items in pages:
        page = document.new_page(width=595, height=842)
        for text, x, y, size in items:
            page.insert_text((x, y), text, fontsize=size)
    document.save(str(path))
    document.close()
    return path


# ------------------------------------------------------------------------ tokenization

def test_token_counter_is_exact_and_truncates_to_budget():
    counter = TokenCounter()
    assert counter.exact, "the bi-encoder tokenizer should be available"
    text = "We evaluate on the benchmark with a batch size of 32. " * 40
    assert counter.count(text) > 100
    assert counter.count(counter.truncate(text, 50)) <= 50


def test_sentence_split_protects_decimals_and_abbreviations():
    sentences = split_sentences(
        "We reach 0.95 accuracy (see Fig. 3). Prior work, e.g. Smith et al. 2020, differs."
    )
    assert len(sentences) == 2
    assert "0.95" in sentences[0] and "Fig. 3" in sentences[0]
    assert "e.g." in sentences[1] and "et al." in sentences[1]


# ---------------------------------------------------------------------------- headings

@pytest.mark.parametrize(
    "text,size,expected",
    [
        ("Abstract", 12.0, "Abstract"),
        ("1. Introduction", 12.0, "1. Introduction"),
        ("3. Experiments", 12.0, "3. Experiments"),
        ("1. We propose a simplified mobile-friendly design", 10.0, None),
        ("500M", 6.5, None),
        ("N/A", 9.0, None),
        ("2297M", 9.0, None),
    ],
)
def test_heading_detection_separates_headings_from_lists_and_table_cells(text, size, expected):
    """Body 9.0, delta 1.5: real headings clear the bar, contribution lists and table
    cells do not. Font name is not consulted because many arXiv PDFs never say 'bold'."""
    assert detect_heading(text, size, body_size=9.0, bold=False, size_delta=1.5) == expected


def test_body_font_size_uses_the_character_weighted_mode():
    """A median over lines is dragged down by many short caption and reference lines,
    which then makes real body prose look prominent enough to be a heading."""
    pages = [[("x" * 500, 10.0, False)] + [("short", 7.0, False)] * 40]
    assert body_font_size(pages) == 10.0


# ---------------------------------------------------------------------------- chunking

def test_hyphen_repair_uses_the_documents_own_vocabulary():
    compounds = {"self-attention", "well-known"}
    assert _join_lines(["vision appli-", "cations via"], compounds) == "vision applications via"
    assert _join_lines(["self-", "attention layers"], compounds) == "self-attention layers"
    assert _join_lines(["multi-", "modal input"], compounds) == "multimodal input"


def test_hyphenated_vocabulary_ignores_the_wrap_itself():
    pages = [{"lines": [{"text": "we use self-attention here"}, {"text": "vision appli-"}]}]
    vocabulary = hyphenated_vocabulary(pages)
    assert "self-attention" in vocabulary
    assert not any(entry.startswith("appli") for entry in vocabulary)


def test_no_chunk_exceeds_the_configured_ceiling(cfg):
    sentence = "The routing network selects two experts per token in every layer. "
    pages = [{"page": 1, "lines": [{"text": sentence * 60, "section": "1. Method"}]}]
    chunks = Chunker(cfg).chunk_pages(pages)
    assert chunks
    assert all(c["n_tokens"] <= cfg.chunking.max_tokens for c in chunks)


def test_overlap_is_not_carried_across_a_section_boundary(cfg):
    pages = [{"page": 1, "lines": [
        {"text": "Alpha sentence one. " * 30, "section": "1. Introduction"},
        {"text": "Beta sentence two. " * 30, "section": "2. Method"},
    ]}]
    chunks = Chunker(cfg).chunk_pages(pages)
    beta = [c for c in chunks if c["section"] == "2. Method"]
    assert beta and not any("Alpha" in c["text"] for c in beta)


# ----------------------------------------------------------------------------- figures

@pytest.mark.parametrize(
    "text,matches",
    [
        ("Figure 3: Routing similarity across tasks.", True),
        ("Figure 2. Model architectures.", True),
        ("Table 4. Varying the number of experts.", True),
        ("Figure 1 shows the similarity matrix across tasks", False),
        ("As Table 2 makes clear, the gap widens", False),
    ],
)
def test_caption_regex_requires_a_separator(text, matches):
    """Without a mandatory separator this also matches prose that merely refers to a
    figure, and that reference usually precedes the figure, so it steals the label."""
    assert bool(figures_module._CAPTION_START.match(text)) is matches


def test_caption_kind_distinguishes_tables_from_figures():
    assert figures_module._kind_of("Table") == "table"
    assert figures_module._kind_of("Fig.") == "figure"


def test_extract_figures_on_a_pdf_with_no_captions(cfg, tmp_path):
    pdf = make_pdf(tmp_path / "plain.pdf", [[("Just prose, no figures here.", 72, 100, 11)]])
    result = figures_module.extract_figures(pdf, "p1", cfg)
    assert result["status"] == "ok"
    assert result["figures"] == []


def test_extract_figures_reports_a_missing_file(cfg, tmp_path):
    result = figures_module.extract_figures(tmp_path / "nope.pdf", "p1", cfg)
    assert result["status"] == "failed"
    assert "no file" in result["note"]


def test_reference_sections_are_dropped(cfg, tmp_path):
    """A bibliography retrieves badly — every entry looks relevant to any query about the
    field — and it wrecks heading detection, which is how "[Iccv, 2021. 2]" became a
    section name. Everything under References is skipped."""
    pdf = make_pdf(tmp_path / "refs.pdf", [[
        ("1. Introduction", 72, 70, 14),
        ("We study routing in sparse mixture of expert models here.", 72, 110, 10),
        ("References", 72, 160, 14),
        ("[12] Andrew Howard and others. Searching for mobilenetv3. ICCV, 2019.", 72, 200, 10),
    ]])
    result = extract_pages(pdf, cfg)
    text = " ".join(line["text"] for page in result["pages"] for line in page["lines"])
    assert "routing in sparse" in text
    assert "mobilenetv3" not in text
    assert "References" not in {line["section"] for page in result["pages"]
                                for line in page["lines"]}


def test_appendix_after_references_is_kept(cfg, tmp_path):
    """Appendices hold real results, so an Appendix heading ends the skip."""
    pdf = make_pdf(tmp_path / "appendix.pdf", [[
        ("References", 72, 70, 14),
        ("[1] Someone. A cited paper title here. In Proceedings, 2019.", 72, 110, 10),
        ("Appendix A", 72, 160, 14),
        ("We report additional ablations on the validation split here.", 72, 200, 10),
    ]])
    result = extract_pages(pdf, cfg)
    text = " ".join(line["text"] for page in result["pages"] for line in page["lines"])
    assert "additional ablations" in text
    assert "cited paper title" not in text


def test_single_prominent_word_is_not_a_section(cfg):
    """One prominent word is far more often a table header than a section title."""
    assert detect_heading("PSPNet", 12.0, body_size=9.0, bold=False, size_delta=1.5) is None
    assert detect_heading("BDD100K", 12.0, body_size=9.0, bold=False, size_delta=1.5) is None
    assert detect_heading("Abstract", 12.0, body_size=9.0, bold=False, size_delta=1.5) == "Abstract"


# ------------------------------------------------------------------------ pdf_text io

def test_extract_pages_reports_a_corrupt_pdf(cfg, tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    result = extract_pages(broken, cfg)
    assert result["status"] == "failed"
    assert result["pages"] == []


def test_extract_pages_reads_text_and_pages(cfg, tmp_path):
    pdf = make_pdf(tmp_path / "two.pdf", [
        [("Abstract", 72, 80, 14), ("We study routing in sparse models today.", 72, 120, 10)],
        [("More prose on the second page of the paper.", 72, 120, 10)],
    ])
    result = extract_pages(pdf, cfg)
    assert result["status"] == "ok" and result["n_pages"] == 2
    assert [p["page"] for p in result["pages"]] == [1, 2]
    assert any("routing" in line["text"] for line in result["pages"][0]["lines"])


# ------------------------------------------------------------------------- description

def test_description_cache_prevents_a_second_vision_call(cfg, tmp_path):
    image = tmp_path / "f.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    figure = {"figure_id": "p__f01", "kind": "figure", "caption": "Figure 1: A plot.",
              "image_path": str(image), "image_hash": "abc123"}

    calls = []

    class StubVision:
        def complete_vision(self, messages, images, role="vision"):
            calls.append(1)
            return type("R", (), {"text": "Scatter of accuracy against FLOPs."})()

    cache = DescriptionCache(cfg)
    first = describe_figure(figure, client=StubVision(), cache=cache, config=cfg)
    second = describe_figure(figure, client=StubVision(), cache=cache, config=cfg)

    assert first["description_cached"] is False and second["description_cached"] is True
    assert second["description"] == first["description"]
    assert len(calls) == 1, "the second call must be served from the cache"


def test_unreadable_verdict_is_cached_so_it_is_not_re_asked(cfg, tmp_path):
    image = tmp_path / "f.png"
    image.write_bytes(b"\x89PNG")
    figure = {"figure_id": "p__f01", "kind": "figure", "caption": "", "image_path": str(image),
              "image_hash": "deadbeef"}

    class StubVision:
        def complete_vision(self, *a, **k):
            return type("R", (), {"text": "UNREADABLE"})()

    cache = DescriptionCache(cfg)
    result = describe_figure(figure, client=StubVision(), cache=cache, config=cfg)
    assert result["description"] == ""
    assert cache.get("deadbeef") is None
    assert "deadbeef" in cache._data, "the verdict itself is cached to avoid re-asking"


def test_vision_failure_degrades_to_caption_only(cfg, tmp_path):
    from llm_client import LLMError

    image = tmp_path / "f.png"
    image.write_bytes(b"\x89PNG")
    figure = {"figure_id": "p__f01", "kind": "figure", "caption": "Figure 1: A plot.",
              "image_path": str(image), "image_hash": "cafe"}

    class DeadVision:
        def complete_vision(self, *a, **k):
            raise LLMError("llm_transient", "rate limited")

    result = describe_figure(figure, client=DeadVision(), cache=DescriptionCache(cfg), config=cfg)
    assert result["description"] == ""
    assert "description_error" in result
    assert figure_chunk_text(figure["caption"], result["description"]) == "Figure 1: A plot."


def test_figure_text_is_caption_and_description_concatenated():
    text = figure_chunk_text("Figure 2: Top-1 accuracy.", "Line plot rising from 71 to 74.")
    assert text.startswith("Figure 2: Top-1 accuracy.")
    assert "Line plot" in text
    assert "\n" in text, "caption and description are joined, never one alone"


# ------------------------------------------------------------------------------ ingest

def test_figures_are_interleaved_by_page_not_appended():
    """position +/- 1 must land on adjacent content, so a page's figure follows that
    page's prose rather than the whole document's."""
    text = [{"page": 1, "text": "a"}, {"page": 2, "text": "b"}, {"page": 3, "text": "c"}]
    figs = [{"page": 2, "text": "fig on page two"}]
    order = [item["text"] for item in _ordered(text, figs)]
    assert order == ["a", "b", "fig on page two", "c"]


def test_dedup_is_within_paper_only():
    items = [{"text": "same"}, {"text": "other"}, {"text": "same"}]
    kept, dropped = _dedupe_within_paper(items)
    assert dropped == 1
    assert [item["text"] for item in kept] == ["same", "other"]


def test_ingest_assigns_dense_positions_after_dedup(cfg, tmp_path):
    repeated = "The gating network routes each token to two experts. "
    pdf = make_pdf(cfg.paths.paper_pdf("p1"), [[
        ("Abstract", 72, 70, 14),
        (repeated * 3, 72, 110, 10),
        (repeated * 3, 72, 160, 10),
        ("A different sentence about load balancing losses entirely.", 72, 210, 10),
    ]])
    manifest = Manifest.load(cfg)
    manifest.add_paper(paper_record(
        arxiv_id="p1", title="T", authors=[], abstract="", published="2024-01-01", year=2024,
        categories=[], pdf_url="", pdf_path=str(pdf), topic_tags=["t"]))
    manifest.save()

    result = ingest_paper("p1", config=cfg, describe=False)
    assert "error" not in result, result

    chunks = ChunkStore(config=cfg).for_paper("p1")
    positions = [c["position"] for c in chunks]
    assert positions == list(range(len(chunks))), "positions must be dense after dedup"
    assert all(c["chunk_id"] == f"p1__c{c['position']:04d}" for c in chunks)


def test_ingest_rejects_an_unknown_paper(cfg):
    result = ingest_paper("nope", config=cfg, describe=False)
    assert result["error"] == "unknown_paper"


def test_ingest_reports_a_corrupt_pdf_as_parse_failed(cfg):
    pdf = cfg.paths.paper_pdf("bad")
    pdf.write_bytes(b"garbage")
    manifest = Manifest.load(cfg)
    manifest.add_paper(paper_record(
        arxiv_id="bad", title="T", authors=[], abstract="", published="2024-01-01", year=2024,
        categories=[], pdf_url="", pdf_path=str(pdf), topic_tags=["t"]))
    manifest.save()

    result = ingest_paper("bad", config=cfg, describe=False)
    assert result["error"] == "parse_failed"
    assert Manifest.load(cfg).get_paper("bad")["parse_status"] == "failed"
