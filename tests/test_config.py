"""
Smoke tests for the config loader: shapes, path resolution, and loud failure on typos.

In:  the repo's real config.yaml, plus small temp files for the failure paths.
Out: assertions only. No network, no API key required.
"""

from pathlib import Path

import pytest

from config import CFG, Config, ConfigError, REPO_ROOT, load_config


def test_required_sections_are_present():
    for section in ("llm", "collection", "chunking", "embedding", "retrieval", "nli", "agent"):
        assert section in CFG.as_dict()


def test_values_have_the_documented_types():
    assert isinstance(CFG.retrieval.k_retrieve, int)
    assert isinstance(CFG.retrieval.rerank_enabled, bool)
    assert isinstance(CFG.analysis.cluster_k_range, list)
    assert CFG.retrieval.k_final < CFG.retrieval.k_retrieve


def test_chunk_ceiling_is_the_min_of_both_encoder_budgets():
    """max_tokens is derived, not picked: see docs/ARCHITECTURE.md section 5.

    Both windows are 512 and both are measured, not assumed — the tokenizers were
    checked to share a vocabulary and produce identical counts. If either model or
    max_query_tokens moves, this recomputes and the assertion below catches the drift.
    """
    window = 512

    # Bi-encoder sees the chunk alone: [CLS] chunk [SEP]
    bi_budget = window - 2

    # Cross-encoder shares one window: [CLS] query [SEP] chunk [SEP]
    cross_budget = window - 3 - CFG.retrieval.max_query_tokens

    binding = min(bi_budget, cross_budget)
    assert cross_budget < bi_budget, "the reranker is expected to be the tighter limit"
    assert CFG.chunking.max_tokens == binding, (
        f"max_tokens is {CFG.chunking.max_tokens}, but the derived ceiling is {binding} "
        f"(bi={bi_budget}, cross={cross_budget})"
    )
    assert CFG.embedding.max_seq_tokens == window
    assert CFG.chunking.target_tokens < CFG.chunking.max_tokens
    assert CFG.chunking.overlap_tokens < CFG.chunking.target_tokens


def test_bge_prefixes_are_asymmetric():
    """BGE puts the instruction on the query only; prefixing passages degrades retrieval."""
    assert CFG.embedding.model.startswith("BAAI/bge-")
    assert CFG.embedding.query_prefix.strip()
    assert CFG.embedding.passage_prefix == ""
    assert CFG.embedding.normalize is True


def test_arxiv_client_settings_are_present():
    assert CFG.collection.request_delay_s > 0
    assert CFG.collection.sort_by == "relevance"
    assert "api_url" not in CFG.collection, "the arxiv package owns its own endpoint"


def test_unknown_key_raises_instead_of_returning_none():
    with pytest.raises(ConfigError, match="no config key 'retrieval.k_retreive'"):
        _ = CFG.retrieval.k_retreive


def test_sections_are_read_only():
    with pytest.raises(ConfigError):
        CFG.retrieval.k_final = 99


def test_paths_are_absolute_and_match_the_data_schema():
    paths = CFG.paths
    assert paths.manifest == REPO_ROOT / "data/index/manifest.json"
    assert paths.chunks == REPO_ROOT / "data/index/chunks.jsonl"
    assert paths.embeddings == REPO_ROOT / "data/index/embeddings.npy"
    assert paths.faiss_index == REPO_ROOT / "data/index/faiss.index"
    assert paths.embedding_cache == REPO_ROOT / "data/cache/embeddings.json"
    assert paths.descriptions == REPO_ROOT / "data/cache/descriptions.json"
    assert paths.arxiv_queries == REPO_ROOT / "data/cache/arxiv_queries.json"
    assert paths.clusters == REPO_ROOT / "data/cache/clusters.json"
    assert all(path.is_absolute() for path in paths.directories())


def test_embedding_cache_is_separate_from_the_index_vectors():
    """data/cache/embeddings.json is a content-hash cache; embeddings.npy is the index."""
    assert CFG.paths.embedding_cache != CFG.paths.embeddings
    assert CFG.paths.embedding_cache.parent == CFG.paths.cache
    assert CFG.paths.embeddings.parent == CFG.paths.index


def test_derived_paths():
    assert CFG.paths.paper_pdf("2103_14030v2").name == "2103_14030v2.pdf"
    assert CFG.paths.figure_image("p1", "p1__f03").parent.name == "p1"
    assert CFG.paths.trace_file("run-1").parent == CFG.paths.traces


def test_ensure_is_idempotent():
    CFG.paths.ensure()
    CFG.paths.ensure()
    assert all(path.is_dir() for path in CFG.paths.directories())


def test_model_roles_are_configured():
    for role in ("agent", "vision", "utility"):
        assert CFG.model_for(role)
    with pytest.raises(ConfigError, match="unknown llm role"):
        CFG.model_for("nonsense")


def test_system_prompt_loads_from_a_file():
    assert "retrieve_evidence" in CFG.prompt("system")
    with pytest.raises(ConfigError, match="prompt not found"):
        CFG.prompt("does_not_exist")


def test_missing_section_is_rejected(tmp_path: Path):
    bad = tmp_path / "config.yaml"
    bad.write_text("llm:\n  provider: openrouter\n")
    with pytest.raises(ConfigError, match="missing sections"):
        load_config(bad)


def test_every_section_in_config_yaml_is_validated():
    """Every section the shipped config defines must be required by the loader.

    `compute` was missing from REQUIRED_SECTIONS, so a config without it loaded cleanly
    and then died inside resolve_device() with an AttributeError that never named the
    file. This compares against config.yaml rather than iterating REQUIRED_SECTIONS —
    iterating the tuple cannot detect a section absent from it, which is exactly the
    bug. A new section added to config.yaml and read by the code fails here until it is
    validated too.
    """
    import yaml

    from config import REPO_ROOT, REQUIRED_SECTIONS

    shipped = set(yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8")))
    unvalidated = shipped - set(REQUIRED_SECTIONS)
    assert not unvalidated, (
        f"config.yaml defines {sorted(unvalidated)} but the loader does not require "
        "them; a config omitting one would fail late instead of at load"
    )


def test_a_config_missing_compute_is_rejected_at_load(tmp_path: Path):
    """The specific regression: no compute section must fail at load, naming the file."""
    import yaml

    from config import REPO_ROOT

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    del raw["compute"]
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing sections: compute"):
        load_config(bad)


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="config file not found"):
        Config(tmp_path / "nope.yaml")
