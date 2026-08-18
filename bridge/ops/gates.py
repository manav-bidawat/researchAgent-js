"""
Bridge ops for the milestone gates and live checks in scripts/: narrow probes of engine
internals (chunk store, manifest, vector index, caches, device, LLM client).

In:  params dicts from scripts/*.ts. Every op is named for the one thing it measures.
Out: JSON-serialisable dicts. Heavy engine modules are imported inside each op, so the
     worker still starts (and lists these ops) on a machine without the ML stack.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge import runtime
from bridge.registry import Emit, op

_CACHE: Dict[str, Any] = {}


def _kwargs(params: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """Only the params the caller actually sent, so engine defaults stay in charge."""
    return {name: params[name] for name in names if params.get(name) is not None}


# ---- configuration -----------------------------------------------------------------


@op("gate_config")
def gate_config(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """The config values and artefact paths the gates print or compare against."""
    cfg = runtime.config()
    paths = cfg.paths
    return {
        "source_path": str(cfg.source_path),
        "llm": {
            "agent_model": cfg.llm.agent_model,
            "vision_model": cfg.llm.vision_model,
            "utility_model": cfg.llm.utility_model,
        },
        "compute_device": str(cfg.compute.device),
        "embedding": {
            "model": cfg.embedding.model,
            "dim": int(cfg.embedding.dim),
            "batch_size": int(cfg.embedding.batch_size),
        },
        "chunking": {"max_tokens": int(cfg.chunking.max_tokens)},
        "retrieval": {
            "relevance_threshold": float(cfg.retrieval.relevance_threshold),
            "bi_encoder_relevance_threshold": cfg.retrieval.get("bi_encoder_relevance_threshold"),
            "k_retrieve": int(cfg.retrieval.k_retrieve),
            "max_chunk_chars": int(cfg.retrieval.max_chunk_chars),
            "reranker_model": str(cfg.retrieval.reranker_model),
        },
        "agent": {
            "max_iterations": int(cfg.agent.max_iterations),
            "trace_text_chars": int(cfg.agent.trace_text_chars),
        },
        "nli": {"model": str(cfg.nli.model), "max_pairs": int(cfg.nli.max_pairs)},
        "paths": {
            "papers": str(paths.papers),
            "index": str(paths.index),
            "embeddings": str(paths.embeddings),
            "faiss_index": str(paths.faiss_index),
            "descriptions": str(paths.descriptions),
            "directories": [str(p) for p in paths.directories()],
        },
    }


@op("write_scratch_config")
def write_scratch_config(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """A copy of the live config.yaml with some sections updated or replaced, written to `out`.

    Written through the YAML rather than poked into a loaded Config: config sections are
    read-only by design, and a scratch run should exercise the same load path the CLI uses.
    """
    import yaml

    import config as config_module

    raw = yaml.safe_load(Path(config_module.config_path()).read_text(encoding="utf-8"))
    for section, values in (params.get("section_updates") or {}).items():
        raw.setdefault(section, {}).update(values)
    for section, values in (params.get("section_replacements") or {}).items():
        raw[section] = values
    out = Path(params["out"]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return {"path": str(out)}


# ---- LLM client (M0) ---------------------------------------------------------------


@op("llm_readiness")
def llm_readiness(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Offline M0 checks: the client imports, directories exist, the prompt loads, a key is set."""
    cfg = runtime.config()
    out: Dict[str, Any] = {}
    try:
        import llm_client

        out["imports_ok"] = bool(llm_client.LLMClient)
        out["import_error"] = None
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        out["imports_ok"] = False
        out["import_error"] = f"{type(exc).__name__}: {exc}"

    out["missing_directories"] = [str(p) for p in cfg.paths.directories() if not p.is_dir()]
    try:
        out["system_prompt_mentions_retrieve_evidence"] = "retrieve_evidence" in cfg.prompt("system")
        out["system_prompt_error"] = None
    except Exception as exc:  # noqa: BLE001
        out["system_prompt_mentions_retrieve_evidence"] = False
        out["system_prompt_error"] = str(exc)

    from config import ConfigError

    try:
        cfg.require_api_key()
        out["api_key_error"] = None
    except ConfigError as exc:
        out["api_key_error"] = str(exc)
    return out


@op("llm_probe")
def llm_probe(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """One live completion of a single user message, optionally offering tools."""
    from llm_client import LLMClient, LLMError

    if "llm" not in _CACHE:
        _CACHE["llm"] = LLMClient()
    client = _CACHE["llm"]
    messages = [{"role": "user", "content": str(params.get("content", ""))}]
    try:
        if params.get("tools"):
            response = client.complete(messages, tools=params["tools"], role=params.get("role", "agent"))
        else:
            response = client.complete(messages, role=params.get("role", "utility"))
    except LLMError as exc:
        return {"error": exc.code, "detail": exc.detail}
    return {
        "text": response.text,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments, "ok": call.ok}
            for call in response.tool_calls
        ],
        "assistant_has_tool_calls": bool(response.assistant_message().get("tool_calls")),
    }


# ---- device (check_gpu) ------------------------------------------------------------


@op("device_report")
def device_report(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """What torch can see, why it cannot see a GPU when it cannot, and what device resolved."""
    cfg = runtime.config()
    report: Dict[str, Any] = {"compute_device": str(cfg.compute.device)}
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 — torch is a hard dependency; report, do not die
        report["torch_error"] = str(exc)
    else:
        report["torch_version"] = torch.__version__
        report["built_for_cuda"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        devices: List[Dict[str, Any]] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": i, "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024 ** 3, 1),
                    "sm": f"{props.major}{props.minor}",
                })
        else:
            # The common case on a fresh machine: CUDA wheel, no host driver.
            report["reason"] = (
                "no NVIDIA kernel driver loaded" if not Path("/proc/driver/nvidia").exists()
                else "driver present but torch cannot use it (version mismatch?)"
            )
        report["devices"] = devices

    from common.device import resolve_device
    from config import ConfigError

    try:
        report["resolved_device"] = resolve_device(cfg)
    except ConfigError as exc:
        report["resolved_device"] = None
        report["resolve_error"] = str(exc)
    return report


def _corpus_texts(cfg: Any, fallback_text: str, fallback_count: int) -> Dict[str, Any]:
    """Real chunk text, because synthetic text makes the GPU look better than it is.

    sentence-transformers sorts a batch by length before padding it, so 406 copies of one
    short sentence pack into ~30-token batches while the real corpus averages 278 tokens.
    Benchmarking the former measured about a ninth of the actual work and reported a 63x
    speedup where the true figure on this corpus is 7x. Falls back to synthetic text of a
    representative length only when no corpus has been indexed yet.
    """
    try:
        from corpus.chunk_store import ChunkStore

        texts = [chunk["text"] for chunk in ChunkStore(config=cfg).all()]
        if texts:
            return {"texts": texts, "source": "corpus"}
    except Exception:  # noqa: BLE001
        pass
    return {"texts": [fallback_text] * fallback_count, "source": "synthetic"}


@op("benchmark_models")
def benchmark_models(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Time the bi-encoder, the reranker and the NLI model on `device`. Streams a line each."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    cfg = runtime.config()
    device = str(params["device"])
    corpus = _corpus_texts(cfg, str(params.get("fallback_text", "")), int(params.get("fallback_count", 406)))
    texts = corpus["texts"]
    out: Dict[str, Any] = {"texts_source": corpus["source"], "n_texts": len(texts)}

    t = time.perf_counter()
    bi = SentenceTransformer(str(cfg.embedding.model), device=device)
    out["bi_encoder_load_s"] = time.perf_counter() - t
    emit({"kind": "timing", "name": "bi_encoder_load_s", "value": out["bi_encoder_load_s"]})
    t = time.perf_counter()
    bi.encode(texts, batch_size=int(cfg.embedding.batch_size), convert_to_numpy=True,
              normalize_embeddings=True, show_progress_bar=False)
    out["embed_s"] = time.perf_counter() - t
    emit({"kind": "timing", "name": "embed_s", "value": out["embed_s"]})

    ce = CrossEncoder(str(cfg.retrieval.reranker_model), device=device)
    k = int(cfg.retrieval.k_retrieve)
    # Distinct chunks, for the same reason _corpus_texts exists: identical pairs pad into
    # one uniform batch and flatter the GPU.
    pool = (texts * (k // max(1, len(texts)) + 1))[:k]
    query = str(params.get("query", ""))
    pairs = [(query, text[:2000]) for text in pool]
    ce.predict(pairs)  # warm up; the first pass pays for kernel compilation
    t = time.perf_counter()
    ce.predict(pairs)
    out["rerank_s"] = time.perf_counter() - t
    out["n_rerank_pairs"] = len(pairs)
    emit({"kind": "timing", "name": "rerank_s", "value": out["rerank_s"]})

    nli = CrossEncoder(str(cfg.nli.model), device=device)
    n = int(cfg.nli.max_pairs)
    sentences = [s for text in texts[:20] for s in text.split(". ")][: n + 1] or [texts[0]] * (n + 1)
    nli_pairs = [(sentences[i][:2000], sentences[i + 1][:2000])
                 for i in range(min(n, len(sentences) - 1))]
    nli.predict(nli_pairs)
    t = time.perf_counter()
    nli.predict(nli_pairs)
    out["nli_s"] = time.perf_counter() - t
    out["n_nli_pairs"] = len(nli_pairs)
    return out


# ---- collection (M1, M3, check_arxiv) ----------------------------------------------


@op("collect_papers")
def collect_papers(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """search_and_fetch only: plan, search, dedup, download. No ingest, no index."""
    from corpus.collect import search_and_fetch

    return search_and_fetch(
        params.get("topic", ""), config=runtime.config(),
        **_kwargs(params, "topic_tag", "max_results", "categories"),
    )


@op("collect_dead_network_probe")
def collect_dead_network_probe(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """search_and_fetch with an arXiv client whose every call raises ConnectionError.

    Reports whether it raised instead of returning an error dict; the patch is undone in
    `finally`, so later ops in this worker see the real client.
    """
    from corpus import collect

    class DeadClient:
        def results(self, search: Any) -> Any:
            raise ConnectionError("network is unreachable")

    original = collect._build_client
    collect._build_client = lambda config: DeadClient()
    try:
        result = collect.search_and_fetch(
            params.get("topic", "anything at all"), config=runtime.config(),
            **_kwargs(params, "topic_tag"),
        )
    except Exception as exc:  # noqa: BLE001 — the point of the check
        return {"raised": f"{type(exc).__name__}: {exc}", "result": None}
    finally:
        collect._build_client = original
    return {"raised": None, "result": result}


@op("append_stand_in_chunks")
def append_stand_in_chunks(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Append `n` placeholder chunk records for one paper, so tag backfill can be exercised
    before real chunking exists."""
    from common.records import chunk_record
    from corpus.chunk_store import ChunkStore

    cfg = runtime.config()
    paper_id = str(params["paper_id"])
    tag = str(params["topic_tag"])
    n = int(params.get("n", 3))
    written = ChunkStore(config=cfg).append([
        chunk_record(paper_id=paper_id, paper_title="stand-in", topic_tags=[tag],
                     chunk_type="text", text=f"stand-in chunk {i}", page=1, position=i, n_tokens=4)
        for i in range(n)
    ])
    return {"appended": written}


# ---- ingestion (M2, M3) ------------------------------------------------------------


@op("ingest_all")
def ingest_all_op(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Ingest every manifest paper that has no chunks yet."""
    from extraction.ingest import ingest_all

    return ingest_all(config=runtime.config(), describe=bool(params.get("describe", True)))


@op("ingest_paper")
def ingest_paper_op(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Ingest one paper by id."""
    from extraction.ingest import ingest_paper

    return ingest_paper(str(params["paper_id"]), config=runtime.config(),
                        describe=bool(params.get("describe", True)))


@op("description_cache_size")
def description_cache_size(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """How many figure descriptions the content-hash cache holds."""
    from extraction.describe import DescriptionCache

    return {"entries": len(DescriptionCache(runtime.config()))}


# ---- indexing (M3, build_corpus) ---------------------------------------------------


@op("index_chunks")
def index_chunks_op(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Embed and index every chunk not yet in the index; the retriever then reloads it."""
    from retrieval.indexer import index_chunks

    result = index_chunks(config=runtime.config(), embedder=runtime.embedder(),
                          rebuild=bool(params.get("rebuild")))
    runtime.invalidate_index()
    return result


@op("vector_index_consistency")
def vector_index_consistency(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """The sizes of faiss.index, embeddings.npy, faiss_id_map and chunks.jsonl, side by side.

    FAISS ids are positional, so any disagreement means lookups silently resolve to the
    wrong chunk.
    """
    import numpy as np

    from corpus.chunk_store import ChunkStore
    from corpus.manifest import Manifest
    from retrieval.vector_index import VectorIndex

    cfg = runtime.config()
    manifest = Manifest.load(cfg)
    index = VectorIndex.load(manifest, cfg)
    stored = np.load(cfg.paths.embeddings)
    return {
        "index_ntotal": int(index.index.ntotal),
        "faiss_id_map": len(manifest.data["faiss_id_map"]),
        "embeddings_rows": int(stored.shape[0]),
        "chunks": ChunkStore(config=cfg).count(),
        "embedding_model": manifest.data.get("embedding_model"),
        "embedding_dim": manifest.data.get("embedding_dim"),
    }


@op("bi_encoder_search")
def bi_encoder_search(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Stage-one search straight off the persisted index, text trimmed for printing."""
    from retrieval.indexer import search

    hits = search(str(params.get("query", "")), k=int(params.get("k", 5)),
                  config=runtime.config(), embedder=runtime.embedder())
    return {"hits": [
        {"chunk_id": h["chunk_id"], "paper_id": h["paper_id"], "score": round(float(h["score"]), 4),
         "chunk_type": h["chunk_type"], "text": " ".join(h["text"].split())[:150]}
        for h in hits
    ]}


@op("embedding_model_mismatch_probe")
def embedding_model_mismatch_probe(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Record a different embedding model in the manifest, ask model_changed, then restore.

    Done inside one op so the manifest is restored in `finally` by the same process that
    changed it.
    """
    from corpus.manifest import Manifest
    from retrieval.indexer import model_changed

    cfg = runtime.config()
    live = Manifest.load(cfg)
    original = live.data["embedding_model"]
    live.data["embedding_model"] = str(params.get("fake_model", "some/other-model"))
    live.save()
    try:
        return {"detected": bool(model_changed(live, cfg))}
    finally:
        live.data["embedding_model"] = original
        live.save()


# ---- retrieval and tools (M4, M6) --------------------------------------------------


class _CountingReranker:
    """Wraps the shared reranker and records how many candidates each rerank call saw."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.seen: List[int] = []

    def rerank(self, query: Any, candidates: Any) -> Any:
        self.seen.append(len(candidates))
        return self._inner.rerank(query, candidates)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@op("retrieve_with_rerank_count")
def retrieve_with_rerank_count(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """retrieve_evidence on a gate-owned retriever whose reranker counts its pairs.

    Counting makes the shortlist claim measured, not asserted. The retriever persists
    across calls so cross-call dedup can be tested; `reset` clears it first.
    """
    from tools.retrieve_evidence import EvidenceRetriever

    if "counted_retriever" not in _CACHE:
        counter = _CountingReranker(runtime.reranker())
        _CACHE["rerank_counter"] = counter
        _CACHE["counted_retriever"] = EvidenceRetriever(
            runtime.config(), embedder=runtime.embedder(), reranker=counter,
        )
    retriever = _CACHE["counted_retriever"]
    counter = _CACHE["rerank_counter"]
    if params.get("reset"):
        retriever.reset()
    before = len(counter.seen)
    result = retriever.retrieve(
        params.get("query", ""), **_kwargs(params, "k", "topic_filter", "chunk_types"),
    )
    return {"result": result, "rerank_pairs": counter.seen[before:]}


@op("inspect_figure_probe")
def inspect_figure_probe(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """inspect_figure standalone, plus whether any value was raw bytes (which JSON would hide)."""
    from tools.inspect_figure import FigureInspector

    result = FigureInspector(runtime.config()).inspect(
        **_kwargs(params, "paper_id", "figure_id", "chunk_id", "user_image", "question"),
    )
    return {
        "result": result,
        "contains_bytes": any(isinstance(v, (bytes, bytearray)) for v in result.values()),
    }


@op("check_evidence_consistency")
def check_evidence_consistency(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """check_evidence_consistency standalone; the NLI model loads once per worker."""
    from tools.check_evidence_consistency import ConsistencyChecker

    if "checker" not in _CACHE:
        _CACHE["checker"] = ConsistencyChecker(runtime.config(), embedder=runtime.embedder())
    return _CACHE["checker"].check(
        params.get("mode", ""), params.get("chunk_ids") or [], answer_text=params.get("answer_text"),
    )


@op("registered_tools")
def registered_tools(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Names in the full tool registry the agent is built with."""
    from agent.tool_registry import build_full_registry

    registry = build_full_registry(retriever=runtime.retriever(), config=runtime.config())
    return {"names": list(registry.names)}


@op("agent_run_with_messages")
def agent_run_with_messages(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """One agent run that also returns the final message array, so the re-send is visible.

    `registry` picks "default" (retrieve_evidence only) or "full" (every tool).
    """
    from agent.loop import AgentLoop
    from agent.tool_registry import build_default_registry, build_full_registry
    from bridge.ops.core import ANSWER_KEYS

    cfg = runtime.config()
    retriever = runtime.retriever()
    if params.get("registry", "default") == "full":
        registry = build_full_registry(retriever=retriever, config=cfg)
    else:
        registry = build_default_registry(retriever, cfg)
    loop = AgentLoop(registry, config=cfg, retriever=retriever)
    result = loop.run(str(params.get("question", "")), question_id=params.get("question_id"))
    if "error" in result:
        return {key: value for key, value in result.items() if key != "conversation"}
    out: Dict[str, Any] = {key: result[key] for key in ANSWER_KEYS if key in result}
    conversation: Optional[Any] = result.get("conversation")
    out["messages"] = conversation.messages() if conversation is not None else []
    return out
