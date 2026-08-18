"""
Bridge ops the TypeScript eval harness needs: batch retrieval under config overrides, and
passage embeddings for fact coverage.

In:  questions plus retrieval-section overrides; or a list of texts to embed.
Out: raw retrieved ids, gate decisions and latencies (scored in TS), or vectors.
     Models are shared across override configs so a sweep loads them once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from bridge import runtime
from bridge.registry import Emit, op

ROOT = Path(__file__).resolve().parents[2]


def config_with(overrides: Dict[str, Any]) -> Any:
    """A Config identical to the project's but with retrieval settings overridden.

    Written through the YAML rather than poked into the loaded object: config sections
    are read-only on purpose, and a sweep that mutates them in place would be testing a
    state the real system can never be in.
    """
    import config as config_module

    cfg = runtime.config()
    if not overrides:
        return cfg
    raw = yaml.safe_load(Path(config_module.config_path()).read_text(encoding="utf-8"))
    raw["retrieval"].update(overrides)
    scratch = ROOT / "eval" / "results" / "_sweep_config.yaml"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_module.load_config(scratch)


@op("retrieve_batch")
def retrieve_batch(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Run retrieve_evidence per question with a fresh dedup state; no agent, no LLM."""
    from tools.retrieve_evidence import EvidenceRetriever

    config = config_with(params.get("overrides") or {})
    retriever = EvidenceRetriever(config, embedder=runtime.embedder(), reranker=runtime.reranker())
    k = int(params.get("k") or 5)

    rows: List[Dict[str, Any]] = []
    for question in params.get("questions") or []:
        retriever.reset()
        started = time.perf_counter()
        result = retriever.retrieve(question["question"], k=k)
        latency_ms = int((time.perf_counter() - started) * 1000)
        chunks = result.get("chunks", []) if "error" not in result else []
        rows.append({
            "question_id": question.get("question_id"),
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "paper_ids": [c["paper_id"] for c in chunks],
            "scores": [c.get("score") for c in chunks],
            "sufficient_evidence": bool(result.get("sufficient_evidence", False)),
            "error": result.get("error"),
            "latency_ms": latency_ms,
        })
    return {"rows": rows}


@op("embed_passages")
def embed_passages(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """Passage embeddings (no query instruction), normalised as the embedder returns them."""
    texts = [str(t) for t in params.get("texts") or []]
    if not texts:
        return {"vectors": []}
    vectors = runtime.embedder().encode_passages(texts)
    return {"vectors": json.loads(json.dumps(vectors.tolist()))}


@op("retrieval_config")
def retrieval_config(params: Dict[str, Any], emit: Emit) -> Dict[str, Any]:
    """The live retrieval and eval sections, so TS sweeps start from the tuned values."""
    raw = yaml.safe_load(Path(__import__("config").config_path()).read_text(encoding="utf-8"))
    return {"retrieval": raw.get("retrieval", {}), "eval": raw.get("eval", {})}
