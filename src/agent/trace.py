"""
Tool-call trace records, one JSON line per call, per docs/DATA_SCHEMA.md section 7.

In:  a run_id, the tool name and args as the model supplied them, and the tool's result.
Out: appended records in logs/traces/{run_id}.jsonl. `chunk_ids_returned` is lifted to a
     top-level field so the eval harness reads recall without parsing result_summary.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from common.records import utc_now
from common.storage import append_jsonl
from config import CFG, Config


def new_run_id() -> str:
    """A short identifier for one question's run, used as the trace filename."""
    return uuid.uuid4().hex[:12]


def summarise_result(result: Dict[str, Any], text_chars: int) -> Dict[str, Any]:
    """A truncated view of a tool result, for the trace.

    Full chunk text in every record makes the log unusable, which is the whole reason
    docs/DATA_SCHEMA.md calls this a summary rather than the payload.
    """
    if not isinstance(result, dict):
        return {"non_dict_result": type(result).__name__}

    summary: Dict[str, Any] = {}
    for key, value in result.items():
        if key == "chunks" and isinstance(value, list):
            summary["chunks"] = [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "paper_id": chunk.get("paper_id"),
                    "score": chunk.get("score"),
                    "chunk_type": chunk.get("chunk_type"),
                    "text": str(chunk.get("text", ""))[:text_chars],
                }
                for chunk in value
                if isinstance(chunk, dict)
            ]
        elif isinstance(value, str):
            summary[key] = value[:text_chars]
        else:
            summary[key] = value
    return summary


def chunk_ids_of(result: Dict[str, Any]) -> List[str]:
    """The chunk_ids a tool returned, for retrieval recall scoring in M8."""
    if not isinstance(result, dict):
        return []
    chunks = result.get("chunks")
    if not isinstance(chunks, list):
        return []
    return [str(chunk.get("chunk_id")) for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("chunk_id")]


class TraceWriter:
    """Appends trace records for one run. Never raises — a broken log must not stop a run."""

    def __init__(self, run_id: Optional[str] = None, question_id: Optional[str] = None,
                 config: Config = CFG) -> None:
        self.run_id = run_id or new_run_id()
        self.question_id = question_id
        self.config = config
        self.path = config.paths.trace_file(self.run_id)
        self.records: List[Dict[str, Any]] = []

    def record(
        self,
        iteration: int,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        latency_ms: int,
    ) -> Dict[str, Any]:
        """Write one record. Field-for-field docs/DATA_SCHEMA.md section 7."""
        entry = {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "iteration": int(iteration),
            "tool_name": tool_name,
            "args": args,
            "result_summary": summarise_result(result, int(self.config.agent.trace_text_chars)),
            "chunk_ids_returned": chunk_ids_of(result),
            "error": result.get("error") if isinstance(result, dict) else None,
            "latency_ms": int(latency_ms),
            "timestamp": utc_now(),
        }
        self.records.append(entry)
        try:
            append_jsonl(self.path, [entry])
        except OSError:
            # Losing a trace line is bad for debugging but must not end the run.
            pass
        return entry
