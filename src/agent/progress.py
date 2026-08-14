"""
Live console progress for one agent run: what the loop is doing, while it does it.

In:  event dicts emitted by AgentLoop.on_event — kind, iteration, tool name, args, result.
Out: one formatted line per event on a stream (stderr by default), so stdout stays the
     answer alone. A loop constructed without a reporter emits nothing at all, which is
     why the eval harness and the gate scripts are unaffected by this module existing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from config import CFG, Config


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(0, width - 1)] + "…"


def format_args(args: Dict[str, Any], width: int) -> str:
    """The call's arguments as the model supplied them, short enough for one line."""
    if not isinstance(args, dict) or not args:
        return ""
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return _truncate(rendered, width)


def format_result(result: Dict[str, Any], width: int) -> str:
    """One phrase saying what a tool came back with.

    Each tool returns a different dict (docs/TOOLS.md), so this reads whichever count
    field the tool actually has rather than assuming a common shape. Anything
    unrecognised falls back to the key names, which is still more informative than
    printing nothing.
    """
    if not isinstance(result, dict):
        return type(result).__name__
    if result.get("error"):
        return _truncate(f"error: {result['error']}", width)
    if isinstance(result.get("chunks"), list):
        note = _plural(len(result["chunks"]), "chunk")
        if result.get("sufficient_evidence") is False:
            note += ", insufficient"
        return note
    if isinstance(result.get("papers_added"), list):
        return (f"{_plural(len(result['papers_added']), 'paper')} added, "
                f"{result.get('chunks_added', 0)} chunks")
    if isinstance(result.get("claims"), list):
        return f"{_plural(len(result['claims']), 'claim')}, grounded {result.get('grounded_ratio')}"
    if "found" in result:
        scored = _plural(int(result.get("n_pairs_scored") or 0), "pair")
        return f"{scored} scored, " + ("conflict found" if result["found"] else "no conflict")
    if result.get("operation"):
        return _truncate(str(result.get("summary") or result["operation"]), width)
    if result.get("image_path"):
        return Path(str(result["image_path"])).name
    return _truncate(", ".join(sorted(result)), width)


class ConsoleReporter:
    """Prints one line per loop event. Stateful: a tool's line is opened, then closed.

    `tool_start` writes a line with no newline and flushes, so a slow tool shows what it
    is doing while it runs; the matching `tool_end` finishes that same line with the
    result. Every other event kind writes a whole line of its own — that invariant is
    what keeps the output from tangling when a call is skipped before it dispatches.
    """

    def __init__(self, stream: Optional[TextIO] = None, config: Config = CFG) -> None:
        import sys

        self.stream = stream if stream is not None else sys.stderr
        self.width = int(config.agent.progress_text_chars)
        self._line_open = False

    def _write(self, text: str, end: str = "\n") -> None:
        self.stream.write(text + end)
        self.stream.flush()

    def _close_line(self) -> None:
        """Terminate a `tool_start` line that never got its `tool_end`."""
        if self._line_open:
            self._write("")
            self._line_open = False

    def __call__(self, event: Dict[str, Any]) -> None:
        kind = event.get("kind")
        iteration = event.get("iteration", 0) + 1
        tag = f"[{iteration}]"

        if kind == "thinking":
            self._close_line()
            self._write(f"{tag} thinking...")
        elif kind == "tool_start":
            self._close_line()
            self._write(
                f"{tag} → {event['tool_name']}({format_args(event.get('args') or {}, self.width)})"
                " ... ",
                end="",
            )
            self._line_open = True
        elif kind == "tool_end":
            note = format_result(event.get("result") or {}, self.width)
            self._write(f"{note}, {event.get('latency_ms', 0)}ms" if self._line_open
                        else f"{tag} → {event['tool_name']} ... {note}")
            self._line_open = False
        elif kind == "skipped":
            self._close_line()
            self._write(f"{tag} → {event.get('tool_name') or 'call'} skipped: {event.get('reason')}")
        elif kind == "image":
            self._close_line()
            self._write(f"{tag} ↳ attaching {Path(str(event.get('image_path'))).name}")
        elif kind == "answering":
            self._close_line()
            self._write(f"{tag} answering")
        elif kind == "cap":
            self._close_line()
            self._write(f"{tag} iteration cap reached — forcing an answer from what is held")
