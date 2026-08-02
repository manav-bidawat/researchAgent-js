"""
Crash-safe file primitives shared by the manifest and the chunk store.

In:  a target path plus the content to write (a JSON-serialisable object, or lines).
Out: the file replaced atomically — written to a temp file in the same directory and
     renamed over the target, so an interrupted write can never leave a partial file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def write_text_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` atomically. Creates parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)  # atomic within a filesystem
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Any, indent: int = 2) -> None:
    """Replace `path` with `payload` serialised as JSON, atomically."""
    write_text_atomic(path, json.dumps(payload, indent=indent, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    """Parse `path` as JSON. Returns `default` when the file is absent or unreadable.

    A corrupt cache should degrade to a cache miss, not crash a tool.
    """
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def write_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Replace `path` with one JSON object per line, atomically. Returns the count.

    This is the rewrite path — used when existing records must be modified in place,
    such as backfilling a new topic tag onto every chunk of a paper.
    """
    lines: List[str] = [json.dumps(record, ensure_ascii=False) for record in records]
    write_text_atomic(path, "".join(line + "\n" for line in lines))
    return len(lines)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Append records to a JSONL file, creating it if absent. Returns the count.

    Appends are not atomic in the way a rewrite is; a crash mid-append can leave a
    truncated final line, which `read_jsonl` skips rather than raising on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        stream.flush()
        os.fsync(stream.fileno())
    return written


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield each JSON object in a JSONL file. Malformed lines are skipped, not raised.

    A truncated trailing line from an interrupted append is recoverable data loss of
    one record; refusing to read the other thousand would not be.
    """
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record
