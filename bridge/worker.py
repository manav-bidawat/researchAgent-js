"""
Long-lived JSON-lines RPC worker: the one door from the TypeScript layer into src/.

In:  one JSON request per line on stdin: {"id", "op", "params"}.
Out: JSON lines on the original stdout: {"id", "event"} while an op streams, then
     {"id", "result"} or {"id", "error", "detail"}. Lives outside src/ and imports it,
     as eval/ does; nothing in src/ knows this exists.
"""

from __future__ import annotations

import importlib
import json
import os
import pkgutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, TextIO

ROOT = Path(__file__).resolve().parent.parent


def _claim_stdout() -> TextIO:
    """Keep the real stdout for the protocol and point everything else at stderr.

    The engine prints progress in places; one stray print on the protocol channel would
    corrupt a frame. Duplicating fd 1 first, then aiming fd 1 at stderr, catches prints
    from Python and from native libraries alike.
    """
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


def _load_ops() -> None:
    import bridge.ops as package

    for module in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"bridge.ops.{module.name}")


def _jsonable(value: Any) -> Any:
    """Round-trip through json with a str() fallback: Paths and numpy scalars appear."""
    return json.loads(json.dumps(value, default=_fallback, allow_nan=True))


def _fallback(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def serve(stdin: TextIO, protocol: TextIO) -> None:
    from bridge.registry import OPS

    def send(frame: Dict[str, Any]) -> None:
        protocol.write(json.dumps(_jsonable(frame)) + "\n")
        protocol.flush()

    send({"ready": True, "ops": sorted(OPS)})
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            send({"id": None, "error": "bad_request", "detail": f"invalid JSON: {exc}"})
            continue

        request_id = request.get("id")
        name = request.get("op")
        fn = OPS.get(name)
        if fn is None:
            send({"id": request_id, "error": "unknown_op", "detail": f"no bridge op {name!r}"})
            continue

        def emit(event: Dict[str, Any], _id: Any = request_id) -> None:
            try:
                send({"id": _id, "event": event})
            except Exception:  # an observer must never end the op it is watching
                pass

        try:
            result = fn(request.get("params") or {}, emit)
            send({"id": request_id, "result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({"id": request_id, "error": "op_crashed", "detail": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    protocol = _claim_stdout()
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    _load_ops()
    serve(sys.stdin, protocol)


if __name__ == "__main__":
    main()
