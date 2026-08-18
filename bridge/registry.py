"""
The table of operations the TypeScript side may invoke on the Python engine.

In:  op functions registered with @op("name"); each takes (params, emit) where emit
     streams an event dict back to the caller before the result is ready.
Out: OPS, a name -> function map the worker dispatches from. A fixed table rather than
     a generic "call any function" hook, so the engine's surface stays reviewable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

Emit = Callable[[Dict[str, Any]], None]
OpFn = Callable[[Dict[str, Any], Emit], Any]

OPS: Dict[str, OpFn] = {}


def op(name: str) -> Callable[[OpFn], OpFn]:
    """Register `fn` under `name`. A duplicate name is a programming error."""

    def register(fn: OpFn) -> OpFn:
        if name in OPS:
            raise RuntimeError(f"bridge op registered twice: {name}")
        OPS[name] = fn
        return fn

    return register
