"""
Protocol tests for the JSON-lines bridge worker, driven in-process through StringIO.

In:  request lines fed to bridge.worker.serve, with throwaway ops registered for the test.
Out: assertions on the frames written back. No models, index, or network required.
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bridge.ops.core  # noqa: E402,F401  (registers ping and the other core ops)
from bridge.registry import OPS  # noqa: E402
from bridge.worker import serve  # noqa: E402


def run(*requests: str) -> list:
    stdin = io.StringIO("".join(f"{line}\n" for line in requests))
    out = io.StringIO()
    serve(stdin, out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_ready_frame_comes_first_and_lists_the_ops():
    frames = run()
    assert frames == [{"ready": True, "ops": sorted(OPS)}]
    assert {"ping", "ask", "retrieve_evidence", "analyze_corpus", "explore_graph"} <= set(OPS)


def test_ping_answers_with_its_id():
    frames = run(json.dumps({"id": 7, "op": "ping", "params": {}}))
    assert frames[1]["id"] == 7
    assert frames[1]["result"]["ok"] is True


def test_unknown_op_is_an_error_frame_and_the_worker_carries_on():
    frames = run(
        json.dumps({"id": 1, "op": "no_such_op"}),
        "",
        json.dumps({"id": 2, "op": "ping"}),
    )
    assert frames[1] == {"id": 1, "error": "unknown_op", "detail": "no bridge op 'no_such_op'"}
    assert frames[2]["id"] == 2 and frames[2]["result"]["ok"] is True


def test_invalid_json_is_reported_without_an_id():
    frames = run("{not json", json.dumps({"id": 3, "op": "ping"}))
    assert frames[1]["id"] is None
    assert frames[1]["error"] == "bad_request"
    assert frames[2]["id"] == 3


def test_events_stream_before_the_result(monkeypatch):
    def streaming(params, emit):
        for i in range(params["n"]):
            emit({"kind": "tick", "i": i})
        return {"done": True, "path": Path("/x/y")}

    monkeypatch.setitem(OPS, "_test_stream", streaming)
    frames = run(json.dumps({"id": 4, "op": "_test_stream", "params": {"n": 3}}))
    assert frames[1:] == [
        {"id": 4, "event": {"kind": "tick", "i": 0}},
        {"id": 4, "event": {"kind": "tick", "i": 1}},
        {"id": 4, "event": {"kind": "tick", "i": 2}},
        # A Path is not JSON; the worker falls back to str() rather than failing the frame.
        {"id": 4, "result": {"done": True, "path": "/x/y"}},
    ]


def test_a_crashing_op_becomes_op_crashed_and_the_worker_survives(monkeypatch, capsys):
    def crashing(params, emit):
        raise ValueError("kaboom")

    monkeypatch.setitem(OPS, "_test_crash", crashing)
    frames = run(
        json.dumps({"id": 5, "op": "_test_crash"}),
        json.dumps({"id": 6, "op": "ping"}),
    )
    assert frames[1] == {"id": 5, "error": "op_crashed", "detail": "ValueError: kaboom"}
    assert frames[2]["id"] == 6 and frames[2]["result"]["ok"] is True
    assert "kaboom" in capsys.readouterr().err
