"""
Smoke tests for compute.device resolution and its wiring into the three local models.

In:  the repo's real config.yaml, plus small in-memory config stand-ins.
Out: assertions only. No network, no model download, no GPU required — the CUDA probe
     is monkeypatched so both branches are exercised on any machine.
"""

from __future__ import annotations

import pytest

from common import device as device_mod
from config import CFG, ConfigError


class _Compute:
    def __init__(self, value: str) -> None:
        self.device = value


class _StubConfig:
    """Just enough config surface for resolve_device."""

    def __init__(self, value: str) -> None:
        self.compute = _Compute(value)


@pytest.fixture(autouse=True)
def _clear_device_cache():
    device_mod.reset_cache()
    yield
    device_mod.reset_cache()


def test_config_declares_a_device():
    assert "compute" in CFG.as_dict()
    assert str(CFG.compute.device) in {"auto", "cpu", "cuda"} or str(
        CFG.compute.device
    ).startswith("cuda:")


def test_auto_picks_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device_mod, "cuda_available", lambda: True)
    assert device_mod.resolve_device(_StubConfig("auto"), cache=False) == "cuda"


def test_auto_falls_back_to_cpu_without_a_driver(monkeypatch):
    """The whole point of 'auto': a host with no working driver still runs."""
    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)
    assert device_mod.resolve_device(_StubConfig("auto"), cache=False) == "cpu"


def test_explicit_cuda_raises_when_unreachable(monkeypatch):
    """An explicitly named device is an assertion, not a preference.

    Falling back silently would turn a missing driver into an unexplained slowdown,
    which is exactly the failure this key exists to make visible.
    """
    monkeypatch.setattr(device_mod, "cuda_available", lambda: False)
    with pytest.raises(ConfigError):
        device_mod.resolve_device(_StubConfig("cuda"), cache=False)


def test_explicit_cuda_index_is_passed_through(monkeypatch):
    monkeypatch.setattr(device_mod, "cuda_available", lambda: True)
    assert device_mod.resolve_device(_StubConfig("cuda:1"), cache=False) == "cuda:1"


def test_cpu_never_probes_cuda(monkeypatch):
    def _boom():
        raise AssertionError("cpu must not probe for CUDA")

    monkeypatch.setattr(device_mod, "cuda_available", _boom)
    assert device_mod.resolve_device(_StubConfig("cpu"), cache=False) == "cpu"


def test_an_unknown_device_name_fails_loudly():
    with pytest.raises(ConfigError):
        device_mod.resolve_device(_StubConfig("tpu"), cache=False)


def test_resolution_is_memoised(monkeypatch):
    calls = []

    def _probe():
        calls.append(1)
        return False

    monkeypatch.setattr(device_mod, "cuda_available", _probe)
    stub = _StubConfig("auto")
    assert device_mod.resolve_device(stub) == "cpu"
    assert device_mod.resolve_device(stub) == "cpu"
    assert len(calls) == 1, "every model loader calls this; the CUDA probe is not free"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: __import__("retrieval.embedder", fromlist=["Embedder"]).Embedder(CFG),
        lambda: __import__("retrieval.reranker", fromlist=["Reranker"]).Reranker(CFG),
        lambda: __import__("analysis.nli", fromlist=["NLIModel"]).NLIModel(CFG),
    ],
    ids=["bi_encoder", "cross_encoder_reranker", "nli"],
)
def test_every_local_model_records_a_device(factory):
    """All three resolve at construction, so none silently defaults elsewhere."""
    assert factory().device == device_mod.resolve_device(CFG)
