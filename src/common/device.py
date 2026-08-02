"""
Resolves the torch device that every locally-run model (bi-encoder, cross-encoders) uses.

In:  the `compute.device` config value — "auto", "cpu", "cuda", or an explicit id like
     "cuda:0".
Out: a device string sentence-transformers accepts. "auto" picks CUDA when it is usable
     and falls back to CPU silently; an explicitly named device that is unusable raises,
     because that is a misconfiguration and running on CPU instead would hide it.
"""

from __future__ import annotations

from typing import Optional

from config import CFG, Config, ConfigError

_resolved: Optional[str] = None


def cuda_available() -> bool:
    """True when torch can actually reach a CUDA device.

    Imports torch lazily and treats any failure as "no CUDA": on a host whose driver is
    missing or mismatched, `torch.cuda.is_available()` returns False rather than raising,
    but the import itself can still fail on a broken install.
    """
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(config: Config = CFG, cache: bool = True) -> str:
    """The device string to hand to SentenceTransformer/CrossEncoder.

    Resolved once per process and memoised: every model loader calls this, and probing
    CUDA imports torch and initialises its driver context, which is not free.
    """
    global _resolved
    if cache and _resolved is not None:
        return _resolved

    requested = str(config.compute.device).strip().lower()
    if requested == "auto":
        device = "cuda" if cuda_available() else "cpu"
    elif requested.startswith("cuda"):
        if not cuda_available():
            raise ConfigError(
                f"compute.device is '{requested}' but torch cannot reach a CUDA device. "
                "Install the NVIDIA driver (see README), or set compute.device: auto to "
                "fall back to CPU automatically."
            )
        device = requested
    elif requested == "cpu":
        device = "cpu"
    else:
        raise ConfigError(
            f"compute.device must be 'auto', 'cpu', 'cuda' or 'cuda:<n>', not {requested!r}"
        )

    if cache:
        _resolved = device
    return device


def reset_cache() -> None:
    """Forget the memoised device. For tests that swap config between cases."""
    global _resolved
    _resolved = None
