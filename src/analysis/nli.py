"""
Natural language inference over sentence pairs, for contradiction and groundedness.

In:  (premise, hypothesis) pairs — premise is always evidence, hypothesis always a claim.
Out: {"label": 'entailment'|'neutral'|'contradiction', "scores": {...}} per pair. Labels
     are read from the model's own id2label, never assumed from position.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.device import resolve_device
from config import CFG, Config

LABELS = ("contradiction", "entailment", "neutral")


class NLIError(RuntimeError):
    """The NLI model could not be loaded."""


def _softmax(row: np.ndarray) -> np.ndarray:
    shifted = row - np.max(row)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


class NLIModel:
    """Wraps the cross-encoder NLI checkpoint named in config.

    Label order is read from the checkpoint rather than assumed. This model orders them
    (contradiction, entailment, neutral), which is *not* the MNLI convention, and
    hardcoding a position would silently invert entailment and contradiction — the two
    labels the whole tool turns on.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.model_name = str(config.nli.model)
        self._model = None
        self.device = resolve_device(config)
        self._labels: List[str] = list(LABELS)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise NLIError(f"sentence-transformers is not installed: {exc}") from None
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                raise NLIError(
                    f"could not load {self.model_name} on {self.device}: {exc}"
                ) from None

            id2label = getattr(getattr(self._model, "config", None), "id2label", None)
            if isinstance(id2label, dict) and len(id2label) == 3:
                self._labels = [str(id2label[i]).lower() for i in sorted(id2label)]
        return self._model

    @property
    def labels(self) -> List[str]:
        self.model  # force the load so labels are read from the checkpoint
        return list(self._labels)

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Score (premise, hypothesis) pairs. Direction is fixed and load-bearing.

        Premise is the evidence, hypothesis is the claim. Swapping them asks a different
        question — "does the claim support the evidence" — and quietly changes the answer.
        """
        if not pairs:
            return []
        raw = self.model.predict(list(pairs), show_progress_bar=False)
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        labels = self.labels
        out: List[Dict[str, Any]] = []
        for row in array:
            probabilities = _softmax(row)
            scores = {label: float(probabilities[i]) for i, label in enumerate(labels)}
            best = max(scores, key=scores.get)
            out.append({"label": best, "confidence": scores[best], "scores": scores})
        return out
