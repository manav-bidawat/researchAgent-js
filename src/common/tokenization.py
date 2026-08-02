"""
Token counting with the bi-encoder's own tokenizer — the ruler the chunker measures with.

In:  text, and optionally a token budget.
Out: exact token counts and budget-respecting truncation. The tokenizer is loaded once,
     lazily, so importing this module costs nothing until a count is actually needed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

from config import CFG, Config

# Roughly the average characters-per-token for English wordpiece. Used only by the
# fallback counter, never when a real tokenizer is available.
_FALLBACK_CHARS_PER_TOKEN = 4.0
_WORD = re.compile(r"\S+")


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    """Load and cache the HF tokenizer. Returns None if transformers is unavailable.

    A missing tokenizer must not stop indexing, so callers degrade to an estimate.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None
    # This instance only ever measures text, it is never fed to the model, so its length
    # cap does not apply. Without this, counting a long buffer emits a spurious
    # "sequence length is longer than the specified maximum" warning on every call.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


class TokenCounter:
    """Counts tokens the way the models that consume the chunk will count them.

    The bi-encoder and the cross-encoder in this stack share a vocabulary, so one
    tokenizer measures both budgets. If either model is swapped for one with a
    different vocabulary, that equivalence stops holding (docs/ARCHITECTURE.md §5).
    """

    def __init__(self, config: Config = CFG) -> None:
        self.model_name = config.embedding.model
        self._tokenizer = _load_tokenizer(self.model_name)

    @property
    def exact(self) -> bool:
        """False when running on the character-ratio estimate instead of a tokenizer."""
        return self._tokenizer is not None

    def count(self, text: str) -> int:
        """Number of tokens in `text`, excluding the special tokens added at encode time."""
        if not text:
            return 0
        if self._tokenizer is None:
            return max(1, int(len(text) / _FALLBACK_CHARS_PER_TOKEN))
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def fits(self, text: str, budget: int) -> bool:
        return self.count(text) <= budget

    def truncate(self, text: str, budget: int) -> str:
        """Cut `text` down to `budget` tokens, on a word boundary where possible."""
        if budget <= 0 or not text:
            return ""
        if self.count(text) <= budget:
            return text
        if self._tokenizer is None:
            return text[: int(budget * _FALLBACK_CHARS_PER_TOKEN)].rstrip()

        ids = self._tokenizer.encode(text, add_special_tokens=False)[:budget]
        cut = self._tokenizer.decode(ids, skip_special_tokens=True).strip()
        # Decoding mid-word leaves a fragment; drop it unless it is all we have.
        if " " in cut and not text.startswith(cut):
            head, _, tail = cut.rpartition(" ")
            if head and text.find(tail) == -1:
                return head
        return cut

    def split_sentences(self, text: str) -> List[str]:
        """Split into sentences. Used by the chunker and, later, by the NLI tool."""
        return split_sentences(text)


# Abbreviations that end in a period but do not end a sentence.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "cf.", "et al.", "etc.", "vs.", "Fig.", "Eq.", "Sec.", "Tab.",
    "Ref.", "Refs.", "approx.", "Dr.", "Prof.", "al.", "No.", "Vol.", "pp.",
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> List[str]:
    """Sentence split tuned for scientific prose.

    Protects the abbreviations and decimal numbers that a naive split-on-period
    mangles, which matters because the NLI tool uses one sentence as its premise.
    """
    if not text or not text.strip():
        return []

    guarded = text
    for index, abbreviation in enumerate(_ABBREVIATIONS):
        guarded = guarded.replace(abbreviation, f"\x00{index}\x00")
    # Protect decimals: "0.95" must not split after the period.
    guarded = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)}\x01{m.group(2)}", guarded)

    parts = _SENTENCE_END.split(guarded)

    restored: List[str] = []
    for part in parts:
        for index, abbreviation in enumerate(_ABBREVIATIONS):
            part = part.replace(f"\x00{index}\x00", abbreviation)
        part = part.replace("\x01", ".").strip()
        if part:
            restored.append(part)
    return restored


def word_count(text: str) -> int:
    """Whitespace-delimited word count, used for cheap heuristics."""
    return len(_WORD.findall(text or ""))
