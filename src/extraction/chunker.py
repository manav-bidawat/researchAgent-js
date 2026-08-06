"""
Turns extracted page lines into overlapping, token-bounded, section-aware chunks.

In:  pages from pdf_text.extract_pages, plus the chunking budget from config.
Out: [{"text", "page", "section", "n_tokens"}] in reading order — every chunk within
     chunking.max_tokens as counted by the bi-encoder's own tokenizer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from config import CFG, Config
from common.tokenization import TokenCounter, split_sentences

# A line ending mid-word, hyphenated across a line break: "represen-" + "tation".
_HYPHEN_BREAK = re.compile(r"(\w)-$")
# Lines that are almost certainly not prose worth embedding.
_MOSTLY_SYMBOLS = re.compile(r"^[^A-Za-z]*$")


# A hyphenated compound written inside a line, e.g. "self-attention" in running text.
_INLINE_COMPOUND = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")


def hyphenated_vocabulary(pages: List[Dict[str, Any]]) -> set:
    """Compounds the paper writes with a hyphen *mid-line*, lowercased.

    This is what separates a soft hyphen from a hard one. "vision appli-" + "cations"
    must splice into "applications", but "self-" + "attention" must stay
    "self-attention" — and nothing in the two fragments themselves says which is which.
    The document does: a paper using "self-attention" writes it inside a line hundreds
    of times, while "applications" never appears hyphenated except where it wrapped.
    """
    vocabulary = set()
    for page in pages:
        for line in page.get("lines", []):
            text = str(line.get("text", ""))
            # Ignore a trailing hyphen: that is the wrap we are trying to classify.
            body = text[:-1] if text.endswith("-") else text
            for first, second in _INLINE_COMPOUND.findall(body):
                vocabulary.add(f"{first.lower()}-{second.lower()}")
    return vocabulary


def _splice(head: str, tail: str, compounds: set) -> str:
    """Join a line ending in a hyphen to the next, keeping the hyphen only if lexical."""
    stem = head[:-1]
    first = re.search(r"([A-Za-z]+)$", stem)
    second = re.match(r"([A-Za-z]+)", tail)
    if first and second:
        compound = f"{first.group(1).lower()}-{second.group(1).lower()}"
        if compound in compounds:
            return f"{head}{tail}"  # a real compound that happened to wrap: keep the hyphen
    return f"{stem}{tail}"


def _join_lines(lines: List[str], compounds: Optional[set] = None) -> str:
    """Join PDF lines into prose, repairing hyphenation broken across line breaks."""
    compounds = compounds or set()
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif _HYPHEN_BREAK.search(out):
            out = _splice(out, line, compounds)
        else:
            out = f"{out} {line}"
    return out


def _is_useful(text: str, min_words: int = 4) -> bool:
    """Reject fragments that carry no retrievable content."""
    if not text or _MOSTLY_SYMBOLS.match(text):
        return False
    return len(text.split()) >= min_words


class Chunker:
    """Accumulates lines into chunks that respect a token budget and section bounds.

    Chunks are cut at `target_tokens` and hard-capped at `max_tokens`; a single
    oversized paragraph is split on sentence boundaries rather than mid-word, so no
    chunk is ever silently truncated by the encoders downstream.
    """

    def __init__(self, config: Config = CFG, counter: Optional[TokenCounter] = None) -> None:
        self.target = int(config.chunking.target_tokens)
        self.maximum = int(config.chunking.max_tokens)
        self.overlap = int(config.chunking.overlap_tokens)
        self.section_aware = bool(config.chunking.section_aware)
        self.counter = counter if counter is not None else TokenCounter(config)

    def reflow(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pages of lines into prose blocks, one per contiguous run of the same section.

        Reflowing before chunking is what keeps a chunk boundary from landing inside a
        word: hyphenation is repaired across the whole run first, so the chunker only
        ever sees finished prose.
        """
        compounds = hyphenated_vocabulary(pages)
        blocks: List[Dict[str, Any]] = []
        lines: List[str] = []
        section: Optional[str] = None
        page_number: Optional[int] = None

        def close() -> None:
            nonlocal lines, page_number
            if lines:
                text = _join_lines(lines, compounds)
                if _is_useful(text):
                    blocks.append({"text": text, "page": page_number or 1, "section": section})
            lines, page_number = [], None

        for page in pages:
            number = int(page.get("page", 1))
            for line in page.get("lines", []):
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                if self.section_aware and line.get("section") != section:
                    close()
                    section = line.get("section")
                if page_number is None:
                    page_number = number
                lines.append(text)
        close()
        return blocks

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk the output of pdf_text.extract_pages, in reading order."""
        chunks: List[Dict[str, Any]] = []
        for block in self.reflow(pages):
            chunks.extend(self._split_block(block))
        return self._carry_overlap(chunks)

    def _split_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """One prose block into chunks of about `target` tokens, on sentence bounds."""
        page, section = block["page"], block["section"]
        sentences = split_sentences(block["text"]) or [block["text"]]

        chunks: List[Dict[str, Any]] = []
        current: List[str] = []
        current_tokens = 0
        for sentence in sentences:
            tokens = self.counter.count(sentence)
            if current and current_tokens + tokens > self.target:
                chunks.extend(self._emit(" ".join(current), page, section))
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += tokens
        if current:
            chunks.extend(self._emit(" ".join(current), page, section))
        return chunks

    def _emit(self, text: str, page: int, section: Optional[str]) -> List[Dict[str, Any]]:
        """One buffer into one or more chunks, splitting if it exceeds the hard cap."""
        if self.counter.count(text) <= self.maximum:
            return [{"text": text, "page": page, "section": section,
                     "n_tokens": self.counter.count(text)}]

        pieces: List[Dict[str, Any]] = []
        current: List[str] = []
        current_tokens = 0
        for sentence in split_sentences(text) or [text]:
            tokens = self.counter.count(sentence)
            if tokens > self.maximum:
                # One sentence longer than the whole budget: hard-truncate it, since
                # the alternative is a chunk the encoders would silently cut anyway.
                if current:
                    pieces.append({"text": " ".join(current), "page": page, "section": section,
                                   "n_tokens": current_tokens})
                    current, current_tokens = [], 0
                clipped = self.counter.truncate(sentence, self.maximum)
                pieces.append({"text": clipped, "page": page, "section": section,
                               "n_tokens": self.counter.count(clipped)})
                continue
            if current and current_tokens + tokens > self.maximum:
                pieces.append({"text": " ".join(current), "page": page, "section": section,
                               "n_tokens": current_tokens})
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += tokens
        if current:
            pieces.append({"text": " ".join(current), "page": page, "section": section,
                           "n_tokens": current_tokens})
        return pieces

    def _carry_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prefix each chunk with the tail of the previous one, within the same section.

        Overlap recovers a claim split across a chunk boundary. It is not carried across
        a section change, where the previous text is genuinely unrelated context.
        """
        if self.overlap <= 0 or len(chunks) < 2:
            return chunks

        out = [chunks[0]]
        for previous, chunk in zip(chunks, chunks[1:]):
            if previous.get("section") != chunk.get("section"):
                out.append(chunk)
                continue
            tail = self._tail(previous["text"], self.overlap)
            merged = f"{tail} {chunk['text']}".strip() if tail else chunk["text"]
            if self.counter.count(merged) > self.maximum:
                out.append(chunk)
                continue
            out.append({**chunk, "text": merged, "n_tokens": self.counter.count(merged)})
        return out

    def _tail(self, text: str, budget: int) -> str:
        """The last `budget` tokens of `text`, cut on a sentence boundary if one is near."""
        sentences = split_sentences(text)
        if not sentences:
            return ""
        picked: List[str] = []
        total = 0
        for sentence in reversed(sentences):
            tokens = self.counter.count(sentence)
            if total + tokens > budget and picked:
                break
            picked.insert(0, sentence)
            total += tokens
        return " ".join(picked)
