# Data Schema

Single source of truth for every record shape in the system. If a field is not
described here, it does not exist. Every module reads and writes these shapes — do
not invent local variants.

All ids are strings. All timestamps are ISO 8601 UTC.

**Topic naming.** Three similar names appear across records and tool surfaces. They are
not interchangeable, and the plural/singular distinction is load-bearing:

| Name | Type | Where it appears | Meaning |
|---|---|---|---|
| `topic_tags` | `list[str]` | paper record (§1), chunk record (§2) | Every topic this record has been reached through. **Always a list**, even when it holds one element. |
| `topic_tag` | `str` | `search_literature` return, keys of `manifest.topics` (§4) | A single tag — the one a call just applied, or the one naming a topic entry. |
| `topic_filter` | `str` | `retrieve_evidence` input | A **membership test**, not an equality test: keep a chunk when `topic_filter in chunk["topic_tags"]`. |

Records are constructed with `topic_tags` asserted to be a list. A bare string passes
every `in` test character-by-character (`"cs" in "cs.LG"` is true), so a scalar that
slips in does not raise — it silently widens `topic_filter` to substring matching.

**Re-encountering a paper under a second topic.** Paper dedup is on `paper_id`, so the
second search neither re-downloads nor re-chunks. It must still append the new tag, and
to **both** the paper record *and* every chunk record that paper already produced.
Chunk records denormalise `topic_tags` (§2) precisely so retrieval needs no join — which
means patching the manifest alone leaves the paper counted under the new topic by
`analyze_corpus` while `topic_filter` returns nothing for it. That failure is silent and
presents as a retrieval bug, a long way from its cause.

Because `chunks.jsonl` is line-oriented, that backfill rewrites the file. Write a temp
file and rename it, so an interrupted backfill cannot leave a half-written corpus.

---

## 1. Paper record

Stored in `manifest.json` under `papers`, keyed by `paper_id`.

```json
{
  "paper_id":      "arxiv id, normalised, dots and slashes replaced with underscores (e.g. '2103_14030v2')",
  "arxiv_id":      "raw arxiv id as returned by the API (e.g. '2103.14030v2')",
  "title":         "string",
  "authors":       ["string"],
  "abstract":      "string",
  "year":          "int",
  "published":     "ISO 8601 date string",
  "categories":    ["string"],
  "pdf_url":       "string",
  "pdf_path":      "string — local path under data/papers/",
  "topic_tags":    ["string"],
  "n_chunks":      "int",
  "n_figures":     "int",
  "indexed_at":    "ISO 8601 timestamp",
  "parse_status":  "'ok' | 'partial' | 'failed'",
  "parse_note":    "string — reason, when status is not ok"
}
```

`paper_id` is the primary dedup key. `topic_tags` is a **list**: the same paper can
arrive via more than one topic search and must not be duplicated — the new tag is
appended to the existing record instead, and to that paper's chunk records in the same
operation. See **Topic naming** above.

`parse_status: 'partial'` is a valid, kept state. A paper whose text extracted but
whose figures failed is still useful.

---

## 2. Chunk record

Stored one-per-line in `chunks.jsonl`. The index-order of lines is meaningless; the
`chunk_id` is the identity.

```json
{
  "chunk_id":    "string — '{paper_id}__c{NNNN}', zero padded, sequential per paper",
  "paper_id":    "string",
  "paper_title": "string — denormalised so retrieval results need no join",
  "topic_tags":  ["string"],
  "chunk_type":  "'text' | 'figure' | 'table'",
  "text":        "string — the embedded text (see note below)",
  "page":        "int — 1-based",
  "section":     "string | null — detected section heading, null if undetectable",
  "position": "int — ordinal within the paper, assigned AFTER within-paper dedup
             so it is always dense. Neighbour expansion depends on this.",
  "n_tokens":    "int",
  "content_hash": "sha256 of text. Used as an embedding cache key, NOT as a
                 cross-paper dedup key. Two papers may legitimately contain
                 identical text; both keep their own chunk records.",
  "figure_id":   "string | null — present only for figure/table chunks",
  "image_path":  "string | null — present only for figure/table chunks",
  "caption":     "string | null — present only for figure/table chunks"
}
```

**`topic_tags` is denormalised onto every chunk** so retrieval can filter without
joining back to the manifest. The cost of that is a write obligation: when a paper
gains a tag, every one of its chunk records gains it too, in the same operation.

**`text` for figure and table chunks is `caption + '\n' + description`, concatenated.**
Not the description alone. The caption carries the authors' exact terminology; the
generated description tends to paraphrase it away. This is what gets embedded.

**`position`** is what neighbour expansion walks. It is assigned **after** within-paper
dedup, so it is dense by construction: `0..n-1` with no holes, covering text and
figure/table chunks alike in reading order, so `position ± 1` always lands on
genuinely adjacent content.

**A chunk is never dropped because a different paper contains the same text.**
Identical text in two papers is two chunk records, with two `chunk_id`s and two
`position`s — the text is shared, the identity is not. Removing one would delete
content from the second paper's reading order and punch a hole in the neighbour chain
that no later stage can detect. `content_hash` only makes the *embedding* cheap (§4);
the dedup that actually removes a record is within-paper and nothing else.

---

## 3. Figure record

Stored in `manifest.json` under `figures`, keyed by `figure_id`. Separate from the
chunk record because the image is inspectable independently of retrieval.

```json
{
  "figure_id":    "string — '{paper_id}__f{NN}'",
  "paper_id":     "string",
  "kind":         "'figure' | 'table'",
  "page":         "int",
  "image_path":   "string — path under data/cache/figures/{paper_id}/",
  "image_hash":   "string — sha256 of the image bytes",
  "caption":      "string — extracted from surrounding text, empty string if none found",
  "description":  "string — vision-LLM generated",
  "width":        "int",
  "height":       "int",
  "chunk_id":     "string — the chunk this figure produced"
}
```

`image_hash` is the cache key for the description. The same image appearing in two
papers (or the same paper re-indexed) costs one vision call, not two.

---

## 4. Manifest

`data/index/manifest.json`. Loaded fully at startup — it is small.

```json
{
  "schema_version":  "int",
  "embedding_model": "string — full model name",
  "embedding_dim":   "int",
  "created_at":      "ISO 8601",
  "updated_at":      "ISO 8601",
  "papers":  { "<paper_id>": { /* paper record */ } },
  "figures": { "<figure_id>": { /* figure record */ } },
  "topics": {
    "<topic_tag>": {
      "query":         "string — original natural language topic",
      "arxiv_query":   "string — the LLM-planned query actually sent",
      "paper_ids":     ["string"],
      "created_at":    "ISO 8601",
      "last_accessed": "ISO 8601"
    }
  },
  "faiss_id_map": ["chunk_id", "chunk_id", "..."],
  "indexed_hashes": ["sha256", "sha256", "..."],
  "content_hashes": "map sha256 -> embedding cache reference. Not a dedup index.",
  "manifest_hash": "string — sha256 over paper_ids + chunk count, used as cache key"
}
```

**`faiss_id_map` is critical.** FAISS assigns sequential integer ids on `add()`. The
mapping from FAISS position → `chunk_id` exists nowhere else and is unrecoverable if
lost. It is a plain list where index `i` holds the `chunk_id` at FAISS position `i`,
and it must be written to disk in the same operation that writes the index. Never
write one without the other.

**`indexed_hashes` is positional too**, parallel to `faiss_id_map`: index `i` holds the
`content_hash` of the text actually embedded at FAISS position `i`. It answers a
question `faiss_id_map` cannot — *is this vector still current?*

`chunk_id` is `{paper_id}__c{position}`, so it survives a re-ingest unchanged even when
the text beneath it does not. Figure chunks are the case that proves it: a figure
indexed before its vision description exists keeps the same `chunk_id` when the
description arrives, but its `text` goes from `caption` to `caption + description`.
Deciding what to embed by `chunk_id` alone leaves the old caption-only vector in place
forever, and retrieval keeps matching text the chunk no longer contains — silently,
with no error and no way to notice from the outside.

So a chunk is re-embedded when its `content_hash` differs from its `indexed_hashes`
entry, not merely when its `chunk_id` is absent. An index written before this field
existed has no entries; those vectors are treated as unknown rather than current, which
costs one full rebuild on first load and nothing after.

**Changed content forces a rebuild, not an append.** This is the second exception to
"incremental, never rebuild" (§5 of `ARCHITECTURE.md`), alongside a changed embedding
model. A flat index cannot replace a vector in place: `remove_ids` swap-compacts, which
renumbers every position after the hole and invalidates the rest of `faiss_id_map`.
Rebuilding is cheap in practice because unchanged text hits the embedding cache — the
rebuild that introduced this field re-embedded 17 figure chunks and served the other 98
from cache.

`embedding_model` and `embedding_dim` are recorded so a model change is detected on
load. A mismatch means a full rebuild — appending vectors from a different model to
an existing index silently corrupts retrieval, so this check is mandatory, not
advisory.

`content_hashes` is an **embedding cache index, not a dedup index.** It maps
`sha256(text)` to its cached vector in `data/cache/embeddings.json`, so text appearing
in two papers is embedded once and reused twice. Both papers keep their own chunk
record; a hash hit removes nothing. Reading this map as a dedup key is exactly what
would put holes in `position` — see §2.

`manifest_hash` is what `analyze_corpus` caches cluster results against.

---

## 5. On-disk layout

```
data/
  papers/
    {paper_id}.pdf
  index/
    manifest.json
    chunks.jsonl
    embeddings.npy          # float32, shape (n_chunks, embedding_dim), row i aligns with faiss_id_map[i]
    faiss.index
  cache/
    embeddings.json         # { "<content_hash>": [float, ...] } — embedding reuse across papers
    figures/{paper_id}/{figure_id}.png
    descriptions.json       # { "<image_hash>": "<description>" }
    arxiv_queries.json      # { "<sha256(topic)>": "<planned arxiv query>" }
    clusters.json           # { "<manifest_hash>": { /* cluster result */ } }
```

`embeddings.npy` is kept alongside the FAISS index because `analyze_corpus` needs raw
vectors for clustering and reading them back out of FAISS is awkward. Row `i` of the
array corresponds to `faiss_id_map[i]` — the two must be appended in lockstep.

---

## 6. Tool return envelope

Every tool returns a JSON-serialisable dict. Success and failure are distinguished by
the presence of `error`.

Success: the shape documented per-tool in `docs/TOOLS.md`.

Failure:

```json
{
  "error":  "string — short machine-ish code, e.g. 'arxiv_unavailable', 'no_results', 'parse_failed'",
  "detail": "string — human readable, safe to show the model",
  "partial": "object | null — whatever did succeed, when partial success is meaningful"
}
```

Tools never raise. A tool that throws crashes the agent loop; a tool that returns an
error lets the agent adapt.

`partial` matters for `search_literature` — three papers indexed out of five
requested is a useful result, not a failure.

---

## 7. Tool call trace

Appended to `logs/traces/{run_id}.jsonl`, one record per tool call. The eval harness
reads these; they are also the primary debugging surface.

```json
{
  "run_id":      "string",
  "question_id": "string | null — set during eval runs",
  "iteration":   "int — 0-based position in the agent loop",
  "tool_name":   "string",
  "args":        "object — as the model supplied them",
  "result_summary": "object — truncated result, NOT the full payload",
  "chunk_ids_returned": ["string"],
  "error":       "string | null",
  "latency_ms":  "int",
  "timestamp":   "ISO 8601"
}
```

`chunk_ids_returned` is pulled out as a top-level field specifically so retrieval
recall can be computed against gold labels without parsing `result_summary`.

`result_summary` is truncated deliberately — full chunk text in every trace record
makes the log unusable.

---

## 8. Eval record

`eval/questions.json`. Gold labels live here and are **never** passed to the agent.

```json
{
  "question_id":   "string",
  "question":      "string — paraphrased, never copied from paper wording",
  "category":      "'single_paper' | 'multi_paper' | 'not_in_corpus' | 'conflicting'",
  "gold_paper_ids":["string"],
  "gold_chunk_ids":["string"],
  "expected_facts":["string"],
  "expect_abstention": "bool",
  "expected_tools": ["string"] ,
  "notes":         "string — why this question is here, what it is probing"
}
```

`expected_tools` is a soft check. The agent is allowed to reach the right answer by a
different route; the field records what the design anticipates so unexpected traces
get looked at rather than silently passing.

`gold_chunk_ids` may be empty for `not_in_corpus` questions — there is nothing to
retrieve, and the correct behaviour is abstention.