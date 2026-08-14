# Tool Specification

Five tools. The agent chooses which to call and in what order; nothing here is
sequenced by the system.

**Two rules that govern every tool:**

1. **Tools never call other tools.** All control flow goes through the agent. If
   `retrieve_evidence` returns `sufficient_evidence: false`, it does *not* trigger a
   fetch — it reports the gap and the agent decides whether to call
   `search_literature`. That decision is the agentic behaviour the task is asking
   for; automating it away would recreate a fixed pipeline.
2. **Tools never raise.** Every failure path returns `{"error": "...", "detail": ...}`
   so a bad PDF or a dead API degrades the loop instead of killing it.

Tool descriptions below are the **exact text the model sees**. They are written to
disambiguate, because the most common failure in agentic systems is a model that
cannot tell two similar tools apart.

---

## 1. `search_literature`

Fetches new papers from arXiv and adds them to the index.

**Description shown to the model:**

> Search arXiv for papers on a topic, download them, and add them to the index.
> Use this when the corpus has no papers on the subject being asked about, or when
> `retrieve_evidence` reported insufficient evidence and the gap looks like missing
> literature rather than a bad query. This is slow (30–60s) and costs API calls — do
> not use it to answer questions about papers that are already indexed; use
> `retrieve_evidence` for that. Do not call it twice for the same topic.

**Input**

```json
{
  "query":       "string, required — natural language topic",
  "max_results": "int, optional, default 8",
  "categories":  "string[], optional — arXiv categories e.g. ['cs.LG','cs.CL']"
}
```

**Behaviour**

1. LLM call translates `query` into arXiv query syntax (field prefixes, booleans,
   category filters)
2. arXiv API call, `sort_by=relevance`
3. dedup against manifest on `arxiv_id`
4. download new PDFs only
5. extract text + figures/tables, chunk, describe figures, embed, append to index
6. update the manifest with `topic_tag` — and on a dedup hit, append that tag to the
   existing paper record *and* to every chunk record it already produced
   (`docs/DATA_SCHEMA.md`, **Topic naming**)

**Returns**

```json
{
  "papers_added":   [{"arxiv_id","title","year","abstract_snippet"}],
  "papers_skipped": "int — already indexed",
  "chunks_added":   "int",
  "figures_added":  "int",
  "topic_tag":      "string",
  "total_indexed":  "int"
}
```

**Failure modes:** arXiv unreachable, zero results, PDF download failure, PDF parse
failure. Partial success is reported, not thrown away — three good papers out of five
is a valid result.

**Rate limiting.** arXiv answers HTTP 429 per *IP*, for every query, not only the one
that tripped it, and the block lasts minutes. Searches and PDF downloads therefore
share one delay clock (`src/corpus/arxiv_fetch.py`); downloads do not go through
`arxiv.Result.download_pdf`, which bypasses it. On a 429 the tool returns

```json
{"error": "arxiv_rate_limited", "detail": "...when it lifts, and how to clear it",
 "partial": {"papers_added": [...], "chunks_added": 0, "topic_tag": "..."}}
```

and stops immediately rather than retrying — a retry is one more request into the
block. `partial` carries whatever was downloaded and indexed before the 429, because
dedup means papers not ingested now are never ingested. The expiry is recorded in
`paths.arxiv_cooldown` and checked before the next search sends anything, so varying
the wording and calling again costs zero requests; delete that file (or pass
`--clear-arxiv-cooldown` to the CLI's `index`) to clear it early.

---

## 2. `retrieve_evidence`

Core RAG. The tool the agent should reach for by default.

**Description shown to the model:**

> Retrieve passages, figures, and tables from the indexed papers that are relevant to
> a query. This is your primary source of evidence — use it for any question about
> what the papers say. Returns ranked chunks with paper, page, and section metadata
> so you can cite them precisely. If `sufficient_evidence` comes back false, the
> index does not contain good evidence for this query: either rephrase and retry with
> different wording, or — if the topic is genuinely absent from the corpus —
> call `search_literature`, or tell the user you cannot answer from available
> evidence. Do not answer from your own knowledge when this returns nothing.

**Input**

```json
{
  "query":        "string, required",
  "k":            "int, optional, default 5",
  "topic_filter": "string, optional — one tag; matches chunks whose topic_tags contain it",
  "chunk_types":  "string[], optional — subset of ['text','figure','table']"
}
```

**Behaviour**

1. bi-encoder retrieval, `k_retrieve = 30–50`
2. metadata filter (`topic_tags` membership, `chunk_types`) — applied via FAISS ID selector where
   possible, else post-filter
3. cross-encoder rerank over candidates
4. **relevance gate** — top scores vs. empirically tuned threshold
5. neighbour expansion — attach `±1` adjacent chunks from the same paper
6. truncate each chunk to ~500 tokens
7. deduplicate against chunks already returned earlier in this conversation

**Returns**

```json
{
  "chunks": [{
    "chunk_id","paper_id","paper_title","page","section",
    "chunk_type","score","text",
    "figure_id": "present only when chunk_type is figure|table"
  }],
  "sufficient_evidence": "bool",
  "note": "string — why the gate failed, when it did",
  "n_candidates_considered": "int"
}
```

**Note on figure chunks:** these come back with the *stored description plus
caption*, not the image. If detail is needed that the description does not carry,
that is what `inspect_figure` is for — and the agent makes that call, not this tool.

---

## 3. `inspect_figure`

Multimodal inspection of an actual image.

**Description shown to the model:**

> Look at the actual image of a figure or table from a paper, rather than its stored
> text description. Use this when `retrieve_evidence` returned a figure or table and
> you need detail the description may have lost — exact values, axis labels, the
> shape of a trend, or what a specific series does. Also use it to inspect an image
> the user has supplied. Requires a `figure_id` from a previous `retrieve_evidence`
> result, or a user-supplied image reference.

**Input**

```json
{
  "paper_id":   "string, optional — required for indexed figures",
  "figure_id":  "string, optional",
  "chunk_id":   "string, optional — alternative to paper_id+figure_id",
  "user_image": "string, optional — path/reference to a user-supplied image",
  "question":   "string, optional — what specifically to look for"
}
```

**Returns**

```json
{
  "image_path":  "string — the image the agent loop must attach",
  "mime_type":   "string",
  "caption":     "string — original caption from the paper",
  "stored_description": "string — the index-time description",
  "paper_id","page","figure_id"
}
```

For a user-supplied image, caption and stored description are absent.

**How the image actually reaches the model.** The tool returns the record above — a
structured dict, like every other tool — and the *agent loop* attaches the image to the
conversation immediately afterwards, via `llm_client.attach_images`, as an `image_url`
content part on a follow-up user turn.

This is not a stylistic choice. The provider surface is OpenRouter's
OpenAI-compatible `/chat/completions`, where a `tool` message's `content` is a string
and image parts are only valid on a `user` message. (Anthropic's *native* API does
accept image blocks inside `tool_result`; the OpenAI-compatible surface does not, and
the whole point of going through OpenRouter is that the provider stays a config value.)
Returning base64 inside the tool result would also have made this the one tool that
returns a raw string blob rather than a structured dict.

So the ordering the loop must preserve is:

```
assistant(tool_call: inspect_figure)
  -> tool(result: the dict above, no image bytes)
  -> user(image_url part + the question, attached by the loop)
  -> assistant(reads the image)
```

The tool still never calls another tool, and control flow still lives in the agent.

**Why both paths exist:** the indexed description makes figures *findable*; the raw
image makes them *readable*. Retrieval needs the first, precision needs the second.

---

## 4. `check_evidence_consistency`

NLI-based checking, two modes.

**Description shown to the model:**

> Check retrieved evidence for consistency using a natural language inference model.
> Two modes. `contradiction`: given several chunks, find pairs that disagree with each
> other — use this before answering when sources might conflict, so you can surface
> the disagreement rather than silently picking one side. `groundedness`: given your
> drafted answer text and the chunks you used, check whether each claim in the answer
> is actually supported by the evidence — use this before finalising an answer built
> from multiple sources. Claims flagged `neutral` are not supported by the retrieved
> evidence and should be removed, softened, or explicitly marked as uncertain.
> Treat the output as advisory: it is a heuristic signal, not a verdict.

**Input**

```json
{
  "mode":        "'contradiction' | 'groundedness', required",
  "chunk_ids":   "string[], required",
  "answer_text": "string — required for groundedness mode only"
}
```

**Behaviour**

- Chunks are split into sentences; the best-matching sentence is used as premise
  (NLI models degrade badly on paragraph-length inputs)
- Direction is fixed: **premise = evidence, hypothesis = claim**
- `groundedness` mode extracts atomic claims from `answer_text` **inside the tool**
  via its own LLM call — deliberately not left to the agent, which would otherwise be
  grading a claim list it selected itself

**Returns — `contradiction` mode**

```json
{
  "found": "bool",
  "conflicting_pairs": [{
    "chunk_a","chunk_b","snippet_a","snippet_b","confidence"
  }]
}
```

**Returns — `groundedness` mode**

```json
{
  "claims": [{
    "claim","label": "entailed|neutral|contradicted",
    "best_supporting_chunk","confidence"
  }],
  "grounded_ratio": "float",
  "unsupported_claims": ["string"]
}
```

`unsupported_claims` is returned explicitly so the agent knows exactly what to fix
rather than having to re-derive it.

**Stated limitations** (also in the architecture doc): general MNLI-trained models are
weak on scientific text; `neutral` conflates "unsupported" with "not understood";
confidence values are uncalibrated.

---

## 5. `analyze_corpus`

Computation over corpus **metadata**, not content. No retrieval, no LLM call.

**Description shown to the model:**

> Compute aggregate statistics over the indexed corpus: how many papers there are,
> which topics they cover, what years they span, and what sub-themes they cluster
> into. Use this for questions *about the corpus itself* — coverage, recency, size,
> thematic structure — and to orient yourself before deciding whether you need to
> fetch more literature. Do NOT use this for questions about what the papers say;
> that needs `retrieve_evidence`.

**Input**

```json
{
  "operation":    "'stats' | 'timeline' | 'cluster' | 'compare_topics'",
  "topic_filter": "string, optional",
  "params":       "object, optional — e.g. {'k': 4} for cluster"
}
```

**Operations**

| Operation | Computes |
|---|---|
| `stats` | papers per topic, chunk counts, figure/table counts, date range |
| `timeline` | per-topic year histogram, median year, recency |
| `cluster` | KMeans over embeddings, `k` by silhouette, TF-IDF cluster labels, **validated against `topic_tags` via adjusted Rand index + purity** |
| `compare_topics` | vocabulary overlap and centroid distance between topic groups |

**Returns**

```json
{
  "operation": "string",
  "result":    "object — numbers, labels, cluster memberships",
  "summary":   "string — one-paragraph natural language summary the agent can quote"
}
```

**Caching:** `cluster` results are cached against a manifest hash and recomputed only
when the index changes.

**`summary` is templated, not generated.** This tool makes no LLM call, so the
one-paragraph summary is produced by deterministic string templating over the numbers
already in `result` — one template per operation. That keeps the tool free of a model
dependency, keeps its output reproducible for eval, and means the summary can never
disagree with the `result` it is describing.

**Why this is the ML component and not decoration:** `k` is selected rather than
assumed, and clusters are scored against known labels. If the clusters fail to
recover the known topic split, that is a real diagnostic about embedding quality, and
it gets reported as such.

---

## Tool selection: expected behaviour

The system prompt does not prescribe an order. These are the patterns the design
anticipates — they are what the eval's tool-trace check looks for, not a script:

| Question shape | Likely trace |
|---|---|
| Factual, corpus has it | `retrieve_evidence` → answer |
| Factual, corpus lacks it | `retrieve_evidence` (gate fails) → `search_literature` → `retrieve_evidence` → answer |
| Genuinely unanswerable | `retrieve_evidence` (gate fails) → possibly one retry → abstain |
| Multi-paper synthesis | `retrieve_evidence` → `check_evidence_consistency(contradiction)` → answer |
| Numeric detail in a plot | `retrieve_evidence` → `inspect_figure` → answer |
| "What's in this corpus?" | `analyze_corpus` → answer |

**Loop control:** max 6–8 iterations, plus loop detection on repeated
`(tool_name, args)` pairs. On cap, the agent answers with what it has and says the
evidence is partial.
