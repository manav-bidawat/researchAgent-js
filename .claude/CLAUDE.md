# Project context for Claude Code

## What we are building

A topic-agnostic agentic research assistant over scientific papers. Read
`docs/ARCHITECTURE.md` and `docs/TOOLS.md` before writing code — they are the spec,
not background reading. If something in this file conflicts with those, they win.

## Working agreement

**The architecture is already decided.** Design decisions in the docs were made
deliberately and often against plausible alternatives that are documented as
rejected. Do not re-litigate them mid-implementation. If a decision genuinely cannot
be implemented as specified, stop and say so with the reason — do not silently
substitute a different design.

**Implement to spec, in small pieces.** Prefer several small verifiable modules over
one large one. Each module gets a 3–4 line docstring at the top: what it does, what
goes in, what comes out. These are load-bearing — they are how the design stays
reviewable.

**Ask before adding a dependency.** The stack is fixed (below). New libraries need a
reason.

## Stack

- `arxiv` / `requests` — collection
- `PyMuPDF` (fitz) — PDF text + image extraction
- `sentence-transformers` — bi-encoder embeddings, cross-encoder rerank, NLI
- `faiss-cpu` — vector index (flat)
- `scikit-learn`, `pandas`, `numpy` — clustering, metrics, metadata analysis
- OpenRouter via a thin `llm_client` wrapper — all LLM calls

Python 3.11+.

The Python engine in `src/` keeps the stack above. Around it is a TypeScript layer
(Node ≥20): Express 5 (demo web server), `@modelcontextprotocol/sdk` (MCP server), `zod`
(schemas), `commander` (CLI); `tsx` for dev; Vitest + supertest for tests. It reaches the
engine only through the long-lived worker `bridge/worker.py` (JSON lines over
stdin/stdout; stdout is reserved for the protocol).

## Hard constraints

- **No agent framework.** No LangChain, no LlamaIndex, no CrewAI. The tool loop is
  written directly against the provider SDK. How the agentic system is structured is
  itself being evaluated — a framework hides that.
- **No multi-agent orchestration.** One agent, five tools.
- **No UI in the engine.** The CLI is `app/cli.ts`; the Express demo page (`app/web/`,
  `web/static/`) lives outside the engine and reaches it only through the bridge.
- **Tools return structured dicts, never raw strings.** Every tool result is
  JSON-serialisable with explicit fields.
- **Tools never raise.** Errors come back as `{"error": ..., "detail": ...}`.
- **Tools never call other tools.** All control flow goes through the agent loop.
- **Prompts live in files**, not in string literals in code —
  `src/prompts/*.md`, loaded at runtime. They will be iterated on and diffed.
- **No hardcoded topic anywhere in `src/`.** Topics are runtime arguments. Eval
  topics belong in eval config only. If a topic name appears in a source file
  outside `eval/`, that is a bug.

## Conventions

- Config in one place (`config.py` or `config.yaml`) — model names, `k` values,
  thresholds, caps. No magic numbers scattered through modules.
- Caches are keyed by **content hash**, not file path.
- Anything expensive (embeddings, figure descriptions, cluster results) is cached and
  re-checked before recompute.
- Log every tool call with its arguments and a truncated result — the eval harness
  reads these traces, and they are also how we debug agent behaviour.
- Type hints on public functions.
- New engine entry points are bridge ops registered in `bridge/ops/` — a fixed table,
  never a generic eval/call-anything hook.

## Things that have already gone wrong once — do not repeat

- Running the cross-encoder over the whole corpus instead of over the bi-encoder's
  candidate shortlist. Rerank is stage two, on 30–50 candidates, never on the index.
- Losing the `faiss_idx -> chunk_id` mapping across restarts. It must be persisted
  with the index.
- Embedding a figure's generated description without its caption. Both, concatenated.
- Deciding what to re-embed by `chunk_id` alone. A `chunk_id` is positional within its
  paper, so it survives a re-ingest even when the text under it changes — figure chunks
  gaining a vision description is exactly that. Skipping on id leaves the stale vector
  in place and retrieval quietly keeps matching text the chunk no longer contains.
  Compare `content_hash` against `indexed_hashes`, and rebuild when it differs.
- Forgetting that `faiss_id_map` is **positional**. Index `i` means FAISS slot `i`, and
  `IndexFlat.remove_ids` swap-compacts, moving the last vector into the freed slot — so
  any removal invalidates every downstream entry of the map and every downstream row of
  `embeddings.npy`. That is why eviction implies a full rebuild, and why eviction is a
  non-goal (`docs/ARCHITECTURE.md` §6). `IndexIDMap` is the fix, if it ever matters.
- Sizing chunks against the *cross-encoder's* limit while ignoring the *bi-encoder's*.
  The embed stage is the one that decides whether a chunk is findable at all; the
  rerank stage only reorders a shortlist. Check both, and count tokens with the
  bi-encoder's own tokenizer.
- Prefixing BGE passages with the query instruction. The instruction goes on the query
  only — both sides, or neither, degrades retrieval.
- Appending a new `topic_tag` to the paper record only. Chunk records denormalise
  `topic_tags`, so they must be backfilled in the same operation — otherwise
  `analyze_corpus` counts the paper under the new topic while `topic_filter` returns
  nothing for it, and the bug surfaces far from its cause.
- Letting a bare string into `topic_tags`. It must be `list[str]`; a scalar passes
  every `in` test character-by-character and silently turns `topic_filter` into a
  substring match. Assert list-ness where records are constructed.
- Treating a chunk's `content_hash` as a cross-paper dedup key. It is an embedding
  cache key. Two papers may legitimately share text; both keep their own chunk record.
  Dropping one leaves a hole in `position`, and neighbour expansion walks straight
  over it. Within-paper dedup only, and assign `position` after it.
- Reusing the cross-encoder's `relevance_threshold` on the bi-encoder fallback path.
  Cross-encoder scores are uncalibrated logits spanning roughly -11..+11; bi-encoder
  scores are cosines in 0..1, and every cosine clears a threshold of -3.0. The gate
  then passes everything, which does not raise — it removes abstention and answers
  absent-topic questions from whatever ranked first. The threshold must be chosen by
  which scorer actually ran, including when reranking was requested and fell back.
- Concluding that rerank is not worth its latency from the MRR ablation alone. That
  ablation measures recall, MRR and latency, none of which is what rerank earns its
  cost on here: the relevance gate thresholds the cross-encoder's score, so rerank is
  what makes abstention work. Turning it off is a gate change wearing a ranking
  change's clothes.
- Passing explicit `categories` to `search_and_fetch` on top of a planned query. The
  query planner already folds its own categories into `plan["query"]`
  (`arxiv_query.apply_categories`), so the caller's are ANDed onto that filter rather
  than replacing it, and the search narrows twice: a planner choosing
  `q-bio.QM OR cs.LG OR cs.AI` plus a caller passing `q-bio.BM OR cs.LG` searches
  neither set. Pass categories only when overriding the planner deliberately.
- Hammering arXiv while iterating. `export.arxiv.org` rate-limits by IP and answers
  HTTP 429 for *every* query once tripped, not just the one that tripped it — a
  cooling-off period of many minutes, during which no collection work can proceed.
  Retry with a bounded, spaced loop; never a tight one.
- Trying to return an image as base64 inside a tool result. The OpenAI-compatible
  surface only accepts image parts on a `user` message, so the agent loop attaches the
  image on a follow-up turn (`llm_client.attach_images`); the tool returns a path.

## Testing

- Each tool gets a smoke test that runs it in isolation and asserts the return shape.
- The agent loop gets a test with a stubbed LLM that asserts tool dispatch and the
  iteration cap.
- Eval harness is separate from `src/` and reaches it through the bridge — never the reverse.
- Vitest (`npm test`) covers the TypeScript layer; pytest (`npm run test:py`) covers the engine.
