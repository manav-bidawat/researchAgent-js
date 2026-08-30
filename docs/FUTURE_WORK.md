# Known limitations and future work

Things that are incomplete, over-provisioned, or unmeasured. Recorded deliberately:
each one is a decision that was made and could be revisited, not an oversight.

## Query truncation has no telemetry, and the budget is over-provisioned

`retrieval.max_query_tokens: 64` reserves 64 of the cross-encoder's 512-token window for
the query, which is what forces `chunking.max_tokens` down to 445. Measured across all
63 `retrieve_evidence` calls in `logs/traces/`:

| | tokens |
|---|---|
| longest query ever generated | 29 |
| p95 | 22 |
| mean | 13.0 |
| queries that hit the 64-token limit | **0 (0.0%)** |

Two separate problems:

- **Over-provisioned.** Reserving 64 when p95 is 22 gives up ~35 tokens of chunk budget
  for headroom nothing has used. Chunks could be ~480 tokens instead of 445.
- **Silent when it fires.** If a query ever does exceed 64 it is cut in
  `reranker.py:_pair()` with nothing logged — no trace field, no warning. Retrieval would
  quietly degrade on exactly the longest, most specific queries, which are the ones most
  likely to be reformulations after a failed search.

The telemetry is the more valuable fix and the cheaper one: record the pre-truncation
token count on the trace record, then the budget can be set from data rather than from
caution. Raising the limit without telemetry just moves an unmeasured number.

## The rerank ablation does not measure abstention

`eval/ablation.ts` reports recall, MRR, latency and gold-paper hit rate. None of those is
what reranking actually earns its cost on here — the relevance gate thresholds the
cross-encoder's score, so rerank is what makes abstention work. Acting on the ablation
alone produced a wrong conclusion once already (see `docs/EVALUATION.md`).

## `k_retrieve` has never been swept

Fixed at 40. Rerank latency scales with the shortlist, so a smaller pool might keep the
gate at a fraction of the 1527 ms. No measurement has been taken.

## The eval has no "absent but fetchable" category

All three `not_in_corpus` questions are annotated `expect_abstention: true`, but two of
them (`ni02` four-bit quantization on ViTs, `ni03` client drift under non-IID data) are
topics arXiv covers well. The eval therefore rewards abstaining where calling
`search_literature` was arguably the better action. A third category — where fetching is
correct and abstention counts as a miss — would test the tool the current set never
exercises.

## Abstention vs. literature search is resolved by prompt, not by rule

Both answer the same signal: the corpus cannot answer this. Which one fires is left to
the model via `src/prompts/system.md`. In every eval run the agent chose to abstain and
*offer* to search rather than search, so `search_literature` is implemented and tested
but has no end-to-end transcript.

## No cross-session memory

Each CLI `ask` is an independent process; question 2 knows nothing about question 1.
Within a question, context is fully preserved and budget-managed (`conversation.py`).
Making it conversational means holding one `Conversation` across questions — roughly
30–50 lines — plus deciding whether chunk dedup stays per-question or per-conversation.

## Flat index, no eviction

`IndexFlatIP` is exhaustive and exact: 19 µs at 406 vectors, ~48 ms extrapolated to 1M.
Past roughly 100k vectors this wants IVF or HNSW, at the cost of a recall variable that
cannot be measured meaningfully on 11 papers. Removal is a non-goal for the same
underlying reason — `faiss_id_map` is positional, so eviction implies a full rebuild.

## No general Python execution tool

`analyze_corpus` performs bounded data analysis (clustering, metadata statistics) rather
than exposing arbitrary code execution. A deliberate choice, not an omission.

## The web demo is a shim, outside the system under evaluation

`.claude/CLAUDE.md` says "No UI. CLI only", and that constraint exists because how the
agentic system is structured is itself being graded — a UI layer is one more place for
behaviour to hide. The demo page (`web/static/`, served by `app/web/server.ts`) was
added anyway, deliberately, to show the system to someone
who is not going to read a terminal.

What keeps the constraint's intent intact:

- It lives outside `src/` and reaches it only through the bridge worker, the same
  direction as `eval/`. Nothing in `src/` references it, so removing it changes nothing
  about the agent.
- It adds nothing to the engine: the server is the same Express app the TypeScript layer
  already depends on, and the page uses a hand-written markdown renderer rather than a
  JS library.
- It runs the same `AgentLoop` with the same registry and config as the CLI's `ask`,
  through the same bridge op. There is no second code path to keep in sync, and no answer can appear in the
  browser that the CLI would not also give.

What it is not: it holds one shared worker that runs one request at a time, so it serves one question
at a time and is not a server for more than one person. It has no authentication and binds
to localhost by default. Neither is worth fixing unless it stops being a demo.
