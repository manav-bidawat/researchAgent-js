# Build Plan

Build in milestones, in this order. **Each milestone ends at a verification gate.
Do not start the next milestone until the current one is verified.**

The point of the gates is that every stage stays reviewable. A single large
generation that touches collection, indexing, retrieval and the agent at once cannot
be meaningfully checked, and errors compound silently across layers.

At the end of each milestone: report what was built, what the verification produced,
and anything that deviated from spec.

---

## M0 — Skeleton

Project structure, config, dependencies. No logic.

- `config.py` (or `config.yaml` + loader) with every tunable in one place
- `requirements.txt`, pinned
- `.env.example`, `.gitignore`
- Directory scaffolding per `docs/DATA_SCHEMA.md` §5
- `llm_client.py` — thin OpenRouter wrapper exposing `complete(messages, tools=None)`
  and `complete_vision(messages, images)`. Provider and model come from config.

**Gate:** `python -c "import config; import llm_client"` works. One live LLM call
returns text. One tool-calling round-trip returns a well-formed tool call.

---

## M1 — Collection

`search_literature`'s fetch half. No indexing yet.

- arXiv query planning: natural language topic → arXiv query syntax, via one LLM
  call, cached by `sha256(topic)`
- arXiv API call, result parsing into paper records
- dedup against manifest on `paper_id`
- **tag backfill on a dedup hit** — appending the new `topic_tag` to the existing
  paper record *and* to every chunk record that paper already produced, in one
  operation, via temp-file-and-rename on `chunks.jsonl`. Chunks denormalise
  `topic_tags`, so patching the manifest alone silently breaks `topic_filter` for
  that paper (`docs/DATA_SCHEMA.md`, **Topic naming**)
- `topic_tags` asserted to be `list[str]` at record construction — a bare string
  passes `in` character-by-character and quietly turns `topic_filter` into a
  substring match
- PDF download to `data/papers/`

**Gate:** fetch 3 papers on any topic. Manifest has 3 well-formed paper records. Run
it a second time with the same topic — 0 downloads, 3 skips. Then run a **second,
overlapping topic** and confirm a re-encountered paper carries both tags in the
manifest *and* in every one of its chunk records — not just the manifest. Kill the
network and confirm the failure returns an error dict rather than raising.

---

## M2 — Extraction and chunking

- PyMuPDF text extraction with page numbers
- section detection where headings are findable, `null` otherwise
- chunking with overlap, respecting `chunking.max_tokens` — counted with the
  bi-encoder's tokenizer, and derived as the min of the bi-encoder and cross-encoder
  chunk budgets. Which of the two binds depends on the models in use; re-derive rather
  than assume (see `docs/ARCHITECTURE.md` §5)
- figure/table image extraction, filtered by minimum dimensions
- caption extraction (regex on `Figure N:` / `Table N:` near the image)
- vision-LLM description, cached by `image_hash`
- within-paper chunk dedup on `content_hash`, then `position` assigned — in that
  order, so `position` is dense. Never dedup across papers (`docs/DATA_SCHEMA.md` §2)
- chunk records assembled per `docs/DATA_SCHEMA.md` §2

**Gate:** run over the 3 papers from M1. Print 5 random chunks and eyeball that the
text is coherent, not header/footer garbage. Confirm figure chunks have
`caption + description` in `text`, not description alone. Confirm `position` is
contiguous. Re-run and confirm zero vision calls (cache hit).

---

## M3 — Index

- bi-encoder embedding of all chunks, reusing `data/cache/embeddings.json` on a
  `content_hash` hit so shared text across papers costs one embed, not two
- FAISS flat index
- `faiss_id_map` and `embeddings.npy` written in lockstep with the index
- incremental append path — new chunks only, never rebuild
- `embedding_model` / `embedding_dim` mismatch detection on load

**Gate:** build the index, restart the process, load it, run a raw similarity search,
and confirm returned `chunk_id`s resolve to sensible text. This specifically tests
that `faiss_id_map` survived the restart. Then add a 4th paper and confirm the
existing 3 were not re-embedded.

---

## M4 — Retrieval

`retrieve_evidence`, complete.

- bi-encoder retrieve at `k_retrieve` (30–50)
- metadata filter on `topic_tags` and `chunk_types`
- cross-encoder rerank **over candidates only** — never over the index
- relevance gate → `sufficient_evidence`
- neighbour expansion via `position ± 1`
- per-chunk truncation, cross-iteration dedup

**Gate:** three queries — one clearly answerable from the corpus, one clearly
absent, one borderline. The absent one must return `sufficient_evidence: false`.
Log rerank latency. Confirm the cross-encoder ran on the shortlist, not the corpus.

The gate threshold is a placeholder at this point. It gets tuned in M8 against the
eval set, not guessed now.

---

## M5 — Agent loop

The core of the system. One tool wired in (`retrieve_evidence`), nothing else.

- tool registry: schema generation, dispatch by name
- loop: call model → if tool_use, execute → append result → repeat
- iteration cap
- loop detection on repeated `(tool_name, args)`
- trace logging per `docs/DATA_SCHEMA.md` §7
- system prompt loaded from `src/prompts/system.md`

**Gate:** a test with a stubbed LLM asserting dispatch, cap enforcement, and loop
detection — no network. Then one real question end to end, with the trace file
inspected by hand.

---

## M6 — Remaining tools

Add one at a time, each with a smoke test asserting return shape, in this order:

1. `search_literature` — wire M1+M2+M3 behind the tool interface
2. `analyze_corpus` — `stats` and `timeline` first (pure pandas), then `cluster`
   (KMeans, silhouette-selected `k`, TF-IDF labels, ARI + purity against
   `topic_tags`), cached on `manifest_hash`
3. `inspect_figure` — indexed figures first, user-supplied images second
4. `check_evidence_consistency` — `contradiction` mode, then `groundedness` mode
   with in-tool claim extraction

**Gate per tool:** runs standalone, returns the documented shape, returns an error
dict on bad input rather than raising. After all four: one question that should
naturally exercise each, with the trace checked to confirm the model actually
selected it.

---

## M7 — Corpus and eval set

- index two contrasting eval topics (adjacent vocabulary, distinct claims), into the
  **same** index, distinguished by `topic_tags`
- write 12–15 questions in `eval/questions.json` covering all four categories
- **write the questions from paper abstracts, before running the system on them**
- hold out roughly a third; do not look at held-out results until M8 tuning stops

**Gate:** every question has gold labels. No question copies paper wording verbatim.
All four categories are represented, including at least two `not_in_corpus` and two
`conflicting`.

---

## M8 — Eval harness

`eval/` imports `src/`, never the reverse.

- runner: each question through the full agent, traces captured
- retrieval metrics: recall@k, MRR against `gold_chunk_ids`
- answer scoring against `expected_facts`
- abstention correctness for `not_in_corpus`
- groundedness ratio
- tool-trace comparison against `expected_tools` (soft — mismatches get flagged for
  review, not auto-failed)
- **rerank ablation:** same eval set with rerank on and off, quality and latency

Tune the relevance gate threshold here, on the non-held-out questions only.

**Gate:** full run produces a results table. Then run the held-out set once and
report it separately.

---

## M9 — Deliverables

- `README.md` — setup, env vars, how to run indexing / a question / the eval
- cold-start demo: a topic never indexed before, run end to end, transcript saved
- example questions and outputs, including one abstention and one conflicting-evidence
  case
- eval results written up, **including results that came out worse than hoped** —
  a rerank ablation showing a marginal gain at this corpus size is a finding, not
  something to bury
- the "how AI agents were used" section in the architecture doc, filled in with
  concrete examples of what was caught in review

---

## Standing rules

- Update the "Things that have already gone wrong once" list in `.claude/CLAUDE.md`
  whenever something is caught in review. That list is live, not decoration.
- If a spec decision cannot be implemented as written, stop and say so with the
  reason. Do not silently substitute an alternative.
- No topic name in `src/` outside `eval/`. Topics are runtime arguments.