# Architecture & Design Choices

## 1. What this system is

A topic-agnostic agentic research assistant over scientific papers. Given a natural
language question, it decides for itself whether it needs to fetch new literature,
retrieve evidence from what it already has, look at a figure, check evidence for
contradictions, or compute statistics over the corpus — and then answers with
inline, traceable citations.

**Core design commitment:** the agent is not a fixed pipeline. Tools are exposed
with clear boundaries and the model chooses the sequence. A question that needs no
new papers should never trigger a fetch; a question with weak evidence should
trigger either a re-query or an explicit abstention.

## 2. On the choice of topic

The task statement allows any scientific topic. We deliberately do **not** hardcode
one into the system.

A fixed-topic system collapses into a plain RAG pipeline: the paper set is frozen,
`search_literature` is never called, and the "agent decides which tools to use"
requirement becomes vacuous. So:

- **The agent is topic-agnostic.** Any topic can be given at runtime; it fetches,
  indexes, and answers.
- **Two contrasting topics are used for evaluation only**, to give the eval harness a
  stable, known corpus against which gold evidence can be labelled.

The two eval topics are chosen to be *adjacent but distinct* — they share surface
vocabulary but have separate claims and methods. Trivially disjoint topics (e.g.
astrophysics vs. cooking chemistry) would make retrieval look good for the wrong
reason.

Both topics live in **one index**, tagged via each record's `topic_tags`, rather than in two
separate stores. A mixed pool is a more honest test of retrieval: cross-topic
distractors are present by construction, which is what a real index looks like.

A **cold-start demo** is included: a topic never seen before is given, and the agent
runs fetch → index → answer end to end. This is the evidence that the topic choice
is an evaluation artefact, not an architectural one.

## 3. Pipeline

```
user question
      |
      v
  agent loop  <---------------------------------+
      |  (model picks tools, max N iterations)   |
      +--> search_literature      arXiv -> PDF -> index
      +--> retrieve_evidence      bi-encoder -> rerank -> gate -> expand
      +--> inspect_figure         raw image to a vision model
      +--> check_evidence_consistency   NLI: contradiction / groundedness
      +--> analyze_corpus         pandas + KMeans over metadata
      |                                          |
      +------------------------------------------+
      |
      v
  answer with inline per-claim citations
```

## 4. Collection

**Source: arXiv only.** Open access, no auth, PDFs directly downloadable, clean
metadata, strong CS/ML/physics coverage. IEEE / Elsevier / Springer are paywalled —
their APIs need institutional subscriptions and do not allow programmatic full-text
download, so they were ruled out rather than half-implemented.

**LLM-planned queries.** The user's natural language topic is not passed to arXiv
verbatim. A small LLM call translates intent into arXiv query syntax (field prefixes,
boolean operators, category filters). This is what makes collection "automated as
much as possible" rather than "the user must know arXiv query syntax".

Parameters: `query`, `max_results` (default 8), `categories` (optional),
`sort_by` (relevance default).

**One HTTP seam, one clock.** `src/corpus/arxiv_fetch.py` is the only module that
talks to arXiv over HTTP. It exists because arXiv rate-limits by IP and answers 429
for *every* query once tripped, while the `arxiv` package splits its requests across
two unrelated paths: `Client` paces feed requests against its own `_last_request_dt`,
and `Result.download_pdf` calls `urllib.request.urlretrieve` directly, with no delay,
no shared state and no identifying User-Agent. One search plus eight downloads was
therefore one paced request and eight unpaced ones. Downloads now borrow the client's
session and its clock, so the configured delay spans both kinds of request.

A 429 aborts rather than retries — `RateLimited` deliberately sits outside the
exception set `Client._parse_feed` retries on, so the library cannot spend three more
requests on a host that is refusing all of them — and the expiry is recorded in
`paths.arxiv_cooldown` and checked before the next search. That memo is the only
cache here that stores a refusal rather than a result; like the others it is
disposable, and deleting it only means the next call learns the same fact from arXiv.

**Deduplication and caching, kept separate:**

| Level | Key | Effect | Kind |
|---|---|---|---|
| Paper | `arxiv_id` in manifest | skips download and processing | dedup — the record is never created twice |
| Chunk, within one paper | `sha256(text)` | drops that paper's repeated chunk | dedup |
| Chunk, across papers | `sha256(text)` | reuses the cached vector | **cache only** — both chunk records survive |
| Figure | `sha256(image bytes)` | reuses the generated description | cache only |

The chunk row is split deliberately. Two papers sharing a paragraph is a fact about
the corpus, not a duplicate to be cleaned up: dropping the second copy deletes content
from that paper's reading order and breaks the `position ± 1` chain that neighbour
expansion walks — silently, and unrecoverably at query time. So across papers the hash
buys a cheaper embedding and nothing else, and `position` is assigned after dedup so it
stays dense by construction.

Caches are **content-hash keyed, not path keyed**, so the same paper arriving via a
second topic costs nothing.

## 5. Processing & indexing

**Text.** PyMuPDF for extraction (good layout handling, and it also exposes embedded
images from the same parse). Chunking is section-aware where section headers are
detectable, falling back to fixed-size with overlap.

**Chunk size is bounded by two model limits, and the doc records both rather than the
single number that happens to be binding today.** The binding constraint moves whenever
either model changes — and it already has: the sentence "the ceiling is the
cross-encoder's 512" was true of the original stack and went false the moment
`all-MiniLM-L6-v2` was swapped out. A single number gives a later reader no way to
notice it has gone stale. A table does.

| Stage | Model | Window | Chunk budget | If exceeded |
|---|---|---|---|---|
| Bi-encoder (embed) | `bge-small-en-v1.5` | 512, chunk alone | `512 - 2` specials = **510** | Tail is never embedded, so it is unfindable at any `k`. Silent and unrecoverable. |
| Cross-encoder (rerank) | `ms-marco-MiniLM-L-6-v2` | 512, **shared** `[CLS] query [SEP] chunk [SEP]` | `512 - 3 - max_query_tokens(64)` = **445** | Tail is missing only while reordering an already-retrieved shortlist. Recoverable, and far cheaper. |

**Binding today: the cross-encoder, at 445**, because its window is shared with the
query rather than given to the chunk alone. `chunking.max_tokens` is set to exactly
that. No safety margin is subtracted: `retrieval.max_query_tokens` is a truncation
point, not an estimate, so the query cannot overrun its half of the budget.

Why the bi-encoder row matters even though it is not binding: it is the row that was
being ignored. `all-MiniLM-L6-v2` caps at **256** tokens, not 512, so 350-token chunks
would have lost roughly a quarter of their text before ever reaching the index —
truncation at the stage that decides whether a chunk is *findable at all*, not merely
how it ranks. The bi-encoder is therefore `bge-small-en-v1.5`: 512 tokens at the same
384 dimensions, so the index shape is unchanged.

The chunker counts tokens with the bi-encoder's tokenizer, not a word-count
approximation. That is safe for the cross-encoder's budget too because the two models
share a vocabulary (both `bert-base-uncased`, 30522 entries) and were measured to
produce identical token counts on the same text. If either model is ever swapped for
one with a different vocabulary, that equivalence has to be re-checked — it is an
observation about this pair, not a general fact.

**Figures and tables.** Both are treated as images. Tables are not parsed
structurally — PyMuPDF does not extract them reliably, and a second table-specific
library (pdfplumber/camelot) is not worth the time budget for the marginal gain.
Instead:

1. extract image region + nearby caption (regex on `Figure N:` / `Table N:`)
2. filter out decorative images below a minimum dimension threshold
3. one vision-LLM call produces a description
4. **embed `caption + description` concatenated**, not the description alone —
   the caption carries the authors' precise terminology, which a generated
   description often paraphrases away
5. store `paper_id`, `page`, `figure_id`, and the image path in metadata so the
   raw image can be served back later by `inspect_figure`

Figure count per paper is capped to keep indexing cost bounded.

**Embeddings & store.** `sentence-transformers` bi-encoder + FAISS flat index.

BGE is trained **asymmetrically**, so encoding is not symmetric either: the query
instruction (`embedding.query_prefix`) is prepended when encoding a *query* and never
when encoding a *passage*. Applying it to both sides, or to neither, measurably
degrades retrieval, so the prefix belongs to the query-encode path only — it is a
property of the retrieval call, not of the stored chunk. Vectors are L2-normalised,
which is what makes inner-product search equal cosine similarity.

Flat is correct at this scale — 16 papers is roughly 1k chunks, under 2 MB of
float32 vectors. FAISS internal ids are sequential, so an explicit
`faiss_idx -> chunk_id` mapping is persisted alongside the index; without it the
mapping breaks on restart.

**Incremental, never rebuild.** New chunks are appended via `index.add()`. A full
rebuild is only triggered if the embedding model changes.

## 6. Index growth management

The real risk of an ever-growing corpus is **not** memory — it is retrieval quality
degradation, as unrelated content becomes distractors.

- **`topic_tags` metadata filtering** keeps the effective search space scoped even as
  the index grows. One index, scoped retrieval.
- **`max_papers_per_topic` cap** (default 10) bounds growth predictably.

Those two solve the stated problem between them. Filtering bounds the *distractor
set* a query actually competes against, and the cap bounds *how fast the pool grows*.
Neither needs anything removed from the index.

**No eviction. Explicit non-goal.** An earlier draft of this section carried an
`evict_topic(topic_tag)` LRU hook. It is removed, for two reasons:

1. **It solves a problem this system does not have.** Eviction reclaims *disk and
   memory*. Disk is not a constraint at 16 papers, and this section's opening sentence says
   memory is not the risk — retrieval quality is, and quality is handled by filtering,
   which eviction does not improve.
2. **It contradicts the incremental design.** `IndexFlat.remove_ids` compacts by
   swapping the last vector into the freed slot. That silently invalidates every
   downstream entry of the positional `faiss_id_map` and every downstream row of
   `embeddings.npy`, so a correct eviction means rebuilding the index — against the
   "incremental, never rebuild" commitment in §5. Keeping the hook would
   have meant either a rebuild path nothing else needs, or a mapping that goes wrong
   quietly at query time.

**Scale path, not implemented at this size.** Three drop-in upgrades, in the order
they would become worth doing:

| Change | Buys |
|---|---|
| `IndexIDMap` / `IndexIDMap2` around the flat index | Stable caller-assigned int64 ids, so `faiss_id_map` stops being positional and `remove_ids` becomes safe. This is the prerequisite for eviction ever being reconsidered, and the reason eviction is deferred rather than declared impossible. |
| `float32 -> float16` | Halves vector memory at negligible quality cost. |
| `IndexIVFPQ` | 10–50x compression, at the cost of approximate search and a training step. |

Flat plus a positional map is correct at this scale, and each upgrade is a local
change to the index layer.

## 7. Retrieval design

**Retrieve wide, pass narrow.**

```
bi-encoder  k=30-50   (cheap, high recall)
     -> cross-encoder rerank   (expensive, high precision, only on candidates)
     -> relevance gate
     -> neighbor expansion
     -> top 5-6 to the agent
```

Recall lives in the candidate pool; only precision reaches the context window.

**Cross-encoder reranking.** A bi-encoder embeds query and chunk separately and
compares two vectors — the model never sees them together. A cross-encoder scores
`[CLS] query [SEP] chunk [SEP]` in one pass with full cross-attention, which is far
more accurate but cannot be precomputed. Two-stage retrieval gets both.

Reranking is presented here as **standard retrieval hygiene, not as the ML
component** — it is a pretrained checkpoint used off the shelf. Its contribution is
measured honestly via ablation (see Evaluation). At this corpus size the gain may be
small; that result is reported as-is rather than hidden.

Cross-encoder scores are **not calibrated** and are used for ranking only. Where a
threshold is needed (the relevance gate) it is tuned empirically on the eval set, not
assumed.

**Relevance gate.** Naive top-k always returns *something*, even when nothing is
relevant. The gate compares top scores against a tuned threshold and returns
`sufficient_evidence: false` with a reason instead of handing back weak chunks. This
turns "I don't have evidence for that" from a prompt-level instruction into an
actual mechanism, and is what the *not in corpus* eval category tests.

**Neighbor expansion.** Adjacent chunks (`±1` within the same paper) are attached to
selected chunks, recovering information severed at chunk boundaries without
increasing `k`.

**Context discipline.** Tool results are truncated (~500 tokens/chunk), returned as
structured records rather than raw text blobs, and deduplicated across iterations so
repeated queries do not stack duplicate chunks into context. The loop has an
iteration cap (6–8) plus loop detection on repeated `(tool, args)` pairs.

## 8. ML / data-analysis components

**1. NLI-based evidence checking** (`check_evidence_consistency`), in two modes:

- *contradiction* — do retrieved chunks disagree with each other?
- *groundedness* — is each claim in the drafted answer entailed by retrieved
  evidence? `neutral` labels surface claims the model produced from parametric
  knowledge rather than from the corpus.

This is the mechanism behind the "robustness to conflicting evidence" requirement —
a computed signal rather than a prompt instruction.

**Claim extraction happens inside the tool, not in the agent.** If the agent
submitted its own claim list it could self-select the claims it knows are supported.
The tool takes `answer_text` and decomposes it itself.

**Known limitations, stated rather than papered over:** NLI models are trained on
sentence pairs and degrade on paragraph-length inputs, so chunks are split and the
best-matching sentence is used as premise. General MNLI-trained models are weak on
scientific text — a `neutral` label can mean "unsupported" or "the model didn't
understand". Directionality matters (premise = evidence, hypothesis = claim).
Consequently this is an **advisory signal, not a hard gate**: low groundedness
prompts the agent to revise or flag uncertainty, it does not auto-reject.

**2. Corpus clustering with validation** (`analyze_corpus`, `cluster` operation).
KMeans over chunk/paper embeddings, with `k` selected by silhouette score, and
cluster labels derived from top TF-IDF terms. Crucially it is **validated**: cluster
assignments are compared against known `topic_tags` labels via adjusted Rand index and
purity. This doubles as a diagnostic — if clusters do not recover the known topic
split, the embeddings are not carrying semantic signal, which is a real finding about
the retrieval stack rather than a decorative plot.

Cluster results are cached against a manifest hash and recomputed only when the index
changes.

## 9. Multimodal handling

Two paths, both real:

1. **Indexed path** — figures/tables described at index time, embedded, and
   retrievable as `chunk_type: figure|table` alongside text.
2. **Inspection path** — when a retrieved figure needs detail the description lost
   (exact values, trend shape, axis labels), the agent calls `inspect_figure` and the
   raw image reaches a vision model. Additionally, a **user-supplied image** can be
   inspected and used to drive a follow-up evidence search.

**How the image reaches the model, and why it looks indirect.** `inspect_figure`
returns a structured dict containing an `image_path` — never image bytes. The *agent
loop* then attaches the image to the conversation on a **follow-up user turn**, via
`llm_client.attach_images`, and the model reads it there.

This is a provider-schema constraint, not a stylistic preference, and it is written
down here so a later milestone does not "simplify" it back into the tool result. LLM
access goes through OpenRouter's OpenAI-compatible `/chat/completions` (§11), where a
`tool` message's `content` is a string and `image_url` parts are only valid on a `user`
message. Anthropic's *native* API does accept image blocks inside `tool_result`, but
depending on that would make the provider a code dependency rather than the config
value §11 requires it to be. Returning base64 from the tool would also have made
`inspect_figure` the one tool that hands back a raw blob instead of a structured dict.

The resulting turn order is fixed:

```
assistant(tool_call: inspect_figure)
  -> tool(the dict: image_path, caption, stored_description, ...)
  -> user(image_url part, attached by the loop)
  -> assistant(reads the image)
```

The tool still calls no other tool, and control flow still lives in the agent loop.

Only text, figures, and tables are in scope. No audio or video.

## 10. Answer synthesis

Answers carry **inline per-claim citations** (`[chunk_id]`, e.g.
`[2603_11114v1__c0004]`), not a bibliography appended at the end. This makes
groundedness checking mechanical, makes manual eval verification trivial, and means
every claim is individually traceable to its source.

**The citation is the bare `chunk_id`.** An earlier draft specified
`[paper_id:chunk_id]`, which is redundant: `chunk_id` is `{paper_id}__c{position}`, so
that form expands to `[2603_11114v1:2603_11114v1__c0004]` with the paper named twice.
The first live agent run showed why this matters — the model would not write it, and
emitted `[2603_11114v1:c0004]` instead, splitting the id in a way that matches no chunk
in the store. A citation the eval harness cannot resolve is worse than no citation,
because it scores as a hallucinated source. One unambiguous token, which already
carries the paper, avoids the whole problem.

When evidence is insufficient or conflicting, the answer says so explicitly rather
than resolving the conflict silently in favour of one source.

## 11. Model & provider choices

LLM access goes through **OpenRouter** behind a thin `llm_client` interface
(`complete(messages, tools)`), so the provider is a config value, not a code change.

Rationale: tool-calling reliability varies meaningfully between models. Keeping the
client swappable means a tool-selection failure can be diagnosed as a model
limitation versus a prompt problem by switching one setting. Cheap fast models are
used for indexing-time figure description (high call volume, low difficulty); the
agent loop can be pointed at a stronger model if tool selection degrades.

Local models via Ollama were considered and rejected for this time budget —
small local models have notably weaker tool-calling.

## 12. Robustness

Every tool returns structured errors (`{"error": ...}`) rather than raising, so a
failed arXiv call, a corrupt PDF, a timed-out LLM call, or a missing model checkpoint
degrades the loop instead of crashing it.

Failure modes explicitly handled: arXiv API unavailable, PDF parse failure, empty
retrieval, model download failure (TF-IDF/BM25 fallback path), agent looping,
context overflow.

## 13. Evaluation

Detailed in `docs/EVALUATION.md`. Summary of the design:

**Gold labels.** Each eval question is annotated by hand with the paper(s) and
chunk(s) that should be retrieved, plus expected facts. These labels are *never*
shown to the agent — they exist only for the scoring script. The agent runs its
normal loop; the evaluator compares what it retrieved against what it should have.

**Questions are paraphrased**, never copied from paper wording, so retrieval cannot
pass by string matching.

**Four question categories:**

| Category | Tests |
|---|---|
| Single-paper factual | basic retrieval + faithful answering |
| Multi-paper synthesis | combining and citing multiple sources |
| Not in corpus | abstention — does the relevance gate fire, or does it hallucinate |
| Conflicting evidence | does it surface the disagreement instead of silently picking one |

**Metrics:** retrieval recall@k and MRR against gold chunks; answer correctness
against expected facts; groundedness ratio; abstention correctness; and
**tool-call traces** — which tools were called in what order, which is what
distinguishes "right answer" from "right process".

**Ablation:** no-rerank vs. cross-encoder rerank, reported with both quality and
latency.

**Eval hygiene:** questions are written from abstracts *before* seeing system
output, to avoid writing tests the system already passes. A held-out subset is
reserved and run only after tuning stops.

## 14. Deliberate non-goals

- **No multi-agent orchestration.** A single agent with a well-specified tool
  loop is the right shape at this scope; multi-agent adds coordination failure
  modes without adding capability here.
- **No agent framework.** The tool loop is written directly against the provider
  SDK. The structure of the agentic system is itself under evaluation, and a
  hand-written loop shows the design rather than hiding it inside a framework.
- **No human-in-the-loop gates.** This is autonomous QA, not a research workflow
  with approval steps.
- **No UI.** Explicitly out of scope per the task statement.
- **No structural table parsing.** Tables are handled as images; the trade-off is
  documented above.
- **No index eviction.** Growth is bounded by `max_papers_per_topic` and made
  harmless by `topic_tags` filtering. Removal from a flat index with a positional
  id map implies a rebuild, which the incremental design rules out — see §6.	

---

## 15. How AI agents were used to build this

The system was implemented with Claude Code driving the edits, against the specs in
this directory. That arrangement is worth recording honestly, because the failure mode
it produces is specific: **an agent writing code produces plausible code, and plausible
code passes tests written by the same agent.** Almost every real defect below was
caught by something other than a unit test.

### What the arrangement looked like

The specs (`ARCHITECTURE.md`, `TOOLS.md`, `DATA_SCHEMA.md`, `BUILD_PLAN.md`) were
written first and treated as authority. Work went milestone by milestone, M0→M9, each
ending at a **verification gate** — `scripts/mN_gate.ts`, a script that runs the real
pipeline on real data and prints pass/fail per claim. Gates are the reason this
document can cite measurements rather than intentions.

Two standing rules shaped the output more than anything else:

1. **Do not silently substitute a design decision.** Where a spec looked wrong, the
   instruction was to stop and say so rather than quietly implement something better.
   Six specs turned out to be wrong and were *changed in the docs*, not worked around
   in code — because later milestones read the docs, and a stale doc reintroduces the
   bug two milestones later.
2. **Ask when the spec is ambiguous.** A wrong guess that compiles is worse than a
   question.

### What review actually caught

Concrete cases, each with the evidence that exposed it:

**The bi-encoder was silently truncating every chunk.** The spec reasoned about a
512-token limit and `chunking.max_tokens` was sized against the cross-encoder.
`all-MiniLM-L6-v2` — the originally specified embedding model — actually caps at
**256**. Every chunk over that was being cut in half at embed time, the stage that
decides whether a chunk is findable at all, with no error anywhere. Found by reading
the checkpoint's own config rather than trusting the spec's number. Fixed by swapping
to `bge-small-en-v1.5` (512 tokens, same 384 dims) and re-deriving the ceiling from
both encoders: bi-encoder 510, cross-encoder 512−3−64 = **445**, binding.

**Stale vectors under an unchanged `chunk_id`.** `chunk_id` is positional within a
paper, so it survives re-ingest even when the text beneath it changes — which is
exactly what happens when a figure chunk gains a vision description. Deciding what to
re-embed by id left the caption-only vector in the index while the chunk store held
caption+description. Retrieval kept happily matching text the chunk no longer
contained. No exception, no failing test, just quietly worse results. Fixed with a
positional `indexed_hashes` in the manifest, compared against `content_hash`.

**NLI passing its gate for the wrong reason.** `check_evidence_consistency` reported a
1.000-confidence contradiction against a *true* claim. The premise it had picked was
`"Story emphasizes open-ended generation."` — a sentence with nothing to do with the
claim, which was about expert routing. The cause was a misreading of the spec:
"best-matching sentence" had been implemented as *strongest NLI verdict* rather than
*highest semantic similarity*, and over ~30 candidate premises the winner is reliably a
spurious high-confidence contradiction. Fixed with bi-encoder premise selection and a
`min_pair_similarity` floor of 0.75 — measured, not guessed: real conflicts sit at
0.849–0.894, unrelated pairs from the same paper at 0.574–0.586. 254 candidate pairs
dropped to 6, and the positive control still fires at 1.000.

**A gate check that passed because nothing happened.** The M2 gate asserted "a re-run
makes no new vision calls." It passed — against an empty description cache, where zero
calls is trivially true. A check that cannot fail is worse than no check, because it
reads as evidence. Fixed by failing the gate when the cache is empty.

**Cached and uncached queries diverging.** The arXiv query planner folded category
filters in *after* caching, so a cache hit searched a broader query than a cache miss.
Deduplication silently stopped firing. Found by noticing the same paper arriving twice
under one topic. The first fix then double-applied categories — `((q) AND cat) AND cat`
— which was only distinguishable from the correct behaviour after confirming arXiv
itself returns deterministic results (4/4 identical responses) before blaming the code.

**A statistic chosen for the wrong distribution.** Heading detection used the *median*
font size as the body-text baseline. Short caption and table lines dragged the median
below true body size, and an entire abstract was promoted to headings. Replaced with a
character-weighted mode, and the `body + 1.5pt` threshold derived from an actual
font-size distribution: headings 10.96–11.96, contribution lists 9.96, table cells
≤8.97, body 8.97.

**Bibliographies indexed as content.** Every reference entry looks relevant to every
query about the field. Caught by eyeballing sample chunks at a gate — `[Iccv, 2021. 2]`
as an indexed "section" — not by any assertion. Fixed with `extraction.drop_sections`.

**A citation format that could not be resolved.** The spec called for
`[paper_id:chunk_id]` citations. The model emitted `[2603_11114v1:c0004]`, which
matches no chunk, because `chunk_id` already contains the paper id. Zero of the
citations in a sample answer resolved. Changed to a bare `[chunk_id]` in the spec and
the system prompt; resolvable citations went 0 → 4 on the same question, and the eval
now reports 1.000 citations resolved on both splits.

**A tuned threshold that a test had quietly frozen.** `test_weak_chunks_are_withheld`
hardcoded scores against the placeholder threshold of 0.0. When M8 tuned the threshold
to −3.0 the test broke — correctly. The fix was to derive the test's scores from the
configured threshold, so the test asserts the *behaviour* rather than the number.

### What this says about the arrangement

The pattern is consistent: **unit tests written alongside the code confirmed the code
did what it was written to do, and the gates caught what it was written to do being
wrong.** Every defect above surfaced from running the real pipeline on real data and
looking at the output — a printed chunk, a font-size histogram, a citation that
resolved to nothing — rather than from an assertion.

Two habits did most of the work. **Measure instead of guessing:** every threshold in
`config.yaml` carries the measurement that produced it, and the comment explaining the
measurement is frequently longer than the value. **Verify the tool before blaming the
code:** checking that arXiv returned deterministic results, and that a tokenizer's
limit was what the docs claimed, each redirected a debugging session that was pointed
at the wrong layer.

The residual risk is the one this section cannot fully address: the same agent wrote
the code, the tests, and the gates. Gates mitigate it by running real data and printing
output for a human to read, but they do not eliminate it. `docs/EVALUATION.md` reports
what a small, single-annotator question set can and cannot establish.

## 16. The TypeScript layer and the bridge

The engine — everything in `src/`, from collection through the agent loop and its five
tools — is Python, and stays Python: sentence-transformers, FAISS and PyMuPDF have no
equivalents worth trading for. The entry points around it are a TypeScript layer on
Node: the CLI (`app/cli.ts`: `index`, `ask`, `eval`, `graph sync`, `serve`, `mcp`), the
Express demo server (`app/web/server.ts`), and the MCP server (`app/mcp/server.ts`).

```
app/cli.ts ─┬─ app/web/server.ts (Express)      ┐
            └─ app/mcp/server.ts (MCP SDK)      ├─ app/bridge/client.ts
                                                ┘        │  JSON lines, stdin/stdout
                                               bridge/worker.py ── bridge/ops/* ── src/
```

The two halves meet in exactly one place. `app/bridge/client.ts` spawns
`bridge/worker.py` (`$SCIAGENT_PYTHON`, default `python3`; `--config` becomes
`$SCIAGENT_CONFIG`) and exchanges one JSON object per line: a request
`{id, op, params}`, zero or more `{id, event}` progress frames, then `{id, result}` or
`{id, error, detail}`. The worker dispatches from a fixed table of ops in
`bridge/ops/`, not a generic "call any function" hook, so the engine's callable surface
is as reviewable as the tool registry. Tool-level failures still arrive as the engine's
own `{"error", "detail"}` dicts; only bridge failures (worker died, unknown op, op
crashed) surface as exceptions on the TypeScript side. Like `eval/`, `bridge/` lives
outside `src/` and imports it; nothing in `src/` knows it exists.

Why a **long-lived** worker rather than a process per call:

- **Models load once.** The bi-encoder, cross-encoder, NLI model and FAISS index take
  seconds to load and hundreds of MB to hold. A per-request process would pay that on
  every question; the worker pays it on first use and keeps it for the process's life,
  which is what lets `serve` and `mcp` answer at engine speed.
- **One request at a time.** The worker reads stdin in a loop and runs each op to
  completion before the next, so the shared models and index are never used
  concurrently. The client queues calls and they are answered in order. This matches
  the demo's scope (one question at a time) and removes any need for locking in `src/`.
- **Stdout is reserved for the protocol.** The worker duplicates the real stdout for
  frames and points fd 1 at stderr before importing anything, so a stray `print` — from
  the engine or from a native library — cannot corrupt a frame. Engine logs and
  tracebacks go to stderr, which the client forwards. This is also what makes MCP over
  stdio safe: the Node process owns its own stdout, and the worker never writes to it.
