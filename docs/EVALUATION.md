# Evaluation

What was measured, how, and what the numbers actually say — including the ones that
came out worse than hoped. `docs/ARCHITECTURE.md` §13 summarises the design; this is
the results document it points at.

All numbers below were produced at M8 on **2026-09-09** and are reproduced verbatim
from `eval/results/*.json`. They are not re-run for this document on purpose: the
agent model is a hosted service and `eval/build_corpus.ts` queries a live arXiv, so a
re-run would produce different numbers and leave the prose disagreeing with the JSON
beside it.

## The corpus these numbers describe

11 papers · 406 chunks · 48 figures. Two deliberately **adjacent** topics, indexed into
one pool so cross-topic distractors are present by construction — trivially disjoint
topics (astrophysics vs. baking) would make retrieval look good for a reason that has
nothing to do with the system.

| topic tag | query |
|---|---|
| `sparse_mixture_of_experts_routing_in_transformer_models` | sparse mixture-of-experts routing in transformer models |
| `structured_pruning_and_quantization_for_efficient_transforme` | structured pruning and quantization for efficient transformer inference |

<details>
<summary>The 11 papers (arXiv ids as stored)</summary>

`2310_00811v1` Sparse Backpropagation for MoE Training ·
`2410_18035v1` MiLoRA ·
`2412_11629v1` QPruner ·
`2502_01705v4` Progressive Binarization with Semi-Structured Pruning ·
`2503_06184v1` Sample-aware Adaptive Structured Pruning ·
`2505_18232v4` Two-Stage Regularization-Based Structured Pruning (TRSP) ·
`2506_18952v2` LLMs on a Budget? Say HOLA ·
`2605_17598v1` Mixture of Experts for Low-Resource LLMs ·
`2607_08027v1` Structured Pruning via Power Transform ·
`2607_16427v1` Multi-level Context Modeling for Expert Selection ·
`2608_15102v2` A Declarative-Procedural Perspective on Expert Routing

</details>

## Question set and splits

15 questions in `eval/questions.json`, hand-annotated with gold chunks, gold papers,
required facts, and the tools a reasonable agent should reach for. Five categories:
single-paper, multi-paper, conflicting, figure-dependent, and not-in-corpus.

**Ten questions tune, five are held out.** The threshold sweep and every other tuning
decision touched only the first ten. The held-out five were run once, at the end, and
are reported separately below. That is the only defensible way to report a threshold
that was itself chosen from data.

## Results

| metric | tuning (10 q) | held out (5 q) |
|---|---|---|
| fact coverage | **1.000** | **0.917** |
| abstention accuracy | **1.000** | **1.000** |
| — abstains on absent topics | 1.000 | 1.000 |
| — false abstention on answerable | 0.000 | 0.000 |
| citations resolved | **1.000** | **1.000** |
| gold-paper hit rate | **1.000** | **1.000** |
| MRR | 0.660 | 0.458 |
| recall@1 / @3 / @5 | 0.088 / 0.175 / 0.238 | 0.042 / 0.208 / 0.250 |
| tool trace matched | 0.900 | 1.000 |
| harness errors | 0 | 0 |

The generation-side metrics hold up on held-out data. The retrieval-side ones drop
(MRR 0.660 → 0.458), which is what a 10-question tuning split should be expected to do
— it is a small sample and the threshold was fitted on it.

### Why recall@5 is low while gold-paper hit rate is 1.000

These look contradictory. Part of the gap is a ceiling effect, but — measured rather
than assumed — **less of it than is comfortable.**

Gold sets hold 5 or 6 chunks each (mean 5.92 over the 12 answerable questions; the three
`not_in_corpus` questions have no gold chunks and are scored by abstention instead).
`recall@k` counts gold chunks among the first *k* chunks the agent saw across the whole
conversation, so with `k=5` against ~6 gold chunks the ceiling is **0.847**, not 1.0.

Measured recall@5 is 0.238. Against a ceiling of 0.847 that is roughly 28% of what was
achievable — so the ceiling explains some of the gap and **not most of it**. The first
five chunks the agent sees contain about 1.4 of the ~6 chunks an annotator marked as
supporting.

What keeps this from being a failure is what the same runs show alongside it: gold-paper
hit rate 1.000, fact coverage 1.000/0.917, and every citation resolving. The agent is
reliably reaching the right papers and producing correct, supported answers — it is
just not reaching the *specific* chunks the annotation names, in the first five. Two
things drive that, and `eval/metrics.ts` flags the first in its own docstring:

- **Gold sets cannot be exhaustive.** A neighbouring chunk from the right paper often
  supports the claim just as well and is scored as a miss.
- **Chunk boundaries are arbitrary** with respect to where a fact sits. Section-aware
  chunking at ~350 tokens splits an argument across chunks that a human annotator
  labelled as one.

Read recall@5 as a relative signal between configurations, never as an absolute — and
read gold-paper hit rate as the metric that reflects whether the system worked.

## Relevance threshold sweep

Cross-encoder scores are uncalibrated logits spanning roughly −11..+11, so the gate
threshold can only come from data. Swept −8..+2 on the tuning split, scored by harmonic
mean so a threshold cannot win one half by abandoning the other:

| threshold | answers when answerable | abstains when absent | balanced | gold-paper hit |
|---|---|---|---|---|
| −8.0 | 1.000 | 0.500 | 0.667 | 1.000 |
| −6.0 | 1.000 | 0.500 | 0.667 | 0.875 |
| −4.0 | 0.875 | 0.500 | 0.636 | 0.875 |
| **−3.0** | **0.875** | **1.000** | **0.933** | 0.875 |
| −2.0 | 0.875 | 1.000 | 0.933 | 0.875 |
| −1.0 | 0.750 | 1.000 | 0.857 | 0.750 |
| 0.0 | 0.500 | 1.000 | 0.667 | 0.500 |
| +2.0 | 0.125 | 1.000 | 0.222 | 0.125 |

`0.0` was the placeholder the code shipped with before tuning. It refuses **half** the
answerable questions — a bug that no unit test could have caught, because the gate was
behaving exactly as written. −3.0 and −2.0 tie; −3.0 was taken as the more permissive of
the two.

## The rerank ablation — a marginal gain, and what the ablation missed

This is the result that came out worse than hoped, and then turned out to be measuring
the wrong thing. Both halves are recorded here, because the second half only surfaced
by acting on the first.

Same 10 questions, retrieval only, rerank on vs. off:

| | recall@1 | recall@3 | recall@5 | MRR | gold-paper hit | median latency |
|---|---|---|---|---|---|---|
| rerank **on** | 0.042 | 0.083 | 0.125 | 0.344 | 0.875 | 1548 ms |
| rerank **off** | 0.042 | 0.063 | 0.108 | 0.317 | **1.000** | **21 ms** |
| delta | 0 | +0.021 | +0.017 | **+0.027** | **−0.125** | **+1527 ms** |

Reranking buys **+0.027 MRR for +1527 ms** — a 74× latency increase. On gold-paper hit
rate it is *worse*, 0.875 against 1.000.

### Both differences are below the noise floor

The ablation scores the 8 answerable questions in the tuning split. At n=8, one
question moving from rank 2 to rank 1 shifts MRR by **0.0625**. The measured difference
is **0.0271** — less than half the smallest move a single question can make. It is not
a small effect; it is sub-resolution.

The gold-paper hit difference is the same story from the other side: 0.875 vs 1.000 is
**7/8 vs 8/8**, exactly one question.

So neither the gain nor the harm is measurable here. The only robust number in the
table is the cost: **74×**, which is not noise.

### Acting on that read was wrong, and the correction is the finding

The obvious conclusion — rerank is not earning its keep at 406 chunks, so turn it off —
was tested before being applied. It does not survive, because **the ablation never
measured abstention.** It reports recall, MRR, latency and gold-paper hit. It says
nothing about whether the system still refuses questions the corpus cannot answer.

The relevance gate is what produces abstention, and it thresholds whatever score the
retrieval stage produced. With rerank on that is a cross-encoder logit (−11..+11, gated
at −3.0). With rerank off it is a cosine (0..1), and every cosine clears −3.0 — so
turning rerank off silently disables the gate entirely:

```
rerank=True   ABSENT topic ("sourdough fermentation")  -> 0 chunks, sufficient=False
rerank=False  ABSENT topic ("sourdough fermentation")  -> 5 chunks, sufficient=True
```

Giving the bi-encoder its own tuned threshold does not recover it. Swept 0.0–0.80 on the
same 10 tuning questions (`eval/results/bi_threshold_sweep.json`):

| gate | best threshold | answers when answerable | abstains when absent | balanced |
|---|---|---|---|---|
| bi-encoder (rerank off) | 0.55 | 1.000 | **0.500** | **0.667** |
| cross-encoder (rerank on) | −3.0 | 0.875 | **1.000** | **0.933** |

No cosine threshold reaches the cross-encoder's separation. The thresholds that abstain
correctly (0.75+) collapse answerable coverage to 0.125 — refusing 7 of 8 answerable
questions. The ranges genuinely overlap: an absent topic scores 0.57, while real
answerable questions sit at 0.55–0.65.

That is a principled result rather than a tuning accident. Cosine measures overlap
between two independently-encoded vectors. The cross-encoder attends over query and
passage together, so it can represent "this passage does not answer this question" —
a judgement a dot product between separate encodings cannot make.

**Rerank stays enabled.** Not because the architecture says so, but because the metric
it earns its cost on is abstention, and the ablation simply never looked there. What
the MRR ablation does establish is narrower and still true: at this corpus size,
reranking does not measurably improve *ordering*. It is paying 1527 ms for the gate,
not for the ranking.

Two things this leaves open, neither done:
- **The ablation should measure abstention.** It is the metric rerank actually carries,
  and it is absent from `eval/ablation.ts`.
- **`k_retrieve` is untouched at 40.** Rerank cost scales with the shortlist, so a
  smaller pool may keep the gate at a fraction of the latency. That is a measurement
  nobody has taken.

A latent bug surfaced on the way, independent of this decision. `retrieve_evidence`
falls back to bi-encoder scores when the reranker fails at runtime, but kept gating at
the cross-encoder's −3.0 — so a reranker that failed in production would silently
disable abstention with no error. The gate now selects its threshold by which scorer
actually ran, and `bi_encoder_relevance_threshold` is tuned to 0.55 for that path, with
its limits documented above.

## What else the numbers do not say

- **n is small.** 10 tuning and 5 held-out questions. A single question moves fact
  coverage by 0.1, and every metric here should be read with that in mind.
- **One annotator.** Gold chunks and required facts were annotated once, by the same
  person who wrote the questions. No inter-annotator agreement was measured.
- **Tool trace matched 0.900 on tuning**, and the one miss is instructive: on `cf02`
  ("is expert specialisation coherent or unstable?") the agent answered from two
  retrieval calls and never ran `check_evidence_consistency`, which the annotation
  expected. It reached a well-cited, correct answer anyway (fact coverage 1.0, 9/9
  citations resolved) — but it got there without the verification step the system
  prompt asks for on multi-source answers. Nudging that behaviour is prompt work that
  has not been done.
- **Temperature is 0.0**, but the provider is not bit-reproducible, so a re-run will
  vary somewhat regardless.
- **The corpus is not frozen.** `eval/build_corpus.ts` queries live arXiv; the same
  topics will return a different 11 papers later.

## Reproducing

```bash
npx tsx app/cli.ts eval              # tuning split
npx tsx app/cli.ts eval --held-out   # held-out split
npx tsx scripts/m8_gate.ts           # both splits plus the rerank ablation
```

Threshold sweep: `npx tsx eval/tune_threshold.ts`. Ablation alone: `npx tsx eval/ablation.ts`.
Results land in `eval/results/`.
