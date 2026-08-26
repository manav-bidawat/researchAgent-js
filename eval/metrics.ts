/**
 * Scoring functions for the eval: retrieval recall, MRR, fact coverage, abstention.
 *
 * In:  what the system retrieved or answered, and the gold labels for that question.
 * Out: plain numbers. No model calls except bi-encoder passage vectors for fact coverage
 *      (fetched via the bridge), so a run is reproducible and a sweep is comparable.
 */

import type { PythonBridge } from "../app/bridge/client.js";

/** Fraction of gold chunks in the first k retrieved. NaN with no gold (abstention case). */
export function recallAtK(retrieved: string[], gold: string[], k: number): number {
  const goldSet = new Set(gold);
  if (goldSet.size === 0) return Number.NaN;
  const top = new Set(retrieved.slice(0, k));
  let hits = 0;
  for (const id of goldSet) if (top.has(id)) hits += 1;
  return hits / goldSet.size;
}

/** 1/rank of the first gold chunk, or 0 if none was retrieved. */
export function reciprocalRank(retrieved: string[], gold: string[]): number {
  const goldSet = new Set(gold);
  if (goldSet.size === 0) return Number.NaN;
  const index = retrieved.findIndex((id) => goldSet.has(id));
  return index === -1 ? 0 : 1 / (index + 1);
}

/**
 * Whether the right paper was reached, even if not the exact chunk. The gold chunk set is
 * hand-labelled and cannot be exhaustive, so a neighbouring chunk is often a real hit.
 */
export function anyGoldPaperHit(retrievedPapers: string[], goldPapers: string[]): boolean {
  const gold = new Set(goldPapers);
  return retrievedPapers.some((paper) => gold.has(paper));
}

const ABSTAIN_MARKERS = [
  "cannot answer", "can't answer", "no evidence", "not in the corpus",
  "does not contain", "do not contain", "doesn't contain", "no information",
  "not covered", "unable to answer", "insufficient evidence", "nothing in the",
  "not present in", "no relevant", "does not appear",
];

/** Whether an answer says it cannot answer, rather than answering. */
export function looksLikeAbstention(answer: string): boolean {
  const lowered = (answer ?? "").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  return ABSTAIN_MARKERS.some((marker) => lowered.includes(marker));
}

/**
 * Did the system abstain exactly when it should have? The gate firing is the mechanism,
 * but the failure being tested is the model answering anyway from parametric knowledge.
 */
export function abstentionCorrect(answer: string, sufficientEvidence: boolean, expectAbstention: boolean): boolean {
  const stated = looksLikeAbstention(answer);
  return expectAbstention ? stated || !sufficientEvidence : !stated;
}

export interface FactCoverage {
  covered: number;
  total: number;
  ratio: number;
  per_fact: { fact: string; similarity: number; covered: boolean }[];
}

const round = (value: number, places = 4): number => Number(value.toFixed(places));

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

/** Split into sentences the way metrics.py did, keeping those of three words or more. */
export function answerSentences(answer: string): string[] {
  return (answer ?? "")
    .split(/(?<=[.!?])\s+|\n+/)
    .filter((sentence) => sentence.split(/\s+/).filter(Boolean).length >= 3);
}

/**
 * How many expected facts the answer states, by embedding similarity between each fact and
 * its best-matching answer sentence. Facts are paraphrases, so substring matching would
 * score correct answers as wrong. Vectors come back normalised, so a dot is a cosine.
 */
export async function factCoverage(
  bridge: PythonBridge,
  answer: string,
  expectedFacts: string[],
  threshold = 0.62,
): Promise<FactCoverage> {
  const facts = expectedFacts.filter((fact) => fact.trim());
  if (facts.length === 0) return { covered: 0, total: 0, ratio: Number.NaN, per_fact: [] };

  const sentences = answerSentences(answer);
  if (sentences.length === 0) {
    return {
      covered: 0, total: facts.length, ratio: 0,
      per_fact: facts.map((fact) => ({ fact, similarity: 0, covered: false })),
    };
  }

  const { vectors } = await bridge.call<{ vectors: number[][] }>("embed_passages", { texts: [...facts, ...sentences] });
  const factVectors = vectors.slice(0, facts.length);
  const sentenceVectors = vectors.slice(facts.length);

  const perFact = facts.map((fact, index) => {
    const best = Math.max(...sentenceVectors.map((sentence) => dot(factVectors[index] ?? [], sentence)));
    return { fact, similarity: round(best), covered: best >= threshold };
  });
  const covered = perFact.filter((entry) => entry.covered).length;
  return { covered, total: facts.length, ratio: round(covered / facts.length), per_fact: perFact };
}

export interface ToolTraceMatch {
  expected: string[];
  used: string[];
  missing: string[];
  unexpected: string[];
  matched: boolean;
}

/** Tools actually called against the ones anticipated. Soft by design: divergence is logged. */
export function toolTraceMatch(used: string[], expected: string[]): ToolTraceMatch {
  const usedSet = new Set(used);
  const expectedSet = new Set(expected);
  return {
    expected: [...expectedSet].sort(),
    used: [...used],
    missing: [...expectedSet].filter((tool) => !usedSet.has(tool)).sort(),
    unexpected: [...usedSet].filter((tool) => !expectedSet.has(tool)).sort(),
    matched: [...expectedSet].every((tool) => usedSet.has(tool)),
  };
}

/** Mean over the defined values, ignoring NaN; NaN when nothing is defined. */
export function mean(values: number[]): number {
  const usable = values.filter((value) => !Number.isNaN(value));
  if (usable.length === 0) return Number.NaN;
  return round(usable.reduce((sum, value) => sum + value, 0) / usable.length);
}
