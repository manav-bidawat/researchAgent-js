/**
 * Sweeps the relevance-gate threshold and picks one from data, on the tuning split only.
 *
 * In:  the non-held-out questions and a range of candidate thresholds.
 * Out: a table of gate behaviour per threshold and a recommendation. Cross-encoder scores
 *      are uncalibrated logits, so this number can only come from measurement.
 */

import { parseArgs } from "node:util";

import { PythonBridge } from "../app/bridge/client.js";
import { balanced, bestRow, fixed, isEntryPoint, saveEvidence, type SweepRow } from "./lib.js";
import { loadQuestions } from "./questions.js";
import { evaluateRetrieval } from "./retrieval_eval.js";

export const CANDIDATES = [-8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0];

export interface SweepResult {
  rows: SweepRow[];
  recommended: SweepRow | undefined;
  n_questions: number;
}

/** Score each candidate threshold on the tuning split. */
export async function sweep(bridge: PythonBridge, thresholds: number[] = CANDIDATES): Promise<SweepResult> {
  const questions = loadQuestions(false);

  // One model instance across the whole sweep: the worker keeps the embedder and reranker
  // loaded between calls. Reloading per threshold would dominate the runtime and make the
  // latency column meaningless.
  const rows: SweepRow[] = [];
  for (const threshold of thresholds) {
    const result = await evaluateRetrieval(bridge, questions, { relevance_threshold: threshold });
    const answerable = result.gate_correct_on_answerable;
    const absent = result.gate_correct_on_absent;
    rows.push({
      threshold,
      gate_on_answerable: answerable,
      gate_on_absent: absent,
      balanced: balanced(answerable, absent),
      "recall@5": result.recall["5"] ?? Number.NaN,
      mrr: result.mrr,
      gold_paper_hit_rate: result.gold_paper_hit_rate,
    });
  }
  return { rows, recommended: bestRow(rows), n_questions: questions.length };
}

async function runMain(bridge: PythonBridge, overwrite: boolean): Promise<number> {
  const result = await sweep(bridge);
  console.log(`tuning split: ${result.n_questions} questions (held-out questions are not touched)\n`);
  const header = `${"thresh".padStart(7)}  ${"answerable".padStart(10)}  ${"absent".padStart(7)}  `
    + `${"balanced".padStart(8)}  ${"recall@5".padStart(8)}  ${"MRR".padStart(6)}  ${"paper".padStart(6)}`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const row of result.rows) {
    console.log(`${fixed(row.threshold, 1, 7)}  ${fixed(row.gate_on_answerable, 3, 10)}  `
      + `${fixed(row.gate_on_absent, 3, 7)}  ${fixed(row.balanced, 3, 8)}  `
      + `${fixed(row["recall@5"], 3, 8)}  ${fixed(row.mrr, 3, 6)}  ${fixed(row.gold_paper_hit_rate, 3, 6)}`);
  }

  const best = result.recommended;
  if (best) {
    console.log(`\nrecommended relevance_threshold: ${best.threshold}  (balanced ${best.balanced}, `
      + `answerable ${best.gate_on_answerable}, absent ${best.gate_on_absent})`);
  }

  const out = saveEvidence(result, "threshold_sweep", overwrite);
  console.log(`written: ${out}`);
  return 0;
}

async function main(): Promise<void> {
  const { values } = parseArgs({ options: { overwrite: { type: "boolean", default: false } } });
  const bridge = new PythonBridge();
  try {
    process.exitCode = await runMain(bridge, Boolean(values.overwrite));
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  } finally {
    await bridge.close();
  }
}

if (isEntryPoint(import.meta.url)) void main();
