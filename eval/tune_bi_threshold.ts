/**
 * Sweeps the bi-encoder relevance gate, for the configuration where rerank is disabled.
 *
 * In:  the non-held-out questions and candidate cosine thresholds.
 * Out: gate behaviour per threshold plus a recommendation, scored exactly like the
 *      cross-encoder sweep so the two are comparable. Cosine is bounded 0..1, but where
 *      the useful cut sits is still an empirical question.
 */

import { parseArgs } from "node:util";

import { PythonBridge } from "../app/bridge/client.js";
import { balanced, bestRow, fixed, isEntryPoint, saveEvidence, type SweepRow } from "./lib.js";
import { loadQuestions } from "./questions.js";
import { evaluateRetrieval } from "./retrieval_eval.js";
import type { SweepResult } from "./tune_threshold.js";

export const CANDIDATES = [0.0, 0.40, 0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80];

/** Score each candidate on the tuning split, with reranking off. */
export async function sweep(bridge: PythonBridge, thresholds: number[] = CANDIDATES): Promise<SweepResult> {
  const questions = loadQuestions(false);

  const rows: SweepRow[] = [];
  for (const threshold of thresholds) {
    const result = await evaluateRetrieval(bridge, questions, {
      rerank_enabled: false,
      bi_encoder_relevance_threshold: threshold,
    });
    const answerable = result.gate_correct_on_answerable;
    const absent = result.gate_correct_on_absent;
    const score = balanced(answerable, absent);
    rows.push({
      threshold,
      gate_on_answerable: answerable,
      gate_on_absent: absent,
      balanced: score,
      "recall@5": result.recall["5"] ?? Number.NaN,
      mrr: result.mrr,
      gold_paper_hit_rate: result.gold_paper_hit_rate,
    });
    console.log(`  ${fixed(threshold, 2, 5)}  answerable=${fixed(answerable, 3)}  absent=${fixed(absent, 3)}  `
      + `balanced=${fixed(score, 4)}  mrr=${fixed(result.mrr, 4)}`);
  }
  return { rows, recommended: bestRow(rows), n_questions: questions.length };
}

async function runMain(bridge: PythonBridge, overwrite: boolean): Promise<number> {
  console.log("Sweeping the bi-encoder relevance gate (rerank disabled), tuning split only:");
  const result = await sweep(bridge);
  const out = saveEvidence(result, "bi_threshold_sweep", overwrite);
  console.log(`\nrecommended: ${JSON.stringify(result.recommended)}`);
  console.log(`written to ${out}`);
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
