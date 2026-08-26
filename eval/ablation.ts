/**
 * Rerank ablation: the same eval set with the cross-encoder on and off.
 *
 * In:  the tuning split and the tuned threshold from config (via the bridge).
 * Out: quality and latency for both configurations, reported as measured. A marginal gain
 *      at this corpus size is a finding, not something to hide.
 */

import { parseArgs } from "node:util";

import { PythonBridge } from "../app/bridge/client.js";
import { fixed, isEntryPoint, saveEvidence, signed } from "./lib.js";
import { loadQuestions } from "./questions.js";
import { evaluateRetrieval } from "./retrieval_eval.js";

export interface AblationArm {
  recall: Record<string, number>;
  mrr: number;
  gold_paper_hit_rate: number;
  median_latency_ms: number;
}

export interface AblationResult {
  n_questions: number;
  arms: Record<"rerank_on" | "rerank_off", AblationArm>;
}

/** Evaluate retrieval with rerank enabled and disabled; the worker shares one model instance. */
export async function run(bridge: PythonBridge, heldOut = false): Promise<AblationResult> {
  const questions = loadQuestions(heldOut);
  const { retrieval } = await bridge.call<{ retrieval: Record<string, unknown> }>("retrieval_config");
  const tunedThreshold = Number(retrieval.relevance_threshold);

  const arms = {} as AblationResult["arms"];
  for (const [label, enabled] of [["rerank_on", true], ["rerank_off", false]] as const) {
    // With rerank off the bi-encoder's cosine is what the gate sees, and cosine and
    // cross-encoder logits are not on the same scale, so the tuned threshold does not
    // transfer. The arm is run ungated to keep the comparison about ranking quality.
    const result = await evaluateRetrieval(bridge, questions, {
      rerank_enabled: enabled,
      relevance_threshold: enabled ? tunedThreshold : -1e9,
    });
    arms[label] = {
      recall: result.recall,
      mrr: result.mrr,
      gold_paper_hit_rate: result.gold_paper_hit_rate,
      median_latency_ms: result.median_latency_ms,
    };
  }
  return { n_questions: questions.length, arms };
}

async function runMain(bridge: PythonBridge, overwrite: boolean): Promise<number> {
  const result = await run(bridge);
  const { rerank_on: on, rerank_off: off } = result.arms;
  console.log(`rerank ablation over ${result.n_questions} tuning questions\n`);
  const header = `${"arm".padEnd(12)}  ${"recall@1".padStart(8)}  ${"recall@3".padStart(8)}  `
    + `${"recall@5".padStart(8)}  ${"MRR".padStart(6)}  ${"paper".padStart(6)}  ${"latency".padStart(8)}`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const [label, arm] of [["rerank on", on], ["rerank off", off]] as const) {
    console.log(`${label.padEnd(12)}  ${fixed(arm.recall["1"], 3, 8)}  ${fixed(arm.recall["3"], 3, 8)}  `
      + `${fixed(arm.recall["5"], 3, 8)}  ${fixed(arm.mrr, 3, 6)}  `
      + `${fixed(arm.gold_paper_hit_rate, 3, 6)}  ${String(arm.median_latency_ms).padStart(7)}ms`);
  }

  const deltaMrr = on.mrr - off.mrr;
  const deltaLatency = on.median_latency_ms - off.median_latency_ms;
  console.log(`\nrerank changes MRR by ${signed(deltaMrr, 3)} and median latency by ${signed(deltaLatency)}ms`);

  const out = saveEvidence(result, "ablation", overwrite);
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
