/**
 * The M8 verification gate: a full eval run producing a results table.
 *
 * In:  the tuned config and the annotated question set (optional --overwrite).
 * Out: a results table for the tuning split, then the held-out split run once and reported
 *      separately, plus the rerank ablation. Exits non-zero if the harness cannot produce
 *      a table — not if the numbers are disappointing, which is a finding, not a failure.
 */

import { parseArgs } from "node:util";

import { PythonBridge } from "../app/bridge/client.js";
import { run as runAblation } from "../eval/ablation.js";
import { saveEvidence } from "../eval/lib.js";
import { runSplit, type ScoredRow, type SplitResult } from "../eval/runner.js";
import { Gate, errorMessage, fixed, signed } from "./lib.js";

function table(title: string, result: SplitResult): void {
  const rule = "=".repeat(78);
  console.log(`\n${rule}\n${title}  (${result.n_questions} questions)\n${rule}`);
  const header = `${"id".padStart(5)}  ${"category".padEnd(15)}  ${"facts".padStart(6)}  ${"recall@5".padStart(8)}  `
    + `${"MRR".padStart(5)}  ${"abst".padStart(5)}  ${"cites".padStart(7)}  ${"iters".padStart(5)}`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const row of result.rows) {
    if ("error" in row) {
      console.log(`${row.question_id.padStart(5)}  ${row.category.padEnd(15)}  ERROR ${row.error.slice(0, 40)}`);
      continue;
    }
    const scored = row as ScoredRow;
    const mark = scored.abstention_correct ? "ok" : "WRONG";
    console.log(`${scored.question_id.padStart(5)}  ${scored.category.padEnd(15)}  ${scored.facts_covered.padStart(6)}  `
      + `${fixed(scored.recall["5"], 3, 8)}  ${fixed(scored.mrr, 2, 5)}  ${mark.padStart(5)}  `
      + `${scored.citations.resolved}/${String(scored.citations.total).padEnd(5)}  ${String(scored.iterations).padStart(5)}`);
  }

  const n = (key: string): number => Number(result[key]);
  console.log(`\n  recall@1/3/5            ${fixed(result.recall["1"], 3)} / ${fixed(result.recall["3"], 3)} / ${fixed(result.recall["5"], 3)}`);
  console.log(`  MRR                     ${fixed(result.mrr, 3)}`);
  console.log(`  gold paper hit rate     ${fixed(result.gold_paper_hit_rate, 3)}`);
  console.log(`  fact coverage           ${fixed(result.fact_coverage, 3)}`);
  console.log(`  abstention accuracy     ${fixed(result.abstention_accuracy, 3)} `
    + `(abstained on absent ${fixed(n("abstention_on_absent"), 3)}, `
    + `false abstention ${fixed(n("false_abstention_on_answerable"), 3)})`);
  console.log(`  citations resolved      ${fixed(result.citations_resolved, 3)}`);
  console.log(`  tool trace matched      ${fixed(n("tool_trace_matched"), 3)}`);
  console.log(`  mean iterations         ${fixed(n("mean_iterations"), 1)}`);
  console.log(`  median wall clock       ${fixed(n("median_elapsed_ms") / 1000, 1)}s`);
}

async function run(bridge: PythonBridge, overwrite: boolean): Promise<number> {
  const gate = new Gate(false);

  console.log("=== tuning split (threshold was fitted on these) ===\n");
  const tuning = await runSplit(bridge, { heldOut: false });
  table("TUNING SPLIT", tuning);
  console.log(`\nwritten: ${saveEvidence(tuning, "tuning_split", overwrite)}`);

  gate.check("the harness produced a results table", tuning.n_questions > 0, `${tuning.n_questions} questions`);
  gate.check("no question crashed the harness", tuning.errors === 0, `${tuning.errors} errors`);
  gate.check("retrieval metrics are computed", !Number.isNaN(tuning.mrr), `MRR=${tuning.mrr}`);
  gate.check("abstention is scored", !Number.isNaN(tuning.abstention_accuracy), fixed(tuning.abstention_accuracy, 3));
  gate.check("answers carry resolvable citations", tuning.citations_resolved > 0.5,
    `${fixed(tuning.citations_resolved, 3)} of cited ids resolve to retrieved chunks`);

  console.log("\n\n=== rerank ablation ===\n");
  const ablation = await runAblation(bridge);
  const { rerank_on: on, rerank_off: off } = ablation.arms;
  const header = `${"arm".padEnd(12)}  ${"recall@5".padStart(8)}  ${"MRR".padStart(6)}  ${"paper".padStart(6)}  ${"latency".padStart(9)}`;
  console.log(header);
  console.log("-".repeat(header.length));
  for (const [label, arm] of [["rerank on", on], ["rerank off", off]] as const) {
    console.log(`${label.padEnd(12)}  ${fixed(arm.recall["5"], 3, 8)}  ${fixed(arm.mrr, 3, 6)}  `
      + `${fixed(arm.gold_paper_hit_rate, 3, 6)}  ${String(arm.median_latency_ms).padStart(8)}ms`);
  }
  console.log(`\n  rerank delta: MRR ${signed(on.mrr - off.mrr, 3)}, `
    + `latency ${signed(on.median_latency_ms - off.median_latency_ms)}ms`);
  const armNames = Object.keys(ablation.arms).sort();
  gate.check("the ablation ran both arms", armNames.length === 2 && armNames[0] === "rerank_off" && armNames[1] === "rerank_on");

  console.log("\n\n=== held-out split (run once, after tuning stopped) ===\n");
  const held = await runSplit(bridge, { heldOut: true });
  table("HELD-OUT SPLIT", held);
  console.log(`\nwritten: ${saveEvidence(held, "held_out_split", overwrite)}`);
  gate.check("the held-out split ran", held.n_questions > 0, `${held.n_questions} questions`);

  return gate.finish("M8");
}

async function main(): Promise<void> {
  const { values } = parseArgs({ options: { overwrite: { type: "boolean", default: false } } });
  const bridge = new PythonBridge();
  try {
    process.exitCode = await run(bridge, Boolean(values.overwrite));
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  } finally {
    await bridge.close();
  }
}

void main();
