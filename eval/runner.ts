/**
 * Runs every eval question through the full agent and scores the result.
 *
 * In:  a question split (tuning or held-out) and a bridge to the live engine.
 * Out: per-question rows and aggregates — retrieval recall from the traces, fact coverage,
 *      abstention correctness and the tool trace. Gold labels are used only here, in the
 *      scorer; the agent never sees them.
 */

import type { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type AskFailure, type AskResult } from "../app/bridge/types.js";
import {
  abstentionCorrect, anyGoldPaperHit, factCoverage, looksLikeAbstention, mean,
  recallAtK, reciprocalRank, toolTraceMatch, type ToolTraceMatch,
} from "./metrics.js";
import { loadQuestions, saveResult } from "./questions.js";

export interface CitationCount {
  total: number;
  resolved: number;
  unresolved: string[];
}

export interface ScoredRow {
  question_id: string;
  category: string;
  expect_abstention: boolean;
  abstained: boolean;
  abstention_correct: boolean;
  recall: Record<string, number>;
  mrr: number;
  gold_paper_hit: boolean;
  fact_coverage: number;
  facts_covered: string;
  citations: CitationCount;
  tools: ToolTraceMatch;
  iterations: number;
  tool_calls: number;
  stopped_because: string;
  elapsed_ms: number;
  answer: string;
  run_id: string;
}

export interface ErrorRow {
  question_id: string;
  category: string;
  error: string;
  elapsed_ms: number;
}

export type Row = ScoredRow | ErrorRow;

export interface SplitResult extends Record<string, unknown> {
  n_questions: number;
  rows: Row[];
  errors: number;
  recall: Record<string, number>;
  mrr: number;
  gold_paper_hit_rate: number;
  fact_coverage: number;
  abstention_accuracy: number;
  citations_resolved: number;
}

export interface RunOptions {
  heldOut?: boolean;
  limit?: number;
  /** Accepted for CLI parity; the runner has never run an NLI pass itself. */
  checkGroundedness?: boolean;
}

const isScored = (row: Row): row is ScoredRow => !("error" in row);

/**
 * Inline citations in the answer and how many resolve to a retrieved chunk. An
 * unresolvable citation reads as a source but points nowhere: a fabricated reference.
 */
export function countCitations(answer: string, retrieved: string[]): CitationCount {
  const cited = [...(answer ?? "").matchAll(/\[([A-Za-z0-9_]+__c\d+)\]/g)].map((match) => match[1] ?? "");
  const unique = [...new Set(cited)];
  const held = new Set(retrieved);
  return {
    total: unique.length,
    resolved: unique.filter((id) => held.has(id)).length,
    unresolved: unique.filter((id) => !held.has(id)),
  };
}

/** Aggregate metrics, split by the parts of the question set that differ. */
export function aggregate(rows: Row[]): Omit<SplitResult, "n_questions" | "rows"> {
  const scored = rows.filter(isScored);
  const answerable = scored.filter((row) => !row.expect_abstention);
  const abstaining = scored.filter((row) => row.expect_abstention);
  const elapsed = scored.map((row) => row.elapsed_ms).sort((a, b) => a - b);

  return {
    errors: rows.length - scored.length,
    recall: Object.fromEntries(["1", "3", "5"].map((n) => [n, mean(answerable.map((row) => row.recall[n] ?? Number.NaN))])),
    mrr: mean(answerable.map((row) => row.mrr)),
    gold_paper_hit_rate: mean(answerable.map((row) => Number(row.gold_paper_hit))),
    fact_coverage: mean(answerable.map((row) => row.fact_coverage)),
    abstention_accuracy: mean(scored.map((row) => Number(row.abstention_correct))),
    abstention_on_absent: mean(abstaining.map((row) => Number(row.abstained))),
    false_abstention_on_answerable: mean(answerable.map((row) => Number(row.abstained))),
    citations_resolved: mean(answerable.filter((row) => row.citations.total)
      .map((row) => row.citations.resolved / row.citations.total)),
    tool_trace_matched: mean(scored.map((row) => Number(row.tools.matched))),
    mean_iterations: mean(scored.map((row) => row.iterations)),
    median_elapsed_ms: elapsed.length ? (elapsed[Math.floor(elapsed.length / 2)] ?? 0) : 0,
  };
}

/** Run one split end to end. Returns rows plus aggregates. */
export async function runSplit(bridge: PythonBridge, options: RunOptions = {}): Promise<SplitResult> {
  let questions = loadQuestions(Boolean(options.heldOut));
  if (options.limit) questions = questions.slice(0, options.limit);

  const rows: Row[] = [];
  for (const question of questions) {
    const started = performance.now();
    const result = await bridge.call<AskResult | AskFailure>("ask", {
      question: question.question, question_id: question.question_id, events: false,
    });
    const elapsedMs = Math.round(performance.now() - started);

    if (isToolError(result)) {
      rows.push({
        question_id: question.question_id, category: question.category,
        error: `${result.error}: ${result.detail ?? ""}`, elapsed_ms: elapsedMs,
      });
      console.log(`  ${question.question_id}  ERROR ${result.error}`);
      continue;
    }
    const answered = result as AskResult;

    // Retrieval is scored from the trace, which is why chunk_ids_returned is a top-level
    // trace field: no re-running and no parsing of result_summary.
    const retrieved: string[] = [];
    for (const record of answered.trace_records) {
      for (const chunkId of record.chunk_ids_returned ?? []) {
        if (!retrieved.includes(chunkId)) retrieved.push(chunkId);
      }
    }
    const papers = retrieved.map((id) => id.slice(0, id.lastIndexOf("__c")));
    const gold = question.gold_chunk_ids;
    const answer = answered.answer;

    const coverage = await factCoverage(bridge, answer, question.expected_facts);
    const abstained = looksLikeAbstention(answer);
    const row: ScoredRow = {
      question_id: question.question_id,
      category: question.category,
      expect_abstention: question.expect_abstention,
      abstained,
      abstention_correct: abstentionCorrect(answer, retrieved.length > 0, question.expect_abstention),
      recall: Object.fromEntries([1, 3, 5].map((n) => [String(n), recallAtK(retrieved, gold, n)])),
      mrr: reciprocalRank(retrieved, gold),
      gold_paper_hit: anyGoldPaperHit(papers, question.gold_paper_ids),
      fact_coverage: coverage.ratio,
      facts_covered: `${coverage.covered}/${coverage.total}`,
      citations: countCitations(answer, retrieved),
      tools: toolTraceMatch(answered.trace_records.map((record) => record.tool_name), question.expected_tools),
      iterations: answered.iterations,
      tool_calls: answered.tool_calls,
      stopped_because: answered.stopped_because,
      elapsed_ms: elapsedMs,
      answer,
      run_id: answered.run_id,
    };
    rows.push(row);
    console.log(`  ${question.question_id.padStart(5)}  ${question.category.padEnd(15)} `
      + `facts=${row.facts_covered.padStart(5)}  mrr=${row.mrr.toFixed(2)}  `
      + `abstain=${abstained ? "Y" : "n"}${row.abstention_correct ? "" : " WRONG"}  `
      + `tools=${row.tools.used.length}  ${(elapsedMs / 1000).toFixed(1)}s`);
  }

  return { n_questions: rows.length, rows, ...aggregate(rows) } as SplitResult;
}

export function save(result: SplitResult, name: string): string {
  return saveResult(result, name);
}

export function printMetrics(name: string, metrics: SplitResult): void {
  console.log(`\n${name} — ${metrics.n_questions} question(s)`);
  const labels: [string, keyof SplitResult][] = [
    ["fact coverage", "fact_coverage"],
    ["abstention accuracy", "abstention_accuracy"],
    ["citations resolved", "citations_resolved"],
    ["gold-paper hit rate", "gold_paper_hit_rate"],
    ["MRR", "mrr"],
  ];
  for (const [label, key] of labels) {
    const value = metrics[key];
    if (typeof value === "number") console.log(`  ${label.padEnd(22)} ${value.toFixed(3)}`);
  }
  const recall = Object.entries(metrics.recall ?? {});
  if (recall.length) {
    console.log(`  recall@k               ${recall.map(([k, v]) => `@${k}=${v.toFixed(3)}`).join("  ")}`);
  }
  if (metrics.errors) console.log(`  errors                 ${metrics.errors}`);
}
