/**
 * Retrieval-only evaluation: recall, MRR and latency for one retrieval configuration.
 *
 * In:  the question set, plus overrides for the retrieval config (threshold, rerank on/off).
 * Out: per-question and aggregate metrics. Calls retrieve_evidence through the bridge's
 *      retrieve_batch op — no agent, no LLM — so a sweep and the ablation are cheap.
 */

import type { PythonBridge } from "../app/bridge/client.js";
import type { JsonObject, RetrieveBatchRow } from "../app/bridge/types.js";
import { anyGoldPaperHit, mean, recallAtK, reciprocalRank } from "./metrics.js";
import type { Question } from "./questions.js";

export interface RetrievalRow {
  question_id: string;
  category: string;
  expect_abstention: boolean;
  sufficient_evidence: boolean;
  n_retrieved: number;
  recall: Record<string, number>;
  mrr: number;
  gold_paper_hit: boolean;
  latency_ms: number;
  retrieved: string[];
  scores: (number | null)[];
}

export interface RetrievalResult {
  n_questions: number;
  recall: Record<string, number>;
  mrr: number;
  gold_paper_hit_rate: number;
  gate_correct_on_answerable: number;
  gate_correct_on_absent: number;
  median_latency_ms: number;
  rows: RetrievalRow[];
}

export async function evaluateRetrieval(
  bridge: PythonBridge,
  questions: Question[],
  overrides: JsonObject = {},
  k = 5,
  recallKs: number[] = [1, 3, 5],
): Promise<RetrievalResult> {
  const { rows: raw } = await bridge.call<{ rows: RetrieveBatchRow[] }>("retrieve_batch", {
    questions: questions.map((q) => ({ question_id: q.question_id, question: q.question })),
    overrides,
    k,
  });

  const rows: RetrievalRow[] = questions.map((question, index) => {
    const hit = raw[index];
    const retrieved = hit?.chunk_ids ?? [];
    const gold = question.gold_chunk_ids;
    return {
      question_id: question.question_id,
      category: question.category,
      expect_abstention: question.expect_abstention,
      sufficient_evidence: Boolean(hit?.sufficient_evidence),
      n_retrieved: retrieved.length,
      recall: Object.fromEntries(recallKs.map((n) => [String(n), recallAtK(retrieved, gold, n)])),
      mrr: reciprocalRank(retrieved, gold),
      gold_paper_hit: anyGoldPaperHit(hit?.paper_ids ?? [], question.gold_paper_ids),
      latency_ms: hit?.latency_ms ?? 0,
      retrieved,
      scores: hit?.scores ?? [],
    };
  });

  const answerable = rows.filter((row) => !row.expect_abstention);
  const abstaining = rows.filter((row) => row.expect_abstention);
  const latencies = rows.map((row) => row.latency_ms).sort((a, b) => a - b);

  return {
    n_questions: rows.length,
    recall: Object.fromEntries(recallKs.map((n) => [String(n), mean(answerable.map((row) => row.recall[String(n)] ?? Number.NaN))])),
    mrr: mean(answerable.map((row) => row.mrr)),
    gold_paper_hit_rate: mean(answerable.map((row) => Number(row.gold_paper_hit))),
    gate_correct_on_answerable: mean(answerable.map((row) => Number(row.sufficient_evidence))),
    gate_correct_on_absent: mean(abstaining.map((row) => Number(!row.sufficient_evidence))),
    median_latency_ms: latencies.length ? (latencies[Math.floor(latencies.length / 2)] ?? 0) : 0,
    rows,
  };
}
