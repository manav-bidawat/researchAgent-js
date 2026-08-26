import { describe, expect, it } from "vitest";

import { aggregate, countCitations, type ErrorRow, type ScoredRow } from "../eval/runner.js";

describe("countCitations", () => {
  it("counts unique citations and which resolve to retrieved chunks", () => {
    const answer = "A [p1__c0]. B [p1__c0] and [p2__c3]. C [ghost__c9]. Not a cite [p1] or [see 3].";
    expect(countCitations(answer, ["p1__c0", "p2__c3", "p5__c1"])).toEqual({
      total: 3,
      resolved: 2,
      unresolved: ["ghost__c9"],
    });
  });
  it("handles an answer with no citations", () => {
    expect(countCitations("", [])).toEqual({ total: 0, resolved: 0, unresolved: [] });
  });
});

function row(overrides: Partial<ScoredRow>): ScoredRow {
  return {
    question_id: "q", category: "factual", expect_abstention: false, abstained: false,
    abstention_correct: true, recall: { "1": 1, "3": 1, "5": 1 }, mrr: 1, gold_paper_hit: true,
    fact_coverage: 1, facts_covered: "1/1", citations: { total: 2, resolved: 2, unresolved: [] },
    tools: { expected: [], used: [], missing: [], unexpected: [], matched: true },
    iterations: 2, tool_calls: 1, stopped_because: "answered", elapsed_ms: 100, answer: "", run_id: "r",
    ...overrides,
  };
}

describe("aggregate", () => {
  it("splits answerable from abstention questions and skips error rows", () => {
    const rows = [
      row({ question_id: "a1", elapsed_ms: 300 }),
      row({
        question_id: "a2", recall: { "1": 0, "3": 0.5, "5": 1 }, mrr: 0.5, gold_paper_hit: false,
        fact_coverage: 0, citations: { total: 4, resolved: 2, unresolved: ["x__c1", "y__c2"] },
        abstained: true, abstention_correct: false, tools: { expected: ["t"], used: [], missing: ["t"], unexpected: [], matched: false },
        iterations: 4, elapsed_ms: 100,
      }),
      row({
        question_id: "x1", expect_abstention: true, abstained: true, recall: { "1": Number.NaN, "3": Number.NaN, "5": Number.NaN },
        mrr: Number.NaN, fact_coverage: Number.NaN, citations: { total: 0, resolved: 0, unresolved: [] }, elapsed_ms: 200,
      }),
      { question_id: "e1", category: "factual", error: "llm_unavailable: timeout", elapsed_ms: 5 } satisfies ErrorRow,
    ];
    expect(aggregate(rows)).toEqual({
      errors: 1,
      recall: { "1": 0.5, "3": 0.75, "5": 1 },
      mrr: 0.75,
      gold_paper_hit_rate: 0.5,
      fact_coverage: 0.5,
      abstention_accuracy: 0.6667,
      abstention_on_absent: 1,
      false_abstention_on_answerable: 0.5,
      citations_resolved: 0.75,
      tool_trace_matched: 0.6667,
      mean_iterations: 2.6667,
      median_elapsed_ms: 200,
    });
  });

  it("gives NaN aggregates and zero median for no scored rows", () => {
    const result = aggregate([]);
    expect(result.errors).toBe(0);
    expect(result.mrr).toBeNaN();
    expect(result.median_elapsed_ms).toBe(0);
  });
});
