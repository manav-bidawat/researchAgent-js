import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { PythonBridge } from "../app/bridge/client.js";
import {
  abstentionCorrect, answerSentences, anyGoldPaperHit, factCoverage, looksLikeAbstention,
  mean, recallAtK, reciprocalRank, toolTraceMatch,
} from "../eval/metrics.js";
import { stubBridge } from "./helpers.js";

describe("recallAtK", () => {
  it("is the fraction of gold found in the top k", () => {
    const retrieved = ["a", "b", "c", "d"];
    expect(recallAtK(retrieved, ["a", "c"], 1)).toBe(0.5);
    expect(recallAtK(retrieved, ["a", "c"], 3)).toBe(1);
    expect(recallAtK(retrieved, ["z"], 5)).toBe(0);
    expect(recallAtK([], ["a"], 5)).toBe(0);
  });
  it("counts duplicate gold ids once", () => {
    expect(recallAtK(["a"], ["a", "a", "b"], 5)).toBe(0.5);
  });
  it("is NaN with no gold", () => {
    expect(recallAtK(["a"], [], 5)).toBeNaN();
  });
});

describe("reciprocalRank", () => {
  it("is 1/rank of the first gold hit", () => {
    expect(reciprocalRank(["a", "b", "c"], ["a"])).toBe(1);
    expect(reciprocalRank(["a", "b", "c"], ["c", "b"])).toBe(0.5);
    expect(reciprocalRank(["a", "b", "c"], ["c"])).toBeCloseTo(1 / 3);
  });
  it("is 0 on a miss and NaN with no gold", () => {
    expect(reciprocalRank(["a"], ["z"])).toBe(0);
    expect(reciprocalRank(["a"], [])).toBeNaN();
  });
});

describe("anyGoldPaperHit", () => {
  it("is true when any retrieved paper is gold", () => {
    expect(anyGoldPaperHit(["p1", "p2"], ["p2"])).toBe(true);
    expect(anyGoldPaperHit(["p1"], ["p2"])).toBe(false);
    expect(anyGoldPaperHit([], [])).toBe(false);
  });
});

describe("abstention", () => {
  it("recognises abstention phrasing, across case and whitespace", () => {
    expect(looksLikeAbstention("I cannot answer this from the corpus.")).toBe(true);
    expect(looksLikeAbstention("The corpus DOES NOT\n  CONTAIN anything on that.")).toBe(true);
    expect(looksLikeAbstention("There is insufficient evidence.")).toBe(true);
    expect(looksLikeAbstention("Attention scales quadratically [p1__c0].")).toBe(false);
    expect(looksLikeAbstention("")).toBe(false);
  });

  it("scores abstention against expectation", () => {
    const abstain = "No evidence in the corpus covers that.";
    const answer = "It uses dropout [p1__c2].";
    // Expected to abstain: saying so counts, and so does the gate firing.
    expect(abstentionCorrect(abstain, true, true)).toBe(true);
    expect(abstentionCorrect(answer, false, true)).toBe(true);
    expect(abstentionCorrect(answer, true, true)).toBe(false);
    // Expected to answer: only a non-abstaining answer counts.
    expect(abstentionCorrect(answer, true, false)).toBe(true);
    expect(abstentionCorrect(abstain, true, false)).toBe(false);
  });
});

describe("toolTraceMatch", () => {
  it("reports missing and unexpected tools, matching on the expected set", () => {
    const match = toolTraceMatch(
      ["retrieve_evidence", "analyze_corpus", "retrieve_evidence"],
      ["retrieve_evidence", "find_conflicts"],
    );
    expect(match).toEqual({
      expected: ["find_conflicts", "retrieve_evidence"],
      used: ["retrieve_evidence", "analyze_corpus", "retrieve_evidence"],
      missing: ["find_conflicts"],
      unexpected: ["analyze_corpus"],
      matched: false,
    });
    expect(toolTraceMatch(["a", "b"], ["a"]).matched).toBe(true);
    expect(toolTraceMatch([], []).matched).toBe(true);
  });
});

describe("mean", () => {
  it("ignores NaN and rounds to four places", () => {
    expect(mean([1, Number.NaN, 0])).toBe(0.5);
    expect(mean([1, 1, 0])).toBe(0.6667);
  });
  it("is NaN when nothing is defined", () => {
    expect(mean([])).toBeNaN();
    expect(mean([Number.NaN, Number.NaN])).toBeNaN();
  });
});

describe("answerSentences", () => {
  it("splits on sentence ends and newlines and drops fragments under three words", () => {
    expect(answerSentences("First real sentence here. Short one! Is this a question? Ok.\nLine two has words\n\nx y"))
      .toEqual(["First real sentence here.", "Is this a question?", "Line two has words"]);
    expect(answerSentences("")).toEqual([]);
  });
});

describe("factCoverage (via the stub embed_passages op)", () => {
  let bridge: PythonBridge;
  beforeAll(() => { bridge = stubBridge(); });
  afterAll(async () => { await bridge.close(); });

  it("is NaN with no facts", async () => {
    const result = await factCoverage(bridge, "Anything at all here.", ["", "  "]);
    expect(result).toEqual({ covered: 0, total: 0, ratio: Number.NaN, per_fact: [] });
  });

  it("is zero when the answer has no scoreable sentence", async () => {
    const result = await factCoverage(bridge, "Too short.", ["attention is quadratic"]);
    expect(result.ratio).toBe(0);
    expect(result.per_fact).toEqual([{ fact: "attention is quadratic", similarity: 0, covered: false }]);
  });

  it("covers a fact the answer states and not one it omits", async () => {
    const answer = "Self attention cost grows quadratically with length. The weather was sunny today.";
    const result = await factCoverage(bridge, answer, [
      "self attention cost grows quadratically with length",
      "dropout rate was set to ten percent",
    ]);
    expect(result.total).toBe(2);
    expect(result.covered).toBe(1);
    expect(result.ratio).toBe(0.5);
    expect(result.per_fact[0]).toEqual({ fact: "self attention cost grows quadratically with length", similarity: 1, covered: true });
    expect(result.per_fact[1]!.covered).toBe(false);
    expect(result.per_fact[1]!.similarity).toBeLessThan(0.62);
  });

  it("honours a custom threshold", async () => {
    const answer = "Self attention cost grows quadratically with length.";
    const fact = "attention cost grows with sequence length";
    const loose = await factCoverage(bridge, answer, [fact], 0.1);
    const strict = await factCoverage(bridge, answer, [fact], 0.99);
    expect(loose.covered).toBe(1);
    expect(strict.covered).toBe(0);
  });
});
