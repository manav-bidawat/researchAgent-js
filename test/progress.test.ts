import { Writable } from "node:stream";
import { describe, expect, it } from "vitest";

import { ConsoleReporter, formatArgs, formatResult, truncate } from "../app/cli/progress.js";

class Capture extends Writable {
  text = "";
  override _write(chunk: Buffer | string, _enc: BufferEncoding, done: () => void): void {
    this.text += chunk.toString();
    done();
  }
}

describe("truncate / formatArgs", () => {
  it("truncates with an ellipsis only when over width", () => {
    expect(truncate("short", 10)).toBe("short");
    expect(truncate("abcdefghij", 5)).toBe("abcd…");
  });

  it("renders string args quoted and others as JSON", () => {
    expect(formatArgs({ query: "attention", k: 5, types: ["text"] }, 200)).toBe("query='attention', k=5, types=[\"text\"]");
    expect(formatArgs({}, 50)).toBe("");
    expect(formatArgs(undefined, 50)).toBe("");
    expect(formatArgs({ query: "x".repeat(100) }, 20)).toHaveLength(20);
  });
});

describe("formatResult", () => {
  const W = 80;
  it("reports errors first", () => {
    expect(formatResult({ error: "index_missing", chunks: [] }, W)).toBe("error: index_missing");
  });
  it("counts retrieved chunks and flags insufficient evidence", () => {
    expect(formatResult({ chunks: [{}, {}], sufficient_evidence: true }, W)).toBe("2 chunks");
    expect(formatResult({ chunks: [{}], sufficient_evidence: false }, W)).toBe("1 chunk, insufficient");
  });
  it("summarises a collection", () => {
    expect(formatResult({ papers_added: ["a"], chunks_added: 40 }, W)).toBe("1 paper added, 40 chunks");
    expect(formatResult({ papers_added: [] }, W)).toBe("0 papers added, 0 chunks");
  });
  it("summarises a groundedness check", () => {
    expect(formatResult({ claims: [{}, {}, {}], grounded_ratio: 0.67 }, W)).toBe("3 claims, grounded 0.67");
  });
  it("summarises a conflict search", () => {
    expect(formatResult({ found: true, n_pairs_scored: 1 }, W)).toBe("1 pair scored, conflict found");
    expect(formatResult({ found: false, n_pairs_scored: 6 }, W)).toBe("6 pairs scored, no conflict");
  });
  it("prefers an analysis summary over its operation name", () => {
    expect(formatResult({ operation: "stats", summary: "12 papers" }, W)).toBe("12 papers");
    expect(formatResult({ operation: "timeline" }, W)).toBe("timeline");
  });
  it("shows an image by file name", () => {
    expect(formatResult({ image_path: "/data/figures/p1_fig2.png" }, W)).toBe("p1_fig2.png");
  });
  it("falls back to the sorted keys", () => {
    expect(formatResult({ zeta: 1, alpha: 2 }, W)).toBe("alpha, zeta");
  });
  it("handles a missing result", () => {
    expect(formatResult(undefined, W)).toBe("undefined");
  });
});

describe("ConsoleReporter", () => {
  it("opens a line on tool_start and closes it on tool_end", () => {
    const out = new Capture();
    const reporter = new ConsoleReporter(80, out);
    reporter.handle({ kind: "thinking", iteration: 0 });
    reporter.handle({ kind: "tool_start", iteration: 0, tool_name: "retrieve_evidence", args: { query: "q" } });
    expect(out.text.endsWith("... ")).toBe(true);
    reporter.handle({ kind: "tool_end", iteration: 0, tool_name: "retrieve_evidence", result: { chunks: [{}] }, latency_ms: 42 });
    reporter.handle({ kind: "answering", iteration: 1 });
    expect(out.text).toBe(
      "[1] thinking...\n"
      + "[1] → retrieve_evidence(query='q') ... 1 chunk, 42ms\n"
      + "[2] answering\n",
    );
  });

  it("closes a dangling open line before the next event", () => {
    const out = new Capture();
    const reporter = new ConsoleReporter(80, out);
    reporter.handle({ kind: "tool_start", iteration: 2, tool_name: "analyze_corpus", args: {} });
    reporter.handle({ kind: "cap", iteration: 2 });
    expect(out.text).toBe("[3] → analyze_corpus() ... \n[3] iteration cap reached — forcing an answer from what is held\n");
  });

  it("prints a self-contained line for a tool_end with no open line", () => {
    const out = new Capture();
    const reporter = new ConsoleReporter(80, out);
    reporter.handle({ kind: "tool_end", iteration: 0, tool_name: "retrieve_evidence", result: { error: "boom" } });
    expect(out.text).toBe("[1] → retrieve_evidence ... error: boom\n");
  });

  it("formats skipped, image, and ignores unknown kinds", () => {
    const out = new Capture();
    const reporter = new ConsoleReporter(80, out);
    reporter.handle({ kind: "skipped", iteration: 0, tool_name: "retrieve_evidence", reason: "repeated call" });
    reporter.handle({ kind: "skipped", iteration: 0, reason: "bad json" });
    reporter.handle({ kind: "image", iteration: 1, image_path: "/x/y/fig.png" });
    reporter.handle({ kind: "mystery", iteration: 1 });
    expect(out.text).toBe(
      "[1] → retrieve_evidence skipped: repeated call\n"
      + "[1] → call skipped: bad json\n"
      + "[2] ↳ attaching fig.png\n",
    );
  });
});
