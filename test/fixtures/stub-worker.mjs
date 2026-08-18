#!/usr/bin/env node
/**
 * A stand-in for bridge/worker.py that speaks the same JSON-lines protocol with no Python.
 *
 * In:  {"id", "op", "params"} lines on stdin.
 * Out: a ready frame, then {"id","event"} frames and a {"id","result"} or
 *      {"id","error","detail"} frame per request. Canned, deterministic answers only.
 */

import { createInterface } from "node:readline";

const send = (frame) => process.stdout.write(`${JSON.stringify(frame)}\n`);

/** Bag-of-words hashed into a fixed number of buckets, L2-normalised. */
function embed(text, dims = 64) {
  const vector = new Array(dims).fill(0);
  for (const word of String(text).toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    let hash = 0;
    for (const ch of word) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    vector[hash % dims] += 1;
  }
  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  if (norm === 0) {
    vector[0] = 1;
    return vector;
  }
  return vector.map((v) => v / norm);
}

const ANSWER = "Transformers use self-attention [p1__c0]. They scale well [p2__c3].";

const OPS = {
  ping: () => ({ ok: true, python: "stub" }),
  paths: () => ({
    manifest: process.env.STUB_MANIFEST ?? "/nonexistent/manifest.json",
    data_dir: "/tmp/stub-data",
    chunks: "/tmp/stub-data/chunks.jsonl",
    traces: "/tmp/stub-data/traces",
    arxiv_cooldown: "/tmp/stub-data/cooldown",
    trace_text_chars: 200,
    progress_text_chars: 60,
  }),
  ask: (params, emit) => {
    const question = String(params.question ?? "");
    if (question === "fail") {
      return {
        error: "llm_unavailable",
        detail: "the provider timed out",
        partial: { answer: "partial answer so far", iterations: 1 },
      };
    }
    if (question === "bridge-error") throw Object.assign(new Error("boom"), { bridge: true });
    if (params.events !== false) {
      emit({ kind: "thinking", iteration: 0 });
      emit({ kind: "tool_start", iteration: 0, tool_name: "retrieve_evidence", args: { query: question } });
      emit({
        kind: "tool_end", iteration: 0, tool_name: "retrieve_evidence",
        result: { chunks: [{ chunk_id: "p1__c0" }, { chunk_id: "p2__c3" }], sufficient_evidence: true },
        latency_ms: 12,
      });
      emit({ kind: "answering", iteration: 1 });
    }
    return {
      answer: ANSWER,
      question,
      iterations: 2,
      tool_calls: 1,
      stopped_because: "answered",
      context_tokens: 1234,
      run_id: "run-stub-1",
      trace: "/tmp/stub-data/traces/run-stub-1.jsonl",
      trace_records: [
        { iteration: 0, tool_name: "retrieve_evidence", args: { query: question },
          chunk_ids_returned: ["p1__c0", "p2__c3"], error: null, latency_ms: 12 },
      ],
    };
  },
  retrieve_evidence: (params) => ({
    chunks: [{ chunk_id: "p1__c0", text: "self-attention", score: 4.2 }],
    sufficient_evidence: true,
    received: params,
  }),
  analyze_corpus: (params) => {
    if (params.operation === "bogus") {
      return { error: "unknown_operation", detail: `no operation '${params.operation}'` };
    }
    return { operation: params.operation, summary: "2 papers, 10 chunks", n_papers: 2, received: params };
  },
  explore_graph: (params) => ({ entity_id: params.entity_id, depth: params.depth, nodes: [], edges: [] }),
  embed_passages: (params) => ({ vectors: (params.texts ?? []).map((text) => embed(text)) }),
  events_then_result: (params, emit) => {
    for (let i = 0; i < Number(params.n ?? 3); i += 1) emit({ kind: "tick", i });
    return { done: true };
  },
  exit: (params) => {
    process.stderr.write("stub worker exiting on request\n");
    process.exit(Number(params.code ?? 3));
  },
};

send({ ready: true, ops: Object.keys(OPS).sort() });

createInterface({ input: process.stdin }).on("line", (line) => {
  if (!line.trim()) return;
  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    send({ id: null, error: "bad_request", detail: `invalid JSON: ${error.message}` });
    return;
  }
  const { id, op, params } = request;
  const fn = OPS[op];
  if (!fn) {
    send({ id, error: "unknown_op", detail: `no bridge op '${op}'` });
    return;
  }
  const emit = (event) => send({ id, event });
  try {
    send({ id, result: fn(params ?? {}, emit) });
  } catch (error) {
    send({ id, error: "op_crashed", detail: `${error.name}: ${error.message}` });
  }
});
