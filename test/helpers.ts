/**
 * Shared test helpers: a PythonBridge wired to the Node stub worker instead of Python.
 *
 * In:  optional extra env for the stub (e.g. STUB_MANIFEST).
 * Out: a bridge whose ops are the canned ones in test/fixtures/stub-worker.mjs.
 */

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { PythonBridge } from "../app/bridge/client.js";

export const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), "fixtures");
export const STUB_WORKER = join(FIXTURES, "stub-worker.mjs");
export const STUB_PYTHON = join(FIXTURES, "stub-python.sh");

export function stubBridge(env: NodeJS.ProcessEnv = {}): PythonBridge {
  return new PythonBridge({ command: process.execPath, args: [STUB_WORKER], env, forwardStderr: false });
}

/** A small manifest: three papers, two topics, with some fields left out. */
export const MANIFEST = {
  embedding_model: "BAAI/bge-small-en-v1.5",
  updated_at: "2026-09-01T00:00:00Z",
  topics: {
    topic_a: { query: "Topic A as typed", arxiv_query: "all:a" },
    topic_b: null,
  },
  papers: {
    p2: {
      paper_id: "p2", arxiv_id: "2401.00002", title: "beta paper", authors: ["B"], year: 2024,
      categories: ["cs.LG"], topic_tags: ["topic_a", "topic_b"], n_chunks: 7, n_figures: 1,
      pdf_path: "/data/pdfs/p2.pdf", pdf_url: "https://arxiv.org/pdf/2401.00002", parse_status: "ok",
    },
    p1: {
      paper_id: "p1", title: "Alpha paper", topic_tags: ["topic_a"], n_chunks: 3,
    },
    p3: { paper_id: "p3", title: "", topic_tags: ["topic_b"] },
  },
};

export function writeManifest(path: string): void {
  writeFileSync(path, JSON.stringify(MANIFEST));
}
