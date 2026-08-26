/**
 * Fills gold_chunk_ids by matching hand-written keywords against chunk text.
 *
 * In:  eval/questions.json with gold_paper_ids, and the indexed chunk store (path via the bridge).
 * Out: the same file with gold_chunk_ids filled, plus a printed report to eyeball. Exit 1
 *      while any answerable question is left unlabelled.
 *
 * Deliberately does NOT call retrieve_evidence. Labelling with the same component the
 * eval scores would make recall@k trivially 1.0 and measure nothing; plain keyword
 * matching over the gold papers is an independent mechanism.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { PythonBridge } from "../app/bridge/client.js";
import type { EnginePaths } from "../app/bridge/types.js";
import { REPO_ROOT } from "../app/paths.js";
import { isEntryPoint, readJsonl } from "./lib.js";
import type { Question } from "./questions.js";

interface Chunk {
  chunk_id: string;
  paper_id: string;
  chunk_type: string;
  position: number;
  text: string;
}

// Terms a chunk must contain to count as gold for that question. Written by hand from the
// abstracts, using the papers' own vocabulary — the questions themselves are paraphrased,
// which is what keeps retrieval from succeeding by string matching.
// Conjunctive on purpose: single common terms ("calibration", "retraining") match a
// paper's front matter and reference list as readily as its claims, and a gold set
// containing the title page makes recall@k meaningless — retrieving an affiliation block
// would score as a hit.
export const KEYWORDS: Record<string, string[][]> = {
  sp01: [["sparsemixer", "gradient"], ["backpropagation", "sparse"],
    ["gradient", "estimator"], ["st estimator"]],
  sp02: [["milora", "expert"], ["multi-tenant", "latency"], ["prompt-aware", "rout"],
    ["lora", "router"]],
  sp03: [["qpruner", "quantiz"], ["quantization", "pruning", "memory"],
    ["fine-tuning", "memory", "prun"]],
  sp04: [["adapruner", "prun"], ["calibration", "random"], ["calibration", "sample"],
    ["importance", "estimation", "prun"]],
  sp05: [["routing collapse", "entropy"], ["entropy", "final layers"],
    ["hebrew", "rout"], ["deep-layer", "collapse"]],
  mp01: [["trsp", "regulariz"], ["retraining", "prun"], ["knowledge loss", "prun"],
    ["calibration", "importance"], ["quantization", "memory", "prun"]],
  mp02: [["context incompleteness"], ["unstable", "rout"], ["inconsistent", "rout"],
    ["entropy", "collapse"], ["specialis", "rout"], ["specializ", "rout"]],
  mp03: [["subset of experts", "rout"], ["structured pruning", "remov"],
    ["redundant", "parameters", "remov"], ["conditional", "comput"]],
  mp04: [["edge", "memory"], ["speculative decoding"], ["binariz", "prun"],
    ["binarization", "quantization"]],
  cf01: [["necessitat", "fine-tun"], ["retraining", "avoid"], ["extensive retraining"],
    ["knowledge loss", "retrain"], ["accuracy degradation", "prun"]],
  cf02: [["unstable", "inconsistent"], ["semantically inconsistent"],
    ["linguistically structured"], ["mutual information", "rout"],
    ["specialisation", "categor"]],
  cf03: [["routing collapse"], ["context fusion", "rout"], ["bottleneck", "specializ"],
    ["bottleneck", "specialis"]],
};

// Chunks that match a keyword but cannot answer anything: title pages, affiliations,
// copyright blocks, bare author lists.
const NOISE = /(copyright ©|all rights reserved|@[a-z0-9.-]+\.(edu|com|org|cn)|school of |department of |university\b.{0,40}\b(china|usa|uk)\b)/i;

const squash = (text: string): string => text.split(/\s+/).filter(Boolean).join(" ");

/** True for front matter — matched a term, but carries no claim to retrieve. */
export function isNoise(text: string): boolean {
  return NOISE.test(squash(text).slice(0, 400));
}

export function matches(text: string, groups: string[][]): boolean {
  const lowered = text.toLowerCase();
  return groups.some((group) => group.every((term) => lowered.includes(term)));
}

async function runMain(bridge: PythonBridge): Promise<number> {
  const path = join(REPO_ROOT, "eval", "questions.json");
  const payload = JSON.parse(readFileSync(path, "utf-8")) as { questions: Question[] };
  const { chunks: chunksPath } = await bridge.call<EnginePaths>("paths");
  const chunks = readJsonl<Chunk>(chunksPath);

  for (const question of payload.questions) {
    const qid = question.question_id;
    if (question.expect_abstention) {
      question.gold_chunk_ids = [];
      console.log(`${qid}  abstention — no gold chunks by design`);
      continue;
    }

    const groups = KEYWORDS[qid] ?? [];
    const papers = new Set(question.gold_paper_ids);
    const hits = chunks.filter((c) => papers.has(c.paper_id) && matches(c.text, groups) && !isNoise(c.text));
    // Prefer text chunks; a figure caption rarely carries the claim being scored.
    hits.sort((a, b) => Number(a.chunk_type !== "text") - Number(b.chunk_type !== "text") || a.position - b.position);
    question.gold_chunk_ids = hits.slice(0, 6).map((c) => c.chunk_id);

    console.log(`\n${qid}  ${question.gold_chunk_ids.length} gold chunks from ${JSON.stringify([...papers].sort())}`);
    for (const chunk of hits.slice(0, 3)) {
      console.log(`    ${chunk.chunk_id} [${chunk.chunk_type}] ${squash(chunk.text).slice(0, 110)}`);
    }
  }

  writeFileSync(path, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");
  const unlabelled = payload.questions
    .filter((q) => !q.gold_chunk_ids.length && !q.expect_abstention)
    .map((q) => q.question_id);
  console.log(`\nunlabelled non-abstention questions: ${unlabelled.length ? JSON.stringify(unlabelled) : "none"}`);
  return unlabelled.length ? 1 : 0;
}

async function main(): Promise<void> {
  const bridge = new PythonBridge();
  try {
    process.exitCode = await runMain(bridge);
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  } finally {
    await bridge.close();
  }
}

if (isEntryPoint(import.meta.url)) void main();
