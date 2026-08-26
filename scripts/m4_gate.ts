/**
 * The M4 verification gate from docs/BUILD_PLAN.md: retrieve_evidence, end to end.
 *
 * In:  an index built by M3.
 * Out: pass/fail per check across three queries — clearly answerable, clearly absent, and
 *      borderline — plus rerank latency and proof the cross-encoder saw only the shortlist.
 */

import { PythonBridge } from "../app/bridge/client.js";
import type { EnginePaths, JsonObject } from "../app/bridge/types.js";
import { Gate, errorMessage, loadChunks, squash, type GateConfig } from "./lib.js";

const ANSWERABLE = "how does the gating network route tokens to experts";
const ABSENT = "what is the optimal fermentation temperature for sourdough starter culture";
const BORDERLINE = "how do convolutional layers affect inference latency";

interface EvidenceChunk {
  chunk_id: string;
  paper_id: string;
  paper_title: string;
  page: number;
  section: string | null;
  chunk_type: string;
  score: number;
  text: string;
  [key: string]: unknown;
}

interface Evidence {
  chunks?: EvidenceChunk[];
  sufficient_evidence?: boolean;
  n_candidates_considered?: number;
  rerank_ms?: number;
  note?: string;
  error?: string;
  [key: string]: unknown;
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate();
  const paths = await bridge.call<EnginePaths>("paths");
  const config = await bridge.call<GateConfig>("gate_config");
  const totalChunks = loadChunks(paths.chunks).length;
  if (totalChunks === 0) {
    console.log("No chunks indexed. Run scripts/m3_gate.ts first.");
    return 1;
  }
  console.log(`corpus: ${totalChunks} chunks, threshold=${config.retrieval.relevance_threshold}\n`);

  // Count rerank pairs so the shortlist claim is measured, not asserted. The op wraps the
  // reranker in a counter and keeps one retriever for the whole gate, so cross-call dedup
  // and reset() behave exactly as they do inside one conversation.
  const seenPairs: number[] = [];
  const retrieve = async (query: string, options: JsonObject = {}): Promise<Evidence> => {
    const { result, rerank_pairs } = await bridge.call<{ result: Evidence; rerank_pairs: number[] }>(
      "retrieve_with_rerank_count", { query, ...options },
    );
    seenPairs.push(...rerank_pairs);
    return result;
  };

  const show = (title: string, result: Evidence): void => {
    console.log(`\n[${gate.elapsed()}] ---- ${title} ----`);
    console.log(`  sufficient_evidence=${result.sufficient_evidence} candidates=${result.n_candidates_considered} `
      + `rerank_ms=${result.rerank_ms}`);
    if (result.note) console.log(`  note: ${result.note}`);
    for (const chunk of result.chunks ?? []) {
      console.log(`    ${chunk.score.toFixed(3).padStart(8)}  ${chunk.chunk_id}  p${chunk.page}  [${chunk.section}]`);
      console.log(`            ${squash(chunk.text).slice(0, 110)}...`);
    }
  };

  // 1. Clearly answerable.
  const answerable = await retrieve(ANSWERABLE, { k: 5, reset: true });
  const answerableChunks = answerable.chunks ?? [];
  show("answerable query", answerable);
  gate.check("answerable query returns evidence", answerable.sufficient_evidence === true, `${answerableChunks.length} chunks`);
  gate.check("results are ranked descending",
    answerableChunks.every((c, i) => i === 0 || (answerableChunks[i - 1]?.score ?? 0) >= c.score));
  gate.check("every chunk carries citable metadata", answerableChunks.every((chunk) =>
    (["chunk_id", "paper_id", "paper_title", "page", "chunk_type"] as const).every((field) => chunk[field] !== null && chunk[field] !== undefined)));
  gate.check("chunk text respects the character cap",
    answerableChunks.every((c) => c.text.length <= config.retrieval.max_chunk_chars));
  gate.check("rerank latency was recorded", (answerable.rerank_ms ?? 0) > 0,
    `${answerable.rerank_ms}ms for ${seenPairs.length ? seenPairs[seenPairs.length - 1] : 0} pairs`);

  // 2. The cross-encoder must see the shortlist, never the corpus.
  const maxPairs = seenPairs.length ? Math.max(...seenPairs) : 0;
  gate.check("cross-encoder ran on the shortlist, not the index",
    seenPairs.length > 0 && maxPairs <= config.retrieval.k_retrieve && config.retrieval.k_retrieve < totalChunks,
    `max pairs scored=${maxPairs}, k_retrieve=${config.retrieval.k_retrieve}, corpus=${totalChunks}`);

  // 3. Clearly absent: the gate must fire rather than hand back weak chunks.
  const absent = await retrieve(ABSENT, { k: 5, reset: true });
  show("absent query", absent);
  gate.check("absent query returns sufficient_evidence: false", absent.sufficient_evidence === false);
  gate.check("absent query returns no chunks", !absent.chunks?.length);
  gate.check("absent query explains why", Boolean(absent.note));

  // 4. Borderline: whatever it decides, it must decide coherently.
  const borderline = await retrieve(BORDERLINE, { k: 5, reset: true });
  show("borderline query", borderline);
  gate.check("borderline query is internally consistent",
    Boolean(borderline.chunks?.length) === borderline.sufficient_evidence,
    `sufficient=${borderline.sufficient_evidence} chunks=${borderline.chunks?.length ?? 0}`);

  // 5. Cross-call dedup within one conversation.
  const ids = (result: Evidence): Set<string> => new Set((result.chunks ?? []).map((c) => c.chunk_id));
  const firstIds = ids(await retrieve(ANSWERABLE, { k: 3, reset: true }));
  const secondIds = ids(await retrieve(ANSWERABLE, { k: 3 }));
  const overlap = [...firstIds].filter((id) => secondIds.has(id)).length;
  gate.check("a repeated query does not re-return the same chunks", overlap === 0,
    `first=${firstIds.size} second=${secondIds.size} overlap=${overlap}`);
  const thirdIds = ids(await retrieve(ANSWERABLE, { k: 3, reset: true }));
  gate.check("reset() clears the conversation's dedup state",
    thirdIds.size === firstIds.size && [...thirdIds].every((id) => firstIds.has(id)));

  // 6. Filters and bad input.
  const figuresOnly = await retrieve(ANSWERABLE, { k: 5, chunk_types: ["figure", "table"], reset: true });
  gate.check("chunk_types filter is honoured",
    (figuresOnly.chunks ?? []).every((c) => c.chunk_type === "figure" || c.chunk_type === "table"),
    `${figuresOnly.chunks?.length ?? 0} figure/table chunks`);
  const unknownTopic = await retrieve(ANSWERABLE, { topic_filter: "no_such_topic", reset: true });
  gate.check("an unmatched topic_filter fails the gate rather than ignoring the filter",
    unknownTopic.sufficient_evidence === false && !unknownTopic.chunks?.length);
  gate.check("empty query returns an error dict", (await retrieve("")).error === "empty_query");
  gate.check("bad chunk_types returns an error dict",
    (await retrieve("x", { chunk_types: ["nonsense"] })).error === "bad_chunk_types");

  return gate.finish("M4");
}

async function main(): Promise<void> {
  const bridge = new PythonBridge();
  try {
    process.exitCode = await run(bridge);
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  } finally {
    await bridge.close();
  }
}

void main();
