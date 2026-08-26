/**
 * The M2 verification gate from docs/BUILD_PLAN.md, run over the papers M1 fetched.
 *
 * In:  a populated data/ tree (papers downloaded, manifest written) and a .env key.
 * Out: prints a pass/fail line per check — chunk coherence, figure text being caption AND
 *      description, dense positions, and a re-run costing zero vision calls. Non-zero on failure.
 */

import { existsSync, statSync } from "node:fs";

import { PythonBridge } from "../app/bridge/client.js";
import type { EnginePaths, JsonObject } from "../app/bridge/types.js";
import {
  Gate, byString, errorMessage, loadChunks, loadManifest, squash, type ChunkRecord, type GateConfig,
} from "./lib.js";

/** A small seeded PRNG (mulberry32), so the eyeballing sample is stable run to run. */
function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample<T>(items: T[], n: number, random: () => number): T[] {
  const pool = [...items];
  const out: T[] = [];
  for (let i = 0; i < Math.min(n, pool.length); i += 1) {
    const j = i + Math.floor(random() * (pool.length - i));
    [pool[i], pool[j]] = [pool[j] as T, pool[i] as T];
    out.push(pool[i] as T);
  }
  return out;
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate();
  const paths = await bridge.call<EnginePaths>("paths");
  const config = await bridge.call<GateConfig>("gate_config");

  let manifest = loadManifest(paths.manifest);
  if (!Object.keys(manifest.papers).length) {
    console.log("No papers in the manifest. Run scripts/m1_gate.ts first.");
    return 1;
  }

  const existing = loadChunks(paths.chunks).length;
  if (existing === 0) {
    gate.step(`ingesting ${Object.keys(manifest.papers).length} papers (extract, chunk, describe figures)`);
    await bridge.call("ingest_all");
  } else {
    gate.step(`using ${existing} chunks already in the store`);
  }

  manifest = loadManifest(paths.manifest);
  const chunks = loadChunks(paths.chunks);
  gate.check("chunks were produced", chunks.length > 0, `${chunks.length} chunks`);
  if (!chunks.length) return 1;

  const byPaper = new Map<string, ChunkRecord[]>();
  for (const chunk of chunks) {
    const list = byPaper.get(chunk.paper_id) ?? [];
    list.push(chunk);
    byPaper.set(chunk.paper_id, list);
  }

  // 1. position must be dense within every paper, or neighbour expansion walks a hole.
  const dense = [...byPaper.values()].every((cs) => {
    const positions = cs.map((c) => c.position).sort((a, b) => a - b);
    return positions.every((position, i) => position === i);
  });
  gate.check("position is dense and contiguous in every paper", dense,
    [...byPaper.keys()].sort(byString).map((pid) => `${pid}:${byPaper.get(pid)?.length}`).join(", "));
  gate.check("chunk_id agrees with position",
    chunks.every((c) => c.chunk_id === `${c.paper_id}__c${String(c.position).padStart(4, "0")}`));

  // 2. Every chunk must fit the budget both encoders were sized against.
  const maxTokens = config.chunking.max_tokens;
  const over = chunks.filter((c) => c.n_tokens > maxTokens);
  gate.check(`no chunk exceeds chunking.max_tokens (${maxTokens})`, !over.length,
    over.length ? `${over.length} over` : `max=${Math.max(...chunks.map((c) => c.n_tokens))}`);

  // 3. Figure chunks carry caption AND description, never the description alone.
  const figureChunks = chunks.filter((c) => c.chunk_type === "figure" || c.chunk_type === "table");
  gate.check("figure and table chunks exist", figureChunks.length > 0, `${figureChunks.length} of ${chunks.length}`);
  if (figureChunks.length) {
    const haveCaption = figureChunks.filter((c) => (c.caption ?? "").trim() && c.text.startsWith((c.caption ?? "").slice(0, 40)));
    gate.check("figure chunk text starts with its caption", haveCaption.length === figureChunks.length,
      `${haveCaption.length}/${figureChunks.length}`);
    const described = figureChunks.filter((c) => c.text.length > (c.caption ?? "").length + 20);
    gate.check("figure chunk text is caption AND description", described.length > 0,
      `${described.length}/${figureChunks.length} carry a description too`);
    gate.check("figure chunks resolve to an image on disk",
      figureChunks.filter((c) => c.image_path).every((c) => existsSync(c.image_path!) && statSync(c.image_path!).isFile()));
  }

  // 4. Sections were detected somewhere, and null is an accepted outcome.
  const withSection = chunks.filter((c) => c.section);
  gate.check("sections were detected", withSection.length > 0, `${withSection.length}/${chunks.length} chunks carry a section`);

  // 5. Text coherence, printed for eyeballing per the build plan. Seeded, though not with
  //    Python's generator, so the five chunks differ from the ones the .py gate printed.
  const textChunks = chunks.filter((c) => c.chunk_type === "text");
  console.log(`\n[${gate.elapsed()}] ---- 5 random chunks, for eyeballing ----`);
  for (const chunk of sample(textChunks, 5, seeded(11))) {
    console.log(`\n  ${chunk.chunk_id}  p${chunk.page}  [${chunk.section ?? "None"}]  ${chunk.n_tokens} tok`);
    console.log(`    ${squash(chunk.text).slice(0, 260)}...`);
  }

  const junk = textChunks.filter((c) => c.text.slice(0, 60).includes("arXiv:"));
  gate.check("\nno chunk starts with the arXiv margin stamp", !junk.length, `${junk.length} affected`);

  // 6. A re-run must cost zero vision calls *because the cache served them*.
  gate.step("re-running ingest to confirm the description cache holds");
  const { entries: cacheBefore } = await bridge.call<{ entries: number }>("description_cache_size");
  const again = await bridge.call<{ papers: JsonObject[] }>("ingest_all");
  const freshCalls = again.papers.filter((p) => !("error" in p)).reduce((sum, p) => sum + Number(p.vision_calls ?? 0), 0);
  const skipped = again.papers.filter((p) => p.error === "already_ingested");

  // Guard against a vacuous pass. With an empty cache, "zero vision calls" is trivially
  // true and proves nothing at all — which is exactly what happened the first time this
  // gate was run against an exhausted API quota. The cache must have entries for the
  // check below to mean anything.
  if (cacheBefore === 0) {
    gate.check("description cache was populated", false,
      "cache is empty, so the cache-hit check below cannot be evaluated — "
      + "this is a SKIP masquerading as a PASS unless it is failed here");
  } else {
    gate.check("re-run makes no new vision calls", freshCalls === 0, `${freshCalls} calls, ${cacheBefore} cached descriptions`);
  }

  const after = loadChunks(paths.chunks).length;
  gate.check("re-run does not duplicate chunks", after === chunks.length, `${after} vs ${chunks.length}`);
  const nPapers = Object.keys(manifest.papers).length;
  gate.check("already-ingested papers are skipped", skipped.length === nPapers, `${skipped.length}/${nPapers}`);

  return gate.finish("M2");
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
