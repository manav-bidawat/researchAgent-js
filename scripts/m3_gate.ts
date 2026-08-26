/**
 * The M3 verification gate from docs/BUILD_PLAN.md, including a real process restart.
 *
 * In:  a data/ tree with chunks from M2 and a .env key (only for fetching a 4th paper).
 * Out: pass/fail per check — lockstep persistence, faiss_id_map surviving a restart in a
 *      genuinely separate process, incremental append, and model-mismatch detection.
 */

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type EnginePaths, type JsonObject } from "../app/bridge/types.js";
import { Gate, errorMessage, loadChunks, loadManifest, type GateConfig } from "./lib.js";

const QUERY = "how are experts selected for each token by the routing network";
const NEW_TOPIC = "vision transformer patch embedding design";

interface Hit {
  chunk_id: string;
  paper_id: string;
  score: number;
  chunk_type: string;
  text: string;
}

interface Consistency {
  index_ntotal: number;
  faiss_id_map: number;
  embeddings_rows: number;
  chunks: number;
  embedding_model: string;
  embedding_dim: number;
}

/**
 * Load the index and search in a brand-new worker process.
 *
 * Doing this in the gate's own worker would prove nothing: the mapping may already be in
 * memory. The whole point is that faiss_id_map survived to disk, so it has to be read back
 * cold, by an interpreter that has never seen it.
 */
async function restartAndSearch(): Promise<Hit[]> {
  const cold = new PythonBridge();
  try {
    const { hits } = await cold.call<{ hits: Hit[] }>("bi_encoder_search", { query: QUERY, k: 5 });
    return hits;
  } finally {
    await cold.close();
  }
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate();
  const paths = await bridge.call<EnginePaths>("paths");
  const config = await bridge.call<GateConfig>("gate_config");
  if (!loadChunks(paths.chunks).length) {
    console.log("No chunks. Run scripts/m2_gate.ts first.");
    return 1;
  }

  gate.step("indexing any chunks not yet embedded");
  const built = await bridge.call<JsonObject>("index_chunks");
  if (isToolError(built)) {
    gate.check("index builds", false, `${built.error}: ${built.detail}`);
    return 1;
  }
  gate.check("index builds", true, `${built.total_indexed} vectors`);

  // 1. The three artefacts must agree exactly. FAISS ids are positional, so a mismatch
  //    means lookups silently resolve to the wrong chunk.
  const manifest = loadManifest(paths.manifest);
  const sizes = await bridge.call<Consistency>("vector_index_consistency");
  gate.check("faiss.index, embeddings.npy and faiss_id_map agree",
    sizes.index_ntotal === sizes.faiss_id_map && sizes.faiss_id_map === sizes.embeddings_rows
      && sizes.embeddings_rows === sizes.chunks,
    `index=${sizes.index_ntotal} map=${sizes.faiss_id_map} npy=${sizes.embeddings_rows} chunks=${sizes.chunks}`);
  gate.check("embedding model and dim are recorded",
    sizes.embedding_model === config.embedding.model && sizes.embedding_dim === config.embedding.dim,
    `${sizes.embedding_model} dim=${sizes.embedding_dim}`);

  // 2. Restart, cold-load, search. This is the faiss_id_map persistence test.
  gate.step("restarting in a separate process and searching cold");
  let hits: Hit[];
  try {
    hits = await restartAndSearch();
  } catch (error) {
    gate.check("index loads and searches after a restart", false, errorMessage(error).slice(0, 200));
    return 1;
  }

  gate.check("index loads and searches after a restart", hits.length > 0, `${hits.length} hits`);
  if (hits.length) {
    const byId = new Map(loadChunks(paths.chunks).map((c) => [c.chunk_id, c]));
    const resolved = hits.filter((h) => byId.has(h.chunk_id));
    gate.check("returned chunk_ids resolve to real chunks", resolved.length === hits.length, `${resolved.length}/${hits.length}`);
    const matched = hits.filter((h) => byId.get(h.chunk_id)?.text.startsWith(h.text.slice(0, 60)));
    gate.check("resolved text matches what the index returned", matched.length === hits.length,
      `${matched.length}/${hits.length} — proves the map did not shift`);
    gate.check("scores are ranked descending", hits.every((h, i) => i === 0 || (hits[i - 1]?.score ?? 0) >= h.score));

    console.log(`\n[${gate.elapsed()}] ---- top hits for ${JSON.stringify(QUERY)} ----`);
    for (const hit of hits) {
      console.log(`  ${hit.score.toFixed(4)}  ${hit.chunk_id}  [${hit.chunk_type}]`);
      console.log(`          ${hit.text.slice(0, 120)}...`);
    }
    console.log();
  }

  // 3. A new paper must add only its own vectors.
  const beforeIds = [...manifest.faiss_id_map];
  gate.step("fetching a 4th paper on a new topic to test incremental append");
  const fetched = await bridge.call<JsonObject>("collect_papers", { topic: NEW_TOPIC, topic_tag: "gate_m3", max_results: 1 });
  const fetchedAdded = (fetched.papers_added ?? []) as { paper_id: string }[];
  if (isToolError(fetched) || !fetchedAdded.length) {
    const detail = String(fetched.detail ?? "no new paper returned");
    console.log(`[${gate.elapsed()}] [SKIP] incremental append — could not fetch a 4th paper: ${detail}`);
  } else {
    const newPaper = fetchedAdded[0]!.paper_id;
    const ingested = await bridge.call<JsonObject>("ingest_paper", { paper_id: newPaper, describe: false });
    if (isToolError(ingested)) {
      gate.check("4th paper ingests", false, `${ingested.error}: ${ingested.detail}`);
    } else {
      gate.check("4th paper ingests", true, `${ingested.chunks_added} new chunks`);
      const again = await bridge.call<JsonObject>("index_chunks");
      gate.check("only the new paper was embedded",
        again.chunks_indexed === ingested.chunks_added && again.chunks_skipped === beforeIds.length,
        `indexed=${again.chunks_indexed} skipped=${again.chunks_skipped} (existing=${beforeIds.length})`);
      const after = loadManifest(paths.manifest);
      const kept = after.faiss_id_map.slice(0, beforeIds.length);
      gate.check("existing map entries kept their positions",
        kept.length === beforeIds.length && kept.every((id, i) => id === beforeIds[i]),
        "appending must not reorder what is already indexed");
    }
  }

  // 4. A changed embedding model must force a rebuild, not an append. The probe swaps the
  //    recorded model, asks model_changed, and restores the manifest in `finally`.
  gate.step("checking embedding-model mismatch detection");
  const { detected } = await bridge.call<{ detected: boolean }>("embedding_model_mismatch_probe");
  gate.check("a changed embedding model is detected", detected,
    "appending vectors from a different model would corrupt retrieval silently");

  return gate.finish("M3");
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
