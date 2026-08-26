/**
 * The M1 verification gate from docs/BUILD_PLAN.md, run against live arXiv and the LLM.
 *
 * In:  a populated .env and an empty (or existing) data/ tree.
 * Out: prints a pass/fail line per check — fetch 3, refetch with 0 downloads, a second
 *      overlapping topic that backfills tags onto chunk records, and a network failure
 *      that returns an error dict rather than raising. Exits non-zero on failure.
 */

import { existsSync, statSync } from "node:fs";

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type EnginePaths, type JsonObject } from "../app/bridge/types.js";
import {
  Gate, chunksForPaper, errorMessage, loadChunks, loadManifest, papersForTopic, type GateConfig,
} from "./lib.js";

const TOPIC_A = "sparse mixture-of-experts routing in transformers";
// The backfill check re-runs TOPIC_A under a second tag rather than picking a different
// topic and hoping the two overlap. Overlap is the thing under test, so it has to be
// guaranteed, not left to whatever arXiv ranks that day.
const TAG_A = "gate_a";
const TAG_A_ALIAS = "gate_a_alias";

const FIELDS = new Set([
  "paper_id", "arxiv_id", "title", "authors", "abstract", "year", "published",
  "categories", "pdf_url", "pdf_path", "topic_tags", "n_chunks", "n_figures",
  "indexed_at", "parse_status", "parse_note",
]);

const isFile = (path: string): boolean => existsSync(path) && statSync(path).isFile();

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate();
  const paths = await bridge.call<EnginePaths>("paths");
  const config = await bridge.call<GateConfig>("gate_config");
  console.log(`papers dir: ${config.paths.papers}\n`);

  // 1. Fetch three papers on a fresh topic.
  gate.step("fetching 3 papers from arXiv (downloads PDFs, ~30-60s)");
  const first = await bridge.call<JsonObject>("collect_papers", { topic: TOPIC_A, topic_tag: TAG_A, max_results: 3 });
  if (isToolError(first)) {
    gate.check("fetch 3 papers", false, `${first.error}: ${first.detail}`);
    return 1;
  }
  const added = first.papers_added as { paper_id: string }[];
  let detail = `${added.length} added, query=${JSON.stringify(String(first.arxiv_query).slice(0, 60))}`;
  if ((first.failures as unknown[] | undefined)?.length) detail += `, failures=${JSON.stringify(first.failures)}`;
  gate.check("fetch 3 papers", added.length === 3, detail);
  gate.check("query was LLM-planned, not the fallback", !first.query_was_fallback);

  const manifest = loadManifest(paths.manifest);
  const records = added.map((p) => manifest.papers[p.paper_id]);
  gate.check("paper records are well formed", records.every((r) => r
    && Object.keys(r).length === FIELDS.size && Object.keys(r).every((key) => FIELDS.has(key))));
  gate.check("topic_tags is a list on every record", records.every((r) => Array.isArray(r?.topic_tags)));
  const onDisk = records.map((r) => Boolean(r && isFile(r.pdf_path)));
  gate.check("PDFs downloaded", onDisk.every(Boolean), `${onDisk.filter(Boolean).length}/${onDisk.length}`);

  // 2. Same topic again: nothing new should be downloaded.
  const mtimes = new Map(records.filter((r) => r && isFile(r.pdf_path)).map((r) => [r!.paper_id, statSync(r!.pdf_path).mtimeMs]));
  gate.step("refetching the same topic (should download nothing)");
  const second = await bridge.call<JsonObject>("collect_papers", { topic: TOPIC_A, topic_tag: TAG_A, max_results: 3 });
  const unchanged = records.every((r) => r && isFile(r.pdf_path) && statSync(r.pdf_path).mtimeMs === mtimes.get(r.paper_id));
  gate.check("refetch adds 0 papers",
    Array.isArray(second.papers_added) && (second.papers_added as unknown[]).length === 0,
    `skipped=${JSON.stringify(second.papers_skipped ?? null)}`);
  gate.check("refetch re-downloads nothing", unchanged);
  gate.check("planned query came from cache", second.query_was_cached === true);

  // 3. The same papers under a second tag must be tagged, not re-added, and the tag must
  //    reach their chunk records as well as the manifest.
  // Pick the target from what is indexed under TAG_A *now*, not from the first fetch: if
  // planning ever changes between calls, the first fetch's papers may not be the ones the
  // next search re-encounters.
  const indexed = papersForTopic(loadManifest(paths.manifest), TAG_A);
  const target = indexed[0] ?? added[0]?.paper_id ?? "";
  if (!chunksForPaper(loadChunks(paths.chunks), target).length) {
    // Chunking is M2; stand-in records let the backfill path be exercised now.
    await bridge.call("append_stand_in_chunks", { paper_id: target, topic_tag: TAG_A, n: 3 });
  }

  gate.step("same topic under a second tag (should tag, not add)");
  const third = await bridge.call<JsonObject>("collect_papers", { topic: TOPIC_A, topic_tag: TAG_A_ALIAS, max_results: 3 });
  if (isToolError(third)) {
    gate.check("same topic under a second tag", false, `${third.error}: ${third.detail}`);
  } else {
    const thirdAdded = third.papers_added as unknown[];
    gate.check("same papers are tagged, not re-added", thirdAdded.length === 0,
      `added=${thirdAdded.length}, skipped=${JSON.stringify(third.papers_skipped)}, tagged=${JSON.stringify(third.papers_tagged)}`);
    const reloaded = loadManifest(paths.manifest);
    const bothTags = [TAG_A, TAG_A_ALIAS];
    const overlapped = Object.values(reloaded.papers).filter((p) => bothTags.every((tag) => p.topic_tags.includes(tag)));
    gate.check("papers carry both tags in the manifest", overlapped.length === added.length,
      `${overlapped.length}/${added.length}`);
    const chunks = chunksForPaper(loadChunks(paths.chunks), target);
    gate.check("their chunk records were backfilled too",
      chunks.length > 0 && chunks.every((c) => bothTags.every((tag) => (c.topic_tags ?? []).includes(tag))),
      `${chunks.length} chunks for ${target}`);
    gate.check("no phantom ids in topics.paper_ids",
      bothTags.every((tag) => papersForTopic(reloaded, tag).every((pid) => reloaded.papers[pid] !== undefined)));
  }

  // 4. A dead network must return an error dict, not raise.
  gate.step("simulating a dead network");
  const dead = await bridge.call<{ raised: string | null; result: JsonObject | null }>(
    "collect_dead_network_probe", { topic: "anything at all", topic_tag: "gate_dead" },
  );
  if (dead.raised) {
    gate.check("network failure returns an error dict", false, `it raised ${dead.raised}`);
  } else {
    gate.check("network failure returns an error dict", dead.result?.error === "arxiv_unavailable", String(dead.result?.error));
  }

  return gate.finish("M1");
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
