/**
 * One live end-to-end check against arXiv: search, download one PDF, verify it is a PDF.
 *
 * In:  nothing (optional --topic, --clear-cooldown). Out: printed timings; exit 0 on
 *      success, 1 on failure, 2 while a recorded rate limit is still cooling off.
 * Deliberately fetches a single paper — the point is to prove the path works without
 * spending requests on a host that may still be rate-limiting this IP.
 */

import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type JsonObject } from "../app/bridge/types.js";
import { errorMessage, type GateConfig } from "./lib.js";

async function run(bridge: PythonBridge, topic: string, clearCooldown: boolean): Promise<number> {
  const { seconds_remaining: remaining } = await bridge.call<{ seconds_remaining: number }>(
    "arxiv_cooldown", { clear: clearCooldown },
  );
  if (remaining > 0) {
    console.log(`cooldown active: ${remaining.toFixed(0)}s left. Re-run later, or pass --clear-cooldown to try anyway.`);
    return 2;
  }

  const started = performance.now();
  const result = await bridge.call<JsonObject>("collect_papers", { topic, topic_tag: "_arxiv_check", max_results: 1 });
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);

  if (isToolError(result)) {
    console.log(`FAIL  ${result.error}: ${result.detail}  (${elapsed}s)`);
    return 1;
  }

  const added = (result.papers_added ?? []) as { paper_id: string }[];
  if (!added.length && result.papers_skipped) {
    console.log(`already held that paper; nothing downloaded. Search reached arXiv fine (${elapsed}s). `
      + "Delete data/papers to force a real download.");
    return 0;
  }
  const first = added[0];
  if (!first) {
    console.log(`FAIL  no papers and no skips: ${JSON.stringify(result)}`);
    return 1;
  }

  const config = await bridge.call<GateConfig>("gate_config");
  const pdf = join(config.paths.papers, `${first.paper_id}.pdf`);
  const isFile = existsSync(pdf) && statSync(pdf).isFile();
  const magic = isFile ? readFileSync(pdf).subarray(0, 5).toString("latin1") : "";
  const ok = magic.startsWith("%PDF");
  console.log(`${ok ? "OK  " : "FAIL"}  ${first.paper_id}  `
    + `${(isFile ? statSync(pdf).size : 0).toLocaleString("en-US")} B  magic=${JSON.stringify(magic)}  `
    + `(${elapsed}s for 1 search + 1 download)`);
  console.log(`      query: ${String(result.arxiv_query)}`);
  return ok ? 0 : 1;
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      topic: { type: "string", default: "graph neural networks for molecules" },
      "clear-cooldown": { type: "boolean", default: false },
    },
  });
  const bridge = new PythonBridge();
  try {
    process.exitCode = await run(bridge, String(values.topic), Boolean(values["clear-cooldown"]));
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  } finally {
    await bridge.close();
  }
}

void main();
