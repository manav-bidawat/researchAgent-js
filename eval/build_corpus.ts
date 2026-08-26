/**
 * Builds the eval corpus: two adjacent-but-distinct topics into one shared index.
 *
 * In:  eval/topics.yaml (parsed by the bridge's eval_topics op).
 * Out: data/ populated and indexed, and a printed manifest of what landed, so the eval
 *      questions can be written from the abstracts before the system is ever run on them.
 */

import { existsSync } from "node:fs";

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type EnginePaths, type JsonObject } from "../app/bridge/types.js";
import { isEntryPoint, readJson } from "./lib.js";

interface TopicSettings {
  query: string;
  max_results?: number;
  categories?: string[];
}

interface PaperRecord {
  year: number;
  topic_tags: string[];
  title: string;
  n_chunks: number;
  n_figures: number;
  parse_status: string;
  abstract: string;
}

async function runMain(bridge: PythonBridge): Promise<number> {
  const spec = await bridge.call<{ topics: Record<string, TopicSettings> }>("eval_topics");

  for (const [tag, settings] of Object.entries(spec.topics ?? {})) {
    console.log(`\n=== ${tag}: ${JSON.stringify(settings.query)}`);
    let result = await bridge.call<JsonObject>("search_literature", {
      query: settings.query,
      max_results: settings.max_results ?? null,
      categories: settings.categories ?? null,
    });
    let rateLimited = false;
    if (isToolError(result)) {
      const partial = (result.partial ?? {}) as JsonObject;
      const added = partial.papers_added as unknown[] | undefined;
      if (result.error !== "arxiv_rate_limited" || !added?.length) {
        console.log(`    FAILED ${result.error}: ${result.detail}`);
        if (result.error === "arxiv_rate_limited") {
          // Every remaining topic would fast-fail off the cooldown memo anyway.
          console.log("    stopping: arXiv is rate-limiting this IP");
          break;
        }
        continue;
      }
      // A 429 partway through still indexed whole papers. Report and tag them the same
      // way a clean run would, then stop — reporting the topic as a bare FAILED would
      // leave them counted by analyze_corpus but absent from every topic_filter for this tag.
      console.log(`    RATE LIMITED partway: ${result.detail}`);
      result = partial;
      rateLimited = true;
    }
    // search_literature tags by a slug of the query; the tag printed here is the one the
    // questions must reference.
    console.log(`    added ${(result.papers_added as unknown[] | undefined)?.length ?? 0}, `
      + `skipped ${String(result.papers_skipped)}, chunks ${String(result.chunks_added)}, `
      + `figures ${String(result.figures_added)}`);
    console.log(`    tagged ${JSON.stringify(result.topic_tag)}`);
    if ((result.failures as unknown[] | undefined)?.length) {
      console.log(`    failures: ${JSON.stringify(result.failures)}`);
    }
    if (rateLimited) {
      console.log("    stopping: arXiv is rate-limiting this IP");
      break;
    }
  }

  console.log("\n=== indexing");
  console.log("   ", JSON.stringify(await bridge.call("index_chunks")));

  const { manifest: manifestPath } = await bridge.call<EnginePaths>("paths");
  const manifest = existsSync(manifestPath)
    ? readJson<{ papers?: Record<string, PaperRecord> }>(manifestPath)
    : { papers: {} };
  const papers = Object.entries(manifest.papers ?? {}).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  console.log(`\n=== corpus: ${papers.length} papers`);
  for (const [paperId, paper] of papers) {
    console.log(`\n  ${paperId}  (${paper.year})  tags=${JSON.stringify(paper.topic_tags)}`);
    console.log(`    ${paper.title}`);
    console.log(`    chunks=${paper.n_chunks} figures=${paper.n_figures} status=${paper.parse_status}`);
    console.log(`    ABSTRACT: ${(paper.abstract ?? "").slice(0, 700)}`);
  }
  return 0;
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
