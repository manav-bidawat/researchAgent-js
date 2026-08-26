/**
 * The M6 verification gate: the four remaining tools, standalone then agent-selected.
 *
 * In:  an index built by M3 and a funded .env key.
 * Out: pass/fail per tool — documented return shape, an error dict on bad input — then
 *      live questions whose traces show the model actually selecting each tool.
 */

import { existsSync, statSync } from "node:fs";

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type AskFailure, type AskResult, type EnginePaths, type JsonObject } from "../app/bridge/types.js";
import { Gate, byString, errorMessage, loadManifest, squash } from "./lib.js";

interface Claim {
  claim: string;
  label: string;
  confidence: number;
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate();
  const paths = await bridge.call<EnginePaths>("paths");
  const manifest = loadManifest(paths.manifest);
  if (!Object.keys(manifest.papers).length) {
    console.log("Empty corpus. Run scripts/m3_gate.ts first.");
    return 1;
  }

  // ---- analyze_corpus ----------------------------------------------------
  gate.step("analyze_corpus, standalone");
  const analyze = (operation: string) => bridge.call<JsonObject>("analyze_corpus", { operation });
  for (const operation of ["stats", "timeline", "compare_topics", "cluster"]) {
    const result = await analyze(operation);
    const ok = !isToolError(result) && ["operation", "result", "summary"].every((key) => key in result);
    gate.check(`analyze_corpus(${operation}) returns the documented shape`, ok,
      String(result.error ?? String(result.summary ?? "").slice(0, 88)));
  }
  gate.check("analyze_corpus rejects an unknown operation", (await analyze("nonsense")).error === "unknown_operation");

  const cluster = await analyze("cluster");
  if (!isToolError(cluster)) {
    const body = cluster.result as JsonObject;
    gate.check("cluster is validated against known topic tags",
      ["adjusted_rand_index", "purity", "silhouette"].every((key) => key in body),
      `ARI=${body.adjusted_rand_index} purity=${body.purity} k=${body.k}`);
  }

  // ---- inspect_figure ----------------------------------------------------
  gate.step("inspect_figure, standalone");
  const inspect = (params: JsonObject) =>
    bridge.call<{ result: JsonObject; contains_bytes: boolean }>("inspect_figure_probe", params);
  const figureIds = Object.keys(manifest.figures).sort(byString);
  if (figureIds.length) {
    const { result: resolved, contains_bytes } = await inspect({ figure_id: figureIds[0] });
    const imagePath = String(resolved.image_path ?? "");
    const ok = !isToolError(resolved) && Boolean(imagePath) && existsSync(imagePath) && statSync(imagePath).isFile();
    gate.check("inspect_figure resolves an indexed figure to an image on disk", ok,
      String(resolved.error ?? `${resolved.figure_id} p${resolved.page}`));
    gate.check("inspect_figure returns a path, never image bytes", !contains_bytes);
  }
  gate.check("inspect_figure rejects a missing reference", (await inspect({})).result.error === "missing_reference");
  gate.check("inspect_figure rejects an unknown figure",
    (await inspect({ figure_id: "nope__f99" })).result.error === "unknown_figure");

  // ---- check_evidence_consistency ---------------------------------------
  gate.step("check_evidence_consistency, standalone (loads the NLI model)");
  const evidence = await bridge.call<{ chunks?: { chunk_id: string }[] }>("retrieve_evidence", {
    query: "how are experts selected for each token", k: 4,
  });
  const chunkIds = (evidence.chunks ?? []).map((c) => c.chunk_id);
  const check = (mode: string, ids: string[], answerText?: string) =>
    bridge.call<JsonObject>("check_evidence_consistency", { mode, chunk_ids: ids, answer_text: answerText ?? null });

  if (chunkIds.length >= 2) {
    const contradiction = await check("contradiction", chunkIds);
    const ok = !isToolError(contradiction) && "found" in contradiction && "conflicting_pairs" in contradiction;
    gate.check("contradiction mode returns the documented shape", ok,
      String(contradiction.error ?? `found=${contradiction.found}, ${contradiction.n_pairs_scored} pairs scored`));

    const draft = "The router selects experts for each token using a learned gating network. "
      + "The corpus reports that sourdough ferments best at 24 degrees.";
    const grounded = await check("groundedness", chunkIds, draft);
    const groundedOk = !isToolError(grounded)
      && ["claims", "grounded_ratio", "unsupported_claims"].every((key) => key in grounded);
    gate.check("groundedness mode returns the documented shape", groundedOk,
      String(grounded.error ?? `ratio=${grounded.grounded_ratio} `
        + `unsupported=${((grounded.unsupported_claims ?? []) as unknown[]).length}`));
    if (!isToolError(grounded)) {
      const claims = (grounded.claims ?? []) as Claim[];
      for (const claim of claims) {
        console.log(`        ${claim.label.padEnd(14)} ${claim.confidence.toFixed(3)}  ${claim.claim.slice(0, 74)}`);
      }
      gate.check("the fabricated claim is not marked entailed",
        claims.some((c) => c.claim.toLowerCase().includes("sourdough") && c.label !== "entailed"),
        "the planted unsupported claim must not pass");
    }
  }

  gate.check("check_evidence_consistency rejects an unknown mode",
    (await check("vibes", chunkIds.length ? chunkIds : ["x"])).error === "unknown_mode");

  // ---- the agent selecting tools -----------------------------------------
  const { names } = await bridge.call<{ names: string[] }>("registered_tools");
  gate.check("all five tools are registered", names.length === 5, names.join(", "));

  const questions: Record<string, string> = {
    analyze_corpus: "What topics does this corpus cover, and what years do the papers span?",
    retrieve_evidence: "How is the load-balancing loss defined in these papers?",
  };
  for (const [expected, question] of Object.entries(questions)) {
    gate.step(`live question expected to use ${expected}`);
    const result = await bridge.call<AskResult | AskFailure>("ask", { question, question_id: `m6_${expected}`, events: false });
    if (isToolError(result)) {
      gate.check(`the agent answers: ${expected}`, false, `${result.error}: ${result.detail}`);
      continue;
    }
    const answered = result as AskResult;
    const used = answered.trace_records.map((record) => record.tool_name);
    gate.check(`the agent chose ${expected}`, used.includes(expected),
      `called: ${used.length ? JSON.stringify(used) : "nothing"} in ${answered.iterations} iteration(s)`);
    console.log(`        answer: ${squash(answered.answer).slice(0, 150)}...`);
  }

  return gate.finish("M6");
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
