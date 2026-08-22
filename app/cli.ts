#!/usr/bin/env node
/**
 * The command-line entry point: index a topic, ask a question, run the eval, sync the
 * graph, and start the web or MCP server.
 *
 * In:  argv. Out: printed results and a process exit code. Answers go to stdout and
 *      progress to stderr, so `ask ... > answer.txt` captures the answer alone.
 */

import { Command, Option } from "commander";
import { resolve } from "node:path";

import { PythonBridge } from "./bridge/client.js";
import { isToolError, type AskFailure, type AskResult, type EnginePaths, type IndexResult, type JsonObject, type LoopEvent, type ToolError } from "./bridge/types.js";
import { ConsoleReporter } from "./cli/progress.js";
import { serveHttp, serveStdio } from "./mcp/server.js";
import { serve } from "./web/server.js";
import { printMetrics, runSplit, save } from "../eval/runner.js";

function reportError(stage: string, result: ToolError): number {
  process.stderr.write(`\n${stage} failed: ${result.error} — ${result.detail ?? ""}\n`);
  return 1;
}

const num = (value: unknown): number => Number(value ?? 0);

async function cmdIndex(bridge: PythonBridge, topic: string, opts: JsonObject): Promise<number> {
  const result = await bridge.call<IndexResult | (ToolError & { stage: string })>(
    "index",
    {
      topic,
      max_results: opts.maxResults ?? null,
      categories: opts.categories ?? null,
      no_describe: Boolean(opts.describe === false),
      clear_arxiv_cooldown: Boolean(opts.clearArxivCooldown),
      rebuild: Boolean(opts.rebuild),
    },
    (event) => {
      if (event.kind === "note") {
        (event.warn ? process.stderr : process.stdout).write(`      ${String(event.text)}\n`);
      } else if (event.kind === "stage") {
        const label = { collect: `searching arXiv for '${topic}'`, ingest: "extracting text and figures",
          index: "embedding and indexing" }[String(event.stage)];
        console.log(`[${event.step}/${event.of}] ${label} ...`);
      } else if (event.kind === "collected") {
        const c = event.result as JsonObject;
        console.log(`      ${(c.papers_added as unknown[] | undefined)?.length ?? 0} added, `
          + `${num(c.papers_skipped)} already held, ${num(c.papers_tagged)} re-tagged, tag '${String(c.topic_tag)}'`);
      } else if (event.kind === "ingested") {
        const i = event.result as JsonObject;
        console.log(`      ${num(i.chunks_total)} chunks, ${num(i.figures_total)} figures`);
      }
    },
  );
  if (isToolError(result)) return reportError(String(result.stage ?? "indexing"), result);

  const indexed = result.indexed;
  console.log(`      ${num(indexed.chunks_indexed)} embedded, ${num(indexed.chunks_skipped)} already indexed, `
    + `${num(indexed.cache_hits)} cache hit(s), ${num(indexed.total_indexed)} vectors in the index`);
  if (result.graph) {
    if (isToolError(result.graph)) process.stderr.write(`      graph sync skipped: ${result.graph.detail}\n`);
    else console.log(`      graph synced: ${num(result.graph.chunks)} chunks`);
  }
  if (result.rate_limited) {
    // Non-zero: what did land is indexed, but the topic is short of what was asked for,
    // and a caller scripting this must not read that as a complete collection.
    process.stderr.write("\ncollection was cut short by an arXiv rate limit; "
      + "re-run this command once it lifts to fetch the rest\n");
    return 1;
  }
  return 0;
}

async function cmdAsk(bridge: PythonBridge, question: string, opts: JsonObject): Promise<number> {
  let reporter: ConsoleReporter | undefined;
  if (!opts.quiet) {
    const paths = await bridge.call<EnginePaths>("paths");
    reporter = new ConsoleReporter(paths.progress_text_chars);
    // Loading the models is the longest silence in the command, so say so here.
    process.stderr.write("loading models and the index ...\n");
  }
  const result = await bridge.call<AskResult | AskFailure>(
    "ask", { question, events: Boolean(reporter) }, reporter ? (e) => reporter.handle(e as LoopEvent) : undefined,
  );
  if (isToolError(result)) {
    const failure = result as AskFailure;
    if (failure.partial?.answer) console.log(failure.partial.answer);
    return reportError("the agent loop", failure);
  }
  const answer = result as AskResult;
  console.log(answer.answer);
  console.log(`\n--- ${answer.iterations} iteration(s), ${answer.tool_calls} tool call(s), `
    + `${answer.context_tokens} context tokens, stopped: ${answer.stopped_because}`);
  console.log(`--- trace: ${answer.trace}`);
  if (opts.showTools) {
    for (const record of answer.trace_records) {
      const summary = `${(record.chunk_ids_returned ?? []).length} chunk(s)`;
      console.log(`    [${record.iteration}] ${record.tool_name}(${JSON.stringify(record.args ?? {}).slice(0, 80)}) `
        + `-> ${record.error || summary}, ${record.latency_ms}ms`);
    }
  }
  return 0;
}

async function cmdEval(bridge: PythonBridge, opts: JsonObject): Promise<number> {
  const heldOut = Boolean(opts.heldOut);
  const limit = opts.limit ? Number(opts.limit) : undefined;
  const split = heldOut ? "held_out_split" : "tuning_split";
  // A truncated run must not overwrite the full one: docs/EVALUATION.md cites the
  // committed tuning_split.json by number.
  const name = limit ? `${split}_limit${limit}` : split;
  const result = await runSplit(bridge, { heldOut, limit, checkGroundedness: opts.groundedness !== false });
  const path = save(result, name);
  printMetrics(split, result);
  console.log(`\nwritten to ${path}`);
  return 0;
}

async function cmdGraphSync(bridge: PythonBridge): Promise<number> {
  const result = await bridge.call<JsonObject>("graph_sync");
  if (isToolError(result)) return reportError("graph sync", result);
  console.log(`synced ${num(result.papers)} paper(s), ${num(result.chunks)} chunk(s), `
    + `${num(result.topics)} topic(s), and ${num(result.figures)} figure(s) to Neo4j`);
  return 0;
}

function buildProgram(run: (fn: (bridge: PythonBridge) => Promise<number>, keepAlive?: boolean) => Promise<void>): Command {
  const program = new Command("sciagent")
    .description("Agentic research assistant over scientific papers.")
    .option("--config <path>", "config.yaml to use (default: ./config.yaml, or $SCIAGENT_CONFIG). "
      + "A config with different paths.* gives a separate corpus.");

  program.command("index")
    .description("collect, extract and index a topic")
    .argument("<topic>", "what to search arXiv for, in plain language")
    .option("--max-results <n>", "papers to fetch (default: collection.max_results_default)", (v) => Number.parseInt(v, 10))
    .option("--categories <cats...>", "restrict to arXiv categories, e.g. cs.CL cs.LG")
    .option("--no-describe", "skip vision calls; figures index on caption text alone")
    .option("--clear-arxiv-cooldown", "forget a recorded arXiv rate limit and search anyway")
    .option("--rebuild", "rebuild the vector index from scratch")
    .action((topic: string, opts: JsonObject) => run((b) => cmdIndex(b, topic, opts)));

  program.command("ask")
    .description("answer one question against the index")
    .argument("<question>", "the question, quoted")
    .option("--show-tools", "after the answer, print a one-line summary of every tool call")
    .option("--quiet", "suppress the live progress lines printed to stderr")
    .action((question: string, opts: JsonObject) => run((b) => cmdAsk(b, question, opts)));

  program.command("eval")
    .description("run the eval harness")
    .option("--held-out", "run the held-out third instead of the tuning split")
    .option("--limit <n>", "run only the first N questions", (v) => Number.parseInt(v, 10))
    .option("--no-groundedness", "skip the NLI groundedness pass")
    .action((opts: JsonObject) => run((b) => cmdEval(b, opts)));

  program.command("graph")
    .description("manage the optional Neo4j corpus graph")
    .command("sync")
    .description("upsert the local corpus into Neo4j")
    .action(() => run((b) => cmdGraphSync(b)));

  program.command("serve")
    .description("start the Express demo web server")
    .option("--host <host>", "bind address", "127.0.0.1")
    .option("--port <port>", "port", (v) => Number.parseInt(v, 10), 8000)
    .action((opts: { host: string; port: number }) =>
      run(async (b) => { await serve(b, opts.host, opts.port); return 0; }, true));

  program.command("mcp")
    .description("start the MCP server")
    .addOption(new Option("--transport <kind>", "wire transport").choices(["stdio", "streamable-http"]).default("stdio"))
    .option("--host <host>", "bind address for streamable-http", "127.0.0.1")
    .option("--port <port>", "port for streamable-http", (v) => Number.parseInt(v, 10), 8001)
    .action((opts: { transport: string; host: string; port: number }) =>
      run(async (b) => {
        if (opts.transport === "stdio") await serveStdio(b);
        else await serveHttp(b, opts.host, opts.port);
        return 0;
      }, true));

  return program;
}

async function main(argv: string[]): Promise<void> {
  const program = buildProgram(async (fn, keepAlive = false) => {
    const configOpt = program.opts<{ config?: string }>().config;
    // MCP stdio owns stdout, and the worker's stderr is diagnostics either way.
    const bridge = new PythonBridge({ configPath: configOpt ? resolve(configOpt) : undefined });
    try {
      process.exitCode = await fn(bridge);
    } catch (error) {
      process.stderr.write(`\n${error instanceof Error ? error.message : String(error)}\n`);
      process.exitCode = 1;
    } finally {
      if (!keepAlive || process.exitCode) await bridge.close();
    }
  });
  await program.parseAsync(argv);
}

void main(process.argv);
