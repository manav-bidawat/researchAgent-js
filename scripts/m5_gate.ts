/**
 * The M5 verification gate from docs/BUILD_PLAN.md: one real question, end to end.
 *
 * In:  an index built by M3 and a funded .env key.
 * Out: the message array printed per iteration so the stateless re-send is visible, the
 *      answer, and the trace file for manual inspection. Exits non-zero on failure.
 */

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError, type AskResult, type EnginePaths, type JsonObject } from "../app/bridge/types.js";
import { Gate, errorMessage, loadChunks, readJsonl, squash, type GateConfig } from "./lib.js";

const QUESTION = "How do these papers decide which experts a token or patch gets routed to?";

const TRACE_FIELDS = new Set([
  "run_id", "question_id", "iteration", "tool_name", "args", "result_summary",
  "chunk_ids_returned", "error", "latency_ms", "timestamp",
]);

interface Message {
  role: string;
  content?: unknown;
  tool_call_id?: string;
  tool_calls?: { id: string; function?: { name?: string } }[];
}

interface RunWithMessages extends AskResult {
  messages: Message[];
  messages_in_context: number;
  elided_results: number;
}

interface TraceLine {
  iteration: number;
  tool_name: string;
  args: unknown;
  latency_ms: number;
  chunk_ids_returned: string[];
  result_summary: { chunks?: { text?: unknown }[] } & JsonObject;
  [key: string]: unknown;
}

/** One line summarising a message, for the conversation dump. */
function describe(message: Message): string {
  if (message.role === "tool") {
    return `tool(id=${message.tool_call_id}) ${String(message.content ?? "").length} chars`;
  }
  const calls = message.tool_calls ?? [];
  if (calls.length) {
    return `assistant -> tool_calls: ${calls.map((c) => `${c.function?.name}(id=${c.id})`).join(", ")}`;
  }
  const text = typeof message.content === "string" ? message.content : JSON.stringify(message.content);
  return `${message.role}: ${squash(text).slice(0, 80)}`;
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate();
  const paths = await bridge.call<EnginePaths>("paths");
  const config = await bridge.call<GateConfig>("gate_config");
  if (!loadChunks(paths.chunks).length) {
    console.log("No chunks indexed. Run scripts/m3_gate.ts first.");
    return 1;
  }

  console.log(`model: ${config.llm.agent_model}   max_iterations: ${config.agent.max_iterations}`);
  console.log(`question: ${QUESTION}\n`);

  // The default registry (retrieve_evidence alone), as M5 specifies.
  const result = await bridge.call<RunWithMessages | { error: string; detail: string }>(
    "agent_run_with_messages", { question: QUESTION, question_id: "m5_gate", registry: "default" },
  );
  if (isToolError(result)) {
    gate.check("the loop completes", false, `${result.error}: ${result.detail}`);
    return 1;
  }
  gate.check("the loop completes", true, `${result.iterations} iteration(s), stopped_because=${result.stopped_because}`);

  // The conversation is the whole memory. Print it so the re-send is visible.
  const messages = result.messages;
  console.log(`\n[${gate.elapsed()}] ---- final message array (${messages.length} messages) ----`);
  messages.forEach((message, position) => console.log(`  [${position}] ${describe(message)}`));

  const roles = messages.map((m) => m.role);
  gate.check("conversation starts with system then user", roles[0] === "system" && roles[1] === "user");

  // Every assistant tool_call must have exactly one matching tool message, or the API
  // would have rejected the request that carried them.
  const requested = messages.flatMap((m) => (m.tool_calls ?? []).map((c) => c.id)).sort();
  const answered = messages.filter((m) => m.role === "tool").map((m) => String(m.tool_call_id)).sort();
  gate.check("every tool_call has exactly one matching tool result",
    requested.length === answered.length && requested.every((id, i) => id === answered[i]),
    `${requested.length} requested, ${answered.length} answered`);
  gate.check("tool results are strings, not dicts",
    messages.filter((m) => m.role === "tool").every((m) => typeof m.content === "string"));

  gate.check("the model called at least one tool", result.tool_calls > 0, `${result.tool_calls} call(s)`);
  gate.check("an answer was produced", Boolean(result.answer.trim()), `${result.answer.length} chars`);

  console.log(`\n[${gate.elapsed()}] ---- answer ----\n`);
  console.log(result.answer);
  console.log();

  // Citations are what make the answer checkable. Not all questions warrant one, so this
  // is reported rather than enforced.
  const cited = loadChunks(paths.chunks).filter((c) => result.answer.includes(c.chunk_id));
  gate.step(`answer cites ${cited.length} chunk_id(s) inline`);

  // Trace file, per DATA_SCHEMA section 7.
  const records = readJsonl<TraceLine>(result.trace);
  gate.check("a trace record exists per tool call", records.length === result.tool_calls,
    `${records.length} records at ${result.trace}`);
  if (records.length) {
    gate.check("trace records match the documented schema", records.every((r) => {
      const keys = Object.keys(r);
      return keys.length === TRACE_FIELDS.size && keys.every((key) => TRACE_FIELDS.has(key));
    }), `fields: ${JSON.stringify(Object.keys(records[0]!).sort())}`);
    gate.check("chunk_ids_returned is populated for retrievals", records.some((r) => r.chunk_ids_returned?.length));
    gate.check("result_summary is truncated", records.every((r) => (r.result_summary?.chunks ?? [])
      .every((chunk) => String(chunk.text ?? "").length <= config.agent.trace_text_chars)));

    console.log(`\n[${gate.elapsed()}] ---- trace ----`);
    for (const record of records) {
      console.log(`  iter=${record.iteration} ${record.tool_name} ${record.latency_ms}ms args=${JSON.stringify(record.args).slice(0, 70)}`);
      console.log(`        -> ${record.chunk_ids_returned.length} chunks: ${JSON.stringify(record.chunk_ids_returned.slice(0, 4))}`);
    }
  }

  gate.step(`context: ${result.context_tokens} tokens across ${result.messages_in_context} messages, `
    + `${result.elided_results} elided`);

  return gate.finish("M5");
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
