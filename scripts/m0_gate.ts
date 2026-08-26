/**
 * The M0 verification gate from docs/BUILD_PLAN.md, run as a script.
 *
 * In:  a populated .env (OPENROUTER_API_KEY) and config.yaml; nothing else.
 * Out: prints a pass/fail line per check — imports, directory scaffolding, one live text
 *      completion, and one live tool-calling round trip — and exits non-zero on failure.
 */

import { PythonBridge } from "../app/bridge/client.js";
import { isToolError } from "../app/bridge/types.js";
import { Gate, errorMessage, type GateConfig } from "./lib.js";

const PROBE_TOOL = {
  type: "function",
  function: {
    name: "lookup_year",
    description: "Look up the publication year of a paper by its exact title.",
    parameters: {
      type: "object",
      properties: { title: { type: "string", description: "The paper title." } },
      required: ["title"],
    },
  },
};

interface Readiness {
  imports_ok: boolean;
  import_error: string | null;
  missing_directories: string[];
  system_prompt_mentions_retrieve_evidence: boolean;
  api_key_error: string | null;
}

interface ProbeResult {
  text: string;
  tool_calls: { id: string; name: string; arguments: Record<string, unknown>; ok: boolean }[];
  assistant_has_tool_calls: boolean;
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate(false);
  const config = await bridge.call<GateConfig>("gate_config");
  console.log(`config:  ${config.source_path}`);
  console.log(`models:  agent=${config.llm.agent_model}  vision=${config.llm.vision_model}  `
    + `utility=${config.llm.utility_model}\n`);

  const ready = await bridge.call<Readiness>("llm_readiness");
  gate.check("import config, llm_client", ready.imports_ok, ready.import_error ?? "");
  gate.check("directory scaffolding exists", !ready.missing_directories.length, ready.missing_directories.join(", "));
  gate.check("system prompt loads from file", ready.system_prompt_mentions_retrieve_evidence);

  if (ready.api_key_error) {
    gate.check("OPENROUTER_API_KEY is set", false, ready.api_key_error);
    console.log("\nSkipping the two live checks: no API key.");
    return 1;
  }

  const text = await bridge.call<ProbeResult | { error: string; detail: string }>("llm_probe", {
    content: "Reply with exactly: gate ok", role: "utility",
  });
  if (isToolError(text)) gate.check("live text completion", false, `${text.error}: ${text.detail}`);
  else gate.check("live text completion", Boolean(text.text.trim()), JSON.stringify(text.text.trim().slice(0, 60)));

  const tooled = await bridge.call<ProbeResult | { error: string; detail: string }>("llm_probe", {
    content: "What year was 'Attention Is All You Need' published? Use the lookup_year tool.",
    tools: [PROBE_TOOL],
    role: "agent",
  });
  if (isToolError(tooled)) {
    gate.check("live tool-calling round trip", false, `${tooled.error}: ${tooled.detail}`);
  } else {
    const calls = tooled.tool_calls;
    const first = calls[0];
    const wellFormed = calls.length === 1 && first?.name === "lookup_year" && first.ok
      && typeof first.arguments.title === "string";
    const detail = first
      ? `${first.name}(${JSON.stringify(first.arguments)})`
      : `no tool call; text=${JSON.stringify(tooled.text.slice(0, 60))}`;
    gate.check("live tool-calling round trip", Boolean(wellFormed), detail);
    gate.check("assistant turn round-trips tool_call ids", calls.length > 0 && tooled.assistant_has_tool_calls);
  }

  return gate.finish("M0");
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
