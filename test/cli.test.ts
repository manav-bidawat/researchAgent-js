import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { STUB_PYTHON } from "./helpers.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const TSX = join(ROOT, "node_modules", ".bin", "tsx");

function cli(args: string[], env: NodeJS.ProcessEnv = {}) {
  const result = spawnSync(TSX, [join(ROOT, "app", "cli.ts"), ...args], {
    cwd: ROOT,
    encoding: "utf-8",
    env: { ...process.env, ...env },
    timeout: 30_000,
  });
  return { code: result.status, stdout: result.stdout, stderr: result.stderr };
}

describe("CLI", () => {
  it("--help lists every command", () => {
    const { code, stdout } = cli(["--help"]);
    expect(code).toBe(0);
    for (const command of ["index", "ask", "eval", "graph", "serve", "mcp"]) {
      expect(stdout).toMatch(new RegExp(`^\\s+${command}\\b`, "m"));
    }
    expect(stdout).toContain("--config <path>");
  });

  it("ask --help lists its flags", () => {
    const { code, stdout } = cli(["ask", "--help"]);
    expect(code).toBe(0);
    expect(stdout).toContain("<question>");
    expect(stdout).toContain("--show-tools");
    expect(stdout).toContain("--quiet");
  });

  it("index and mcp --help list their flags", () => {
    const index = cli(["index", "--help"]).stdout;
    for (const flag of ["--max-results", "--categories", "--no-describe", "--clear-arxiv-cooldown", "--rebuild"]) {
      expect(index).toContain(flag);
    }
    const mcp = cli(["mcp", "--help"]).stdout;
    expect(mcp).toContain("--transport");
    expect(mcp).toContain("streamable-http");
  });

  it("ask prints the answer on stdout and progress on stderr", () => {
    const { code, stdout, stderr } = cli(["ask", "what is attention?", "--show-tools"], { SCIAGENT_PYTHON: STUB_PYTHON });
    expect(code).toBe(0);
    expect(stdout.split("\n")[0]).toBe("Transformers use self-attention [p1__c0]. They scale well [p2__c3].");
    expect(stdout).toContain("2 iteration(s), 1 tool call(s), 1234 context tokens, stopped: answered");
    expect(stdout).toContain("--- trace: /tmp/stub-data/traces/run-stub-1.jsonl");
    expect(stdout).toMatch(/\[0\] retrieve_evidence\(.*\) -> 2 chunk\(s\), 12ms/);
    expect(stdout).not.toContain("thinking");

    expect(stderr).toContain("loading models and the index");
    expect(stderr).toContain("[1] thinking...");
    expect(stderr).toContain("[1] → retrieve_evidence(query='what is attention?') ... 2 chunks, 12ms");
    expect(stderr).toContain("[2] answering");
    expect(stderr).not.toContain("Transformers use");
  });

  it("ask --quiet prints no progress", () => {
    const { code, stdout, stderr } = cli(["ask", "q", "--quiet"], { SCIAGENT_PYTHON: STUB_PYTHON });
    expect(code).toBe(0);
    expect(stdout).toContain("Transformers use self-attention");
    expect(stderr).toBe("");
  });

  it("ask exits 1 on a loop failure, still printing the partial answer", () => {
    const { code, stdout, stderr } = cli(["ask", "fail", "--quiet"], { SCIAGENT_PYTHON: STUB_PYTHON });
    expect(code).toBe(1);
    expect(stdout.trim()).toBe("partial answer so far");
    expect(stderr).toContain("the agent loop failed: llm_unavailable — the provider timed out");
  });
});
