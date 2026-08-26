/**
 * The M9 verification gate: are the deliverables actually there, and do they say anything?
 *
 * In:  the repo (files only; no engine, so no bridge is started).
 * Out: pass/fail per deliverable. Checks content, not just existence — a README that never
 *      mentions the eval, or a write-up that omits the ablation, is the failure mode here.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, relative } from "node:path";

import { REPO_ROOT } from "../app/paths.js";
import { errorMessage } from "./lib.js";

const failures: string[] = [];

/** `detail` explains a failure and prints only on FAIL; `why` is rationale, always shown. */
function check(label: string, ok: boolean, detail = "", why = ""): void {
  const note = why || (!ok ? detail : "");
  console.log(`[${ok ? "PASS" : "FAIL"}] ${label}${note ? ` — ${note}` : ""}`);
  if (!ok) failures.push(label);
}

function read(path: string): string {
  const full = join(REPO_ROOT, path);
  return existsSync(full) ? readFileSync(full, "utf-8") : "";
}

function* walk(dir: string): Generator<string> {
  if (!existsSync(dir)) return;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "__pycache__" || entry.name === "node_modules") continue;
      yield* walk(full);
    } else if (entry.isFile()) {
      yield full;
    }
  }
}

function run(): number {
  console.log("=== M9: deliverables ===\n");

  // ---- the CLI ---------------------------------------------------------------------

  const cli = read("app/cli.ts");
  check("app/cli.ts exists", Boolean(cli), "", "eval/ imports src/, so the CLI cannot live in src/");

  const completed = spawnSync("npx", ["tsx", "app/cli.ts", "--help"], { cwd: REPO_ROOT, encoding: "utf-8" });
  const helpText = completed.stdout ?? "";
  check("app/cli.ts --help runs", completed.status === 0, (completed.stderr ?? String(completed.error ?? "")).trim().slice(0, 160));
  for (const command of ["index", "ask", "eval"]) {
    check(`CLI exposes \`${command}\``, helpText.includes(command));
  }

  // ---- README ----------------------------------------------------------------------

  const readme = read("README.md");
  check("README.md exists", Boolean(readme));
  // Two halves now: the TypeScript layer and the Python engine it drives.
  check("README documents setup", readme.includes("npm install") && readme.includes("pip install -r requirements.txt"));
  check("README documents the env vars", readme.includes("OPENROUTER_API_KEY"));
  for (const [what, command] of [["indexing", "index"], ["a question", "ask"], ["the eval", "eval"]] as const) {
    const documented = [`cli.ts ${command}`, `npm run cli -- ${command}`, `sciagent ${command}`]
      .some((phrase) => readme.includes(phrase));
    check(`README documents how to run ${what}`, documented);
  }

  // ---- cold-start demo -------------------------------------------------------------

  const cold = read("examples/cold_start.md");
  check("cold-start transcript exists", Boolean(cold), "run scripts/m9_cold_start.ts");
  check("cold-start transcript shows indexing from empty", cold.includes("no papers, no chunks"));
  check("cold-start transcript shows the tool calls", cold.includes("retrieve_evidence"));
  check("cold-start transcript is substantial", cold.length > 2000, `${cold.length} chars`);

  // ---- worked examples -------------------------------------------------------------

  const examples = read("examples/worked_examples.md");
  check("worked examples exist", Boolean(examples));
  check("worked examples include an abstention case", examples.includes("Abstention"));
  check("worked examples include a conflicting-evidence case", examples.includes("Conflicting evidence"));
  check("worked examples show resolved citations", /citations \d+\/\d+ resolved/.test(examples));

  // ---- eval write-up ---------------------------------------------------------------

  const evaluation = read("docs/EVALUATION.md");
  const evaluationLower = evaluation.toLowerCase();
  check("docs/EVALUATION.md exists", Boolean(evaluation), "", "ARCHITECTURE section 13 has always pointed at this file");
  check("write-up names the corpus that produced the numbers", evaluation.includes("406 chunks"));
  check("write-up separates tuning from held-out", evaluationLower.includes("held out"));
  check("write-up reports the rerank ablation", evaluationLower.includes("rerank"));
  check("write-up does not bury the ablation's cost", evaluation.includes("+0.027") && evaluation.includes("1527"),
    "", "the marginal gain and its latency cost must both be stated");
  check("write-up states what the numbers do not say",
    evaluationLower.includes("do not say") || evaluationLower.includes("limitation"));
  check("results JSON backing the write-up is present",
    ["tuning_split.json", "held_out_split.json", "ablation.json", "threshold_sweep.json"]
      .every((name) => existsSync(join(REPO_ROOT, "eval", "results", name))));

  // ---- the architecture section ----------------------------------------------------

  const architecture = read("docs/ARCHITECTURE.md");
  check("ARCHITECTURE has the 'how AI agents were used' section", architecture.includes("How AI agents were used"));
  check("that section gives concrete examples, not narrative",
    architecture.split("**").length - 1 > 20 && architecture.includes("256"),
    "", "expected specific measurements, e.g. the bi-encoder's real token limit");
  check("ARCHITECTURE's reference to EVALUATION.md now resolves",
    architecture.includes("docs/EVALUATION.md") && existsSync(join(REPO_ROOT, "docs", "EVALUATION.md")));

  // ---- standing rules --------------------------------------------------------------

  // Derived from eval/topics.yaml rather than hardcoded, so a topic added later is covered
  // without editing this gate. Matched on the first few words of each query: a whole
  // query string never appears verbatim, but its distinctive opening would.
  const phrases: string[] = [];
  for (const line of read("eval/topics.yaml").split("\n")) {
    const stripped = line.trim();
    if (stripped.startsWith("query:")) {
      const query = stripped.slice(stripped.indexOf(":") + 1).trim().replace(/^["']+|["']+$/g, "");
      phrases.push(query.toLowerCase().split(/\s+/).filter(Boolean).slice(0, 4).join(" "));
    }
  }
  check("eval topics are readable", phrases.length > 0, "eval/topics.yaml has no query: lines",
    `${phrases.length} topic(s) checked for`);

  // src/ is the rule's home; app/ and bridge/ are the new source trees around it and are
  // held to the same standard.
  const leaked: string[] = [];
  for (const tree of ["src", "app", "bridge"]) {
    for (const path of walk(join(REPO_ROOT, tree))) {
      if (![".py", ".md", ".ts"].includes(extname(path)) || !statSync(path).isFile()) continue;
      const text = readFileSync(path, "utf-8").toLowerCase();
      for (const phrase of phrases) {
        if (text.includes(phrase)) leaked.push(`${relative(REPO_ROOT, path)}: ${phrase}`);
      }
    }
  }
  check("no eval topic name leaked into src/, app/ or bridge/", !leaked.length, leaked.join("; "));

  console.log();
  if (failures.length) {
    console.log(`M9 GATE: ${failures.length} FAILURE(S) — ${failures.join("; ")}`);
    return 1;
  }
  console.log("M9 GATE: all checks passed");
  return 0;
}

function main(): void {
  try {
    process.exitCode = run();
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  }
}

main();
