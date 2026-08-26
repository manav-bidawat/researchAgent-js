/**
 * Reports whether the GPU path is live, and times the three models that would use it.
 *
 * In:  nothing (optional --skip-bench). Out: printed device, driver state and timings;
 *      exit 0 when CUDA is in use, 1 when the run fell back to CPU. Run it before and after
 *      installing the NVIDIA driver — the numbers are directly comparable.
 */

import { parseArgs } from "node:util";

import { PythonBridge } from "../app/bridge/client.js";
import { errorMessage } from "./lib.js";

interface DeviceReport {
  compute_device: string;
  torch_error?: string;
  torch_version?: string;
  built_for_cuda?: string | null;
  cuda_available?: boolean;
  devices?: { index: number; name: string; total_memory_gib: number; sm: string }[];
  reason?: string;
  resolved_device: string | null;
  resolve_error?: string;
}

interface BenchResult {
  texts_source: string;
  n_texts: number;
  bi_encoder_load_s: number;
  embed_s: number;
  rerank_s: number;
  n_rerank_pairs: number;
  nli_s: number;
  n_nli_pairs: number;
}

// Used only when no corpus has been indexed yet. Real chunk text is preferred because
// synthetic text makes the GPU look better than it is (see benchmark_models in
// bridge/ops/gates.py); this filler is sized to the measured corpus mean of ~278 tokens.
const FILLER = "Mixture-of-experts routing sends each token to a small subset of experts, "
  + "which keeps the parameter count high while holding inference cost flat. ";
const FALLBACK_TEXT = FILLER.repeat(14);
const FALLBACK_COUNT = 406;
const RERANK_QUERY = "how does expert routing affect accuracy";

/** What torch can see, and why it cannot see a GPU when it cannot. */
function printDriverReport(report: DeviceReport): void {
  if (report.torch_error) {
    console.log(`  torch            NOT IMPORTABLE: ${report.torch_error}`);
    return;
  }
  console.log(`  torch            ${report.torch_version}`);
  console.log(`  built for CUDA   ${report.built_for_cuda || "cpu-only wheel"}`);
  console.log(`  cuda.is_available ${report.cuda_available ? "True" : "False"}`);
  for (const device of report.devices ?? []) {
    console.log(`  device ${device.index}         ${device.name}, ${device.total_memory_gib.toFixed(1)} GiB, sm_${device.sm}`);
  }
  if (!report.cuda_available && report.reason) console.log(`  reason           ${report.reason}`);
}

const secs = (value: number): string => value.toFixed(2).padStart(6);

async function run(bridge: PythonBridge, skipBench: boolean): Promise<number> {
  const report = await bridge.call<DeviceReport>("device_report");
  console.log("driver / torch");
  printDriverReport(report);

  console.log("\nconfig");
  console.log(`  compute.device   ${report.compute_device}`);
  if (!report.resolved_device) {
    // resolve_device raises on an explicitly named, unusable device: a misconfiguration.
    console.log(`  resolve failed   ${report.resolve_error}`);
    return 1;
  }
  const device = report.resolved_device;
  console.log(`  resolved to      ${device}`);

  if (!skipBench) {
    console.log(`\nbenchmark on ${device}`);
    const bench = await bridge.call<BenchResult>("benchmark_models", {
      device, fallback_text: FALLBACK_TEXT, fallback_count: FALLBACK_COUNT, query: RERANK_QUERY,
    });
    console.log(`  bi-encoder load   ${secs(bench.bi_encoder_load_s)}s`);
    console.log(`  EMBED ${bench.n_texts} chunks ${secs(bench.embed_s)}s   (index time)`);
    console.log(`  RERANK ${bench.n_rerank_pairs} pairs  ${secs(bench.rerank_s)}s   (per query)`);
    console.log(`  NLI ${bench.n_nli_pairs} pairs     ${secs(bench.nli_s)}s   (per check)`);
    if (bench.texts_source !== "corpus") console.log("  (no corpus indexed yet: synthetic text of representative length)");
  }

  if (device.startsWith("cuda")) {
    console.log("\nGPU is in use.");
    return 0;
  }
  console.log("\nRunning on CPU. Install the NVIDIA driver to use the GPU; "
    + "compute.device is 'auto', so nothing else needs to change.");
  return 1;
}

async function main(): Promise<void> {
  const { values } = parseArgs({ options: { "skip-bench": { type: "boolean", default: false } } });
  const bridge = new PythonBridge();
  try {
    process.exitCode = await run(bridge, Boolean(values["skip-bench"]));
  } catch (error) {
    process.stderr.write(`${errorMessage(error)}\n`);
    process.exitCode = 1;
  } finally {
    await bridge.close();
  }
}

void main();
