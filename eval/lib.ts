/**
 * Helpers shared by the eval scripts and the gates: NaN-tolerant JSON reading, Python-style
 * number formatting, the balanced gate score, and writes that never clobber evidence.
 *
 * In:  file paths, numbers, and result objects about to be written.
 * Out: parsed records, fixed-width strings, and the path a result actually landed at.
 */

import { existsSync, readFileSync } from "node:fs";
import { basename, dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { RESULTS_DIR, saveResult } from "./questions.js";

/**
 * JSON.parse that accepts the bare NaN / Infinity tokens Python's json writes. Only tokens
 * outside string literals are rewritten, so a chunk whose text says "NaN" is untouched.
 */
export function parseJsonText<T = unknown>(text: string): T {
  const cleaned = text.replace(/"(?:[^"\\]|\\.)*"|-?\bNaN\b|-?\bInfinity\b/g, (token) => (token.startsWith('"') ? token : "null"));
  return JSON.parse(cleaned) as T;
}

export function readJson<T = unknown>(path: string): T {
  return parseJsonText<T>(readFileSync(path, "utf-8"));
}

/** Every record of a JSON-lines file; a missing file reads as empty, as ChunkStore does. */
export function readJsonl<T = Record<string, unknown>>(path: string): T[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf-8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => parseJsonText<T>(line));
}

/** Python's `f"{x:>{width}.{digits}f}"`, with NaN and null printed as "nan". */
export function fixed(value: number | null | undefined, digits: number, width = 0): string {
  const text = value === null || value === undefined || Number.isNaN(value) ? "nan" : value.toFixed(digits);
  return text.padStart(width);
}

/** Python's `f"{x:+.{digits}f}"` / `f"{x:+d}"`. */
export function signed(value: number, digits = 0): string {
  if (Number.isNaN(value)) return "nan";
  const text = value.toFixed(digits);
  return value >= 0 ? `+${text}` : text;
}

/**
 * Both halves of the gate matter and they trade off: a permissive gate always answers and
 * never abstains, a strict one abstains on everything. The harmonic mean punishes a
 * configuration that wins one half by abandoning the other.
 */
export function balanced(answerable: number, absent: number): number {
  if (answerable + absent === 0 || Number.isNaN(answerable + absent)) return 0;
  return Number(((2 * answerable * absent) / (answerable + absent)).toFixed(4));
}

export interface SweepRow {
  threshold: number;
  gate_on_answerable: number;
  gate_on_absent: number;
  balanced: number;
  "recall@5": number;
  mrr: number;
  gold_paper_hit_rate: number;
}

/** Python's `max(rows, key=lambda r: (r["balanced"], r["mrr"]))`: first row wins a tie. */
export function bestRow(rows: SweepRow[]): SweepRow | undefined {
  const key = (value: number) => (Number.isNaN(value) ? Number.NEGATIVE_INFINITY : value);
  let best: SweepRow | undefined;
  for (const row of rows) {
    if (!best || key(row.balanced) > key(best.balanced)
      || (key(row.balanced) === key(best.balanced) && key(row.mrr) > key(best.mrr))) {
      best = row;
    }
  }
  return best;
}

/**
 * Where a measurement should be written. The committed result files are what
 * docs/EVALUATION.md cites by number, so an existing file is only replaced when the caller
 * says so; otherwise the new result goes beside it as `<name>.rerun<ext>` for comparison.
 */
export function evidencePath(target: string, overwrite: boolean): string {
  if (overwrite || !existsSync(target)) return target;
  const ext = extname(target);
  const alternate = join(dirname(target), `${basename(target, ext)}.rerun${ext}`);
  process.stderr.write(`note: ${target} already exists and is committed evidence; writing to `
    + `${alternate} instead (pass --overwrite to replace it)\n`);
  return alternate;
}

/** saveResult, routed through evidencePath. Returns the path written. */
export function saveEvidence(result: unknown, name: string, overwrite: boolean): string {
  const target = evidencePath(join(RESULTS_DIR, `${name}.json`), overwrite);
  return saveResult(result, basename(target, ".json"));
}

/** True when the module at `metaUrl` is the script being run, not an import of it. */
export function isEntryPoint(metaUrl: string): boolean {
  const script = process.argv[1];
  return Boolean(script) && resolve(script as string) === fileURLToPath(metaUrl);
}
