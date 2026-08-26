/**
 * The eval question set and result persistence shared by every eval script.
 *
 * In:  eval/questions.json; a result object and a file stem.
 * Out: questions filtered to a split, and results written to eval/results/<name>.json.
 *      JSON has no NaN, so NaN is written as null — the same value Python's reader saw.
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { REPO_ROOT } from "../app/paths.js";

export interface Question {
  question_id: string;
  question: string;
  category: string;
  held_out: boolean;
  expect_abstention: boolean;
  gold_chunk_ids: string[];
  gold_paper_ids: string[];
  expected_facts: string[];
  expected_tools: string[];
  [key: string]: unknown;
}

export const RESULTS_DIR = join(REPO_ROOT, "eval", "results");

/**
 * The question set, optionally filtered. heldOut=false is the tuning split; the held-out
 * third is run once, after tuning stops — looking earlier turns it into training data.
 */
export function loadQuestions(heldOut?: boolean): Question[] {
  const payload = JSON.parse(readFileSync(join(REPO_ROOT, "eval", "questions.json"), "utf-8")) as { questions: Question[] };
  if (heldOut === undefined) return payload.questions;
  return payload.questions.filter((question) => Boolean(question.held_out) === heldOut);
}

export function saveResult(result: unknown, name: string): string {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const out = join(RESULTS_DIR, `${name}.json`);
  const text = JSON.stringify(result, (_key, value) => (typeof value === "number" && Number.isNaN(value) ? null : value), 2);
  writeFileSync(out, `${text}\n`, "utf-8");
  return out;
}
