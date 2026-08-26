/**
 * The M7 verification gate: the eval corpus and the question set are sound.
 *
 * In:  data/ built by eval/build_corpus.ts and eval/questions.json annotated (the
 *      manifest and chunk paths come from the bridge; both files are read directly).
 * Out: pass/fail per requirement — gold labels present, all four categories covered, and
 *      no question copying paper wording, which is checked by n-gram overlap not by eye.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { PythonBridge } from "../app/bridge/client.js";
import type { EnginePaths } from "../app/bridge/types.js";
import { REPO_ROOT } from "../app/paths.js";
import type { Question } from "../eval/questions.js";
import { Gate, byString, errorMessage, loadChunks, loadManifest } from "./lib.js";

const CATEGORIES = new Set(["single_paper", "multi_paper", "not_in_corpus", "conflicting"]);
const NGRAM = 5;

function ngrams(text: string, n = NGRAM): Set<string> {
  const words = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  const out = new Set<string>();
  for (let i = 0; i + n <= words.length; i += 1) out.add(words.slice(i, i + n).join(" "));
  return out;
}

async function run(bridge: PythonBridge): Promise<number> {
  const gate = new Gate(false);
  const payload = JSON.parse(readFileSync(join(REPO_ROOT, "eval", "questions.json"), "utf-8")) as {
    questions: (Question & { absent_markers?: string[] })[];
  };
  const questions = payload.questions;
  const paths = await bridge.call<EnginePaths>("paths");
  const manifest = loadManifest(paths.manifest);
  const chunks = loadChunks(paths.chunks);
  const papers = Object.values(manifest.papers);

  console.log(`corpus: ${papers.length} papers, ${chunks.length} chunks, ${Object.keys(manifest.figures).length} figures\n`);

  // 1. Two topics, one index.
  const tags = [...new Set(papers.flatMap((paper) => paper.topic_tags))].sort(byString);
  const countFor = (tag: string) => papers.filter((p) => p.topic_tags.includes(tag)).length;
  gate.check("two contrasting topics are indexed", tags.length === 2, tags.map((t) => t.slice(0, 38)).join(", "));
  gate.check("both topics have papers", tags.every((tag) => countFor(tag) >= 4),
    tags.map((tag) => `${tag.slice(0, 26)}=${countFor(tag)}`).join(", "));
  gate.check("both topics share one index",
    new Set(chunks.map((c) => c.chunk_id)).size === chunks.length && chunks.length > 0,
    `${chunks.length} chunks in a single store`);

  // 2. Question set shape.
  const counts: Record<string, number> = {};
  for (const q of questions) counts[q.category] = (counts[q.category] ?? 0) + 1;
  const seen = Object.keys(counts);
  gate.check("12-15 questions", questions.length >= 12 && questions.length <= 15, String(questions.length));
  gate.check("all four categories are represented",
    seen.length === CATEGORIES.size && seen.every((c) => CATEGORIES.has(c)), JSON.stringify(counts));
  gate.check("at least two not_in_corpus", (counts.not_in_corpus ?? 0) >= 2, String(counts.not_in_corpus ?? 0));
  gate.check("at least two conflicting", (counts.conflicting ?? 0) >= 2, String(counts.conflicting ?? 0));
  const held = questions.filter((q) => q.held_out).length;
  gate.check("roughly a third is held out", held / questions.length >= 0.2 && held / questions.length <= 0.45,
    `${held}/${questions.length}`);
  const heldCategories = [...new Set(questions.filter((q) => q.held_out).map((q) => q.category))].sort(byString);
  gate.check("held-out set covers more than one category", heldCategories.length >= 2, JSON.stringify(heldCategories));

  // 3. Gold labels.
  const knownChunks = new Set(chunks.map((c) => c.chunk_id));
  const missingGold = questions.filter((q) => !q.expect_abstention && !q.gold_chunk_ids.length).map((q) => q.question_id);
  gate.check("every answerable question has gold chunks", !missingGold.length, JSON.stringify(missingGold));
  gate.check("abstention questions have no gold chunks",
    questions.filter((q) => q.expect_abstention).every((q) => !q.gold_chunk_ids.length));
  const badPapers = questions.flatMap((q) => q.gold_paper_ids.filter((p) => !(p in manifest.papers)).map((p) => [q.question_id, p]));
  gate.check("every gold paper_id resolves", !badPapers.length, JSON.stringify(badPapers.slice(0, 4)));
  const badChunks = questions.flatMap((q) => q.gold_chunk_ids.filter((c) => !knownChunks.has(c)).map((c) => [q.question_id, c]));
  gate.check("every gold chunk_id resolves", !badChunks.length, JSON.stringify(badChunks.slice(0, 4)));
  const mismatched = questions.flatMap((q) => q.gold_chunk_ids
    .filter((c) => !q.gold_paper_ids.includes(c.includes("__c") ? c.slice(0, c.lastIndexOf("__c")) : c))
    .map(() => q.question_id));
  gate.check("gold chunks belong to their gold papers", !mismatched.length, JSON.stringify([...new Set(mismatched)].sort(byString)));
  gate.check("every question has expected_facts or expects abstention",
    questions.every((q) => q.expected_facts.length || q.expect_abstention));

  // 4. No question copies paper wording. Checked, not asserted.
  const corpusNgrams = new Set<string>();
  for (const chunk of chunks) for (const gram of ngrams(chunk.text)) corpusNgrams.add(gram);
  const copied: [string, string[]][] = [];
  for (const question of questions) {
    const overlap = [...ngrams(question.question)].filter((gram) => corpusNgrams.has(gram));
    if (overlap.length) copied.push([question.question_id, overlap.sort(byString).slice(0, 2)]);
  }
  gate.check(`no question copies a ${NGRAM}-gram from the corpus`, !copied.length, JSON.stringify(copied.slice(0, 3)));

  // 5. Abstention questions must genuinely be absent, not merely assumed absent.
  //
  // Checked on declared subject markers, not on every word. An earlier version flagged
  // any shared vocabulary, which fails exactly the questions that are supposed to share
  // it: ni02 deliberately reuses "quantization", "accuracy" and "transformers" so that
  // the relevance gate has to react to meaning rather than to surface overlap. What must
  // be absent is the subject, so the subject is what gets named and tested.
  const corpusText = chunks.map((c) => c.text.toLowerCase()).join(" ");
  for (const question of questions) {
    if (!question.expect_abstention) continue;
    const markers = question.absent_markers ?? [];
    gate.check(`${question.question_id} declares what makes it absent`, markers.length > 0);
    const present = markers.filter((m) => corpusText.includes(m.toLowerCase()));
    gate.check(`${question.question_id} subject really is absent from the corpus`, !present.length,
      present.length ? `found in corpus: ${JSON.stringify(present)}` : `markers: ${JSON.stringify(markers)}`);
  }

  // A shared-vocabulary abstention case is what makes the gate meaningful, so confirm at
  // least one exists rather than only trivially absurd questions.
  const corpusWords = new Set<string>();
  for (const chunk of chunks) for (const word of ngrams(chunk.text, 1)) corpusWords.add(word);
  const overlapping = questions
    .filter((q) => q.expect_abstention && [...ngrams(q.question, 1)].filter((w) => corpusWords.has(w)).length >= 6)
    .map((q) => q.question_id);
  gate.check("at least one abstention question shares vocabulary with the corpus", overlapping.length > 0,
    `hard cases: ${JSON.stringify(overlapping)}`);

  return gate.finish("M7");
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
