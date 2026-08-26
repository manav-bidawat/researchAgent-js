/**
 * Shared scaffolding for the milestone gates in scripts/: pass/fail bookkeeping, typed
 * reads of the engine's data files, and the gate_config op's shape.
 *
 * In:  check labels and outcomes; file paths from the bridge's `paths` / `gate_config` ops.
 * Out: printed PASS/FAIL lines, parsed manifest and chunk records, and a final exit code.
 */

import { existsSync } from "node:fs";

import { readJson, readJsonl } from "../eval/lib.js";

export { fixed, isEntryPoint, parseJsonText, readJson, readJsonl, signed } from "../eval/lib.js";

/** PASS/FAIL lines and the list of failures behind a gate's exit code. */
export class Gate {
  readonly failures: string[] = [];
  private readonly started = performance.now();

  /** `timed` prefixes every line with seconds since start, as the network-bound gates do. */
  constructor(private readonly timed = true) {}

  elapsed(): string {
    return `${((performance.now() - this.started) / 1000).toFixed(1).padStart(6)}s`;
  }

  private prefix(): string {
    return this.timed ? `[${this.elapsed()}] ` : "";
  }

  /** Announce a slow step before it starts, so a stall is visible. */
  step(message: string): void {
    console.log(`${this.prefix()}.... ${message}`);
  }

  check(label: string, ok: boolean, detail = ""): void {
    console.log(`${this.prefix()}[${ok ? "PASS" : "FAIL"}] ${label}${detail ? ` — ${detail}` : ""}`);
    if (!ok) this.failures.push(label);
  }

  /** Prints the verdict line and returns the exit code. */
  finish(name: string): number {
    console.log(`\n${this.failures.length ? `${name} GATE FAILED: ${this.failures.join(", ")}` : `${name} GATE PASSED`}`);
    return this.failures.length ? 1 : 0;
  }
}

export interface GateConfig {
  source_path: string;
  llm: { agent_model: string; vision_model: string; utility_model: string };
  compute_device: string;
  embedding: { model: string; dim: number; batch_size: number };
  chunking: { max_tokens: number };
  retrieval: {
    relevance_threshold: number;
    bi_encoder_relevance_threshold: number | null;
    k_retrieve: number;
    max_chunk_chars: number;
    reranker_model: string;
  };
  agent: { max_iterations: number; trace_text_chars: number };
  nli: { model: string; max_pairs: number };
  paths: {
    papers: string;
    index: string;
    embeddings: string;
    faiss_index: string;
    descriptions: string;
    directories: string[];
  };
}

export interface PaperRecord {
  paper_id: string;
  title: string;
  abstract: string;
  year: number;
  pdf_path: string;
  topic_tags: string[];
  n_chunks: number;
  n_figures: number;
  parse_status: string;
  [key: string]: unknown;
}

export interface ManifestData {
  papers: Record<string, PaperRecord>;
  topics: Record<string, { paper_ids?: string[]; [key: string]: unknown }>;
  figures: Record<string, Record<string, unknown>>;
  faiss_id_map: string[];
  embedding_model?: string;
  embedding_dim?: number;
  [key: string]: unknown;
}

/** manifest.json as Manifest.load(strict_model_check=False) sees it; missing reads as empty. */
export function loadManifest(path: string): ManifestData {
  const raw = existsSync(path) ? readJson<Partial<ManifestData>>(path) : {};
  return {
    ...raw,
    papers: raw.papers ?? {},
    topics: raw.topics ?? {},
    figures: raw.figures ?? {},
    faiss_id_map: raw.faiss_id_map ?? [],
  };
}

/** Manifest.papers_for_topic. */
export function papersForTopic(manifest: ManifestData, tag: string): string[] {
  return [...(manifest.topics[tag]?.paper_ids ?? [])];
}

export interface ChunkRecord {
  chunk_id: string;
  paper_id: string;
  paper_title?: string;
  chunk_type: string;
  text: string;
  page: number;
  position: number;
  n_tokens: number;
  section?: string | null;
  caption?: string | null;
  image_path?: string | null;
  topic_tags: string[];
  content_hash?: string;
  [key: string]: unknown;
}

export function loadChunks(path: string): ChunkRecord[] {
  return readJsonl<ChunkRecord>(path);
}

/** ChunkStore.for_paper: one paper's chunks sorted by position. */
export function chunksForPaper(chunks: ChunkRecord[], paperId: string): ChunkRecord[] {
  return chunks.filter((c) => c.paper_id === paperId).sort((a, b) => (a.position ?? 0) - (b.position ?? 0));
}

/** Whitespace collapsed to single spaces, as `" ".join(text.split())`. */
export const squash = (text: string): string => String(text ?? "").split(/\s+/).filter(Boolean).join(" ");

/** Code-point order, matching Python's sorted() on ASCII ids. */
export const byString = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
