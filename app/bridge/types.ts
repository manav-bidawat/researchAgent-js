/**
 * Wire types for the Python bridge: request/response frames and the op results the
 * TypeScript layer reads fields from.
 *
 * In:  nothing at runtime; shapes mirror bridge/ops/*.py and docs/TOOLS.md.
 * Out: types only. Results stay open-ended (`[key: string]: unknown`) because the engine
 *      owns their shape and TS reads only what it prints or serves.
 */

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type JsonObject = { [key: string]: unknown };

/** The engine's convention: tools never raise, failures come back as a dict. */
export interface ToolError {
  error: string;
  detail?: string;
  [key: string]: unknown;
}

export interface ReadyFrame {
  ready: true;
  ops: string[];
}

export interface ResponseFrame {
  id: number | null;
  event?: JsonObject;
  result?: unknown;
  error?: string;
  detail?: string;
}

export interface EnginePaths {
  manifest: string;
  data_dir: string;
  chunks: string;
  traces: string;
  arxiv_cooldown: string;
  trace_text_chars: number;
  progress_text_chars: number;
}

export interface TraceRecord {
  iteration: number;
  tool_name: string;
  args?: JsonObject;
  chunk_ids_returned: string[];
  error?: string | null;
  latency_ms: number;
  [key: string]: unknown;
}

export interface AskResult {
  answer: string;
  question?: string;
  iterations: number;
  tool_calls: number;
  stopped_because: string;
  context_tokens: number;
  run_id: string;
  trace: string;
  trace_records: TraceRecord[];
  [key: string]: unknown;
}

export interface AskFailure extends ToolError {
  partial?: { answer?: string; iterations?: number; run_id?: string; trace?: string } | null;
}

/** One AgentLoop event (src/agent/loop.py `_emit`), with tool results summarised. */
export interface LoopEvent {
  kind: "thinking" | "tool_start" | "tool_end" | "skipped" | "image" | "answering" | "cap" | string;
  iteration?: number;
  tool_name?: string;
  args?: JsonObject;
  result?: JsonObject;
  latency_ms?: number;
  reason?: string;
  image_path?: string;
  [key: string]: unknown;
}

export interface IndexResult {
  collected: JsonObject;
  ingested: JsonObject;
  indexed: JsonObject;
  graph: JsonObject | null;
  rate_limited: boolean;
}

export interface RetrieveBatchRow {
  question_id: string;
  chunk_ids: string[];
  paper_ids: string[];
  scores: (number | null)[];
  sufficient_evidence: boolean;
  error: string | null;
  latency_ms: number;
}

export function isToolError(value: unknown): value is ToolError {
  return typeof value === "object" && value !== null && "error" in value
    && Boolean((value as ToolError).error);
}
