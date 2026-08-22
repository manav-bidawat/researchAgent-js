/**
 * Live console progress for one agent run: a port of src/agent/progress.py.
 *
 * In:  LoopEvent dicts streamed over the bridge — kind, iteration, tool name, args, result.
 * Out: one formatted line per event on a stream (stderr by default), so stdout stays the
 *      answer alone.
 */

import { basename } from "node:path";

import type { JsonObject, LoopEvent } from "../bridge/types.js";

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function truncate(text: string, width: number): string {
  return text.length <= width ? text : `${text.slice(0, Math.max(0, width - 1))}…`;
}

/** The call's arguments as the model supplied them, short enough for one line. */
export function formatArgs(args: JsonObject | undefined, width: number): string {
  if (!args || typeof args !== "object" || Object.keys(args).length === 0) return "";
  const rendered = Object.entries(args)
    .map(([key, value]) => `${key}=${typeof value === "string" ? `'${value}'` : JSON.stringify(value)}`)
    .join(", ");
  return truncate(rendered, width);
}

/** One phrase saying what a tool came back with; each tool has its own count field. */
export function formatResult(result: JsonObject | undefined, width: number): string {
  if (!result || typeof result !== "object") return typeof result;
  if (result.error) return truncate(`error: ${String(result.error)}`, width);
  if (Array.isArray(result.chunks)) {
    let note = plural(result.chunks.length, "chunk");
    if (result.sufficient_evidence === false) note += ", insufficient";
    return note;
  }
  if (Array.isArray(result.papers_added)) {
    return `${plural(result.papers_added.length, "paper")} added, ${Number(result.chunks_added ?? 0)} chunks`;
  }
  if (Array.isArray(result.claims)) {
    return `${plural(result.claims.length, "claim")}, grounded ${String(result.grounded_ratio)}`;
  }
  if ("found" in result) {
    const scored = plural(Number(result.n_pairs_scored ?? 0), "pair");
    return `${scored} scored, ${result.found ? "conflict found" : "no conflict"}`;
  }
  if (result.operation) return truncate(String(result.summary ?? result.operation), width);
  if (result.image_path) return basename(String(result.image_path));
  return truncate(Object.keys(result).sort().join(", "), width);
}

/**
 * Prints one line per loop event. Stateful: `tool_start` opens a line without a newline so
 * a slow tool shows what it is doing, and the matching `tool_end` finishes it.
 */
export class ConsoleReporter {
  private lineOpen = false;

  constructor(
    private readonly width: number,
    private readonly stream: NodeJS.WritableStream = process.stderr,
  ) {}

  private write(text: string, end = "\n"): void {
    this.stream.write(text + end);
  }

  private closeLine(): void {
    if (this.lineOpen) {
      this.write("");
      this.lineOpen = false;
    }
  }

  handle = (event: LoopEvent): void => {
    const tag = `[${(event.iteration ?? 0) + 1}]`;
    switch (event.kind) {
      case "thinking":
        this.closeLine();
        this.write(`${tag} thinking...`);
        break;
      case "tool_start":
        this.closeLine();
        this.write(`${tag} → ${event.tool_name}(${formatArgs(event.args, this.width)}) ... `, "");
        this.lineOpen = true;
        break;
      case "tool_end": {
        const note = formatResult(event.result, this.width);
        this.write(this.lineOpen
          ? `${note}, ${event.latency_ms ?? 0}ms`
          : `${tag} → ${event.tool_name} ... ${note}`);
        this.lineOpen = false;
        break;
      }
      case "skipped":
        this.closeLine();
        this.write(`${tag} → ${event.tool_name || "call"} skipped: ${event.reason}`);
        break;
      case "image":
        this.closeLine();
        this.write(`${tag} ↳ attaching ${basename(String(event.image_path))}`);
        break;
      case "answering":
        this.closeLine();
        this.write(`${tag} answering`);
        break;
      case "cap":
        this.closeLine();
        this.write(`${tag} iteration cap reached — forcing an answer from what is held`);
        break;
      default:
        break;
    }
  };
}
