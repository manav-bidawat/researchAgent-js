/**
 * Client for the long-lived Python bridge worker (bridge/worker.py).
 *
 * In:  op names and params; optionally an onEvent callback for ops that stream progress.
 * Out: a Promise per call resolving to the op's result dict. Bridge-level failures (worker
 *      died, unknown op, op crashed) reject with BridgeError; tool-level failures resolve
 *      as {"error", "detail"} dicts, as the engine returns them.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { join } from "node:path";

import { REPO_ROOT } from "../paths.js";
import type { JsonObject, ReadyFrame, ResponseFrame } from "./types.js";

export class BridgeError extends Error {
  constructor(readonly code: string, readonly detail: string) {
    super(`${code}: ${detail}`);
    this.name = "BridgeError";
  }
}

export interface BridgeOptions {
  /** Executable to run. Default: $SCIAGENT_PYTHON, else python3. */
  command?: string;
  /** Arguments. Default: the worker script. */
  args?: string[];
  /** config.yaml to hand the engine; sets $SCIAGENT_CONFIG for the worker. */
  configPath?: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv;
  /** Forward the worker's stderr (engine logs, tracebacks). Default true. */
  forwardStderr?: boolean;
}

type EventHandler = (event: JsonObject) => void;

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onEvent?: EventHandler;
}

/** Lines of worker stderr kept to explain an unexpected exit. */
const STDERR_TAIL_LINES = 20;

export class PythonBridge {
  private child: ChildProcessWithoutNullStreams | null = null;
  private ready: Promise<ReadyFrame> | null = null;
  private readonly pending = new Map<number, Pending>();
  private nextId = 1;
  private stderrTail: string[] = [];
  private closing = false;

  constructor(private readonly options: BridgeOptions = {}) {}

  /** Start the worker if it is not running, and wait for its ready frame. */
  start(): Promise<ReadyFrame> {
    if (this.ready) return this.ready;
    this.closing = false;

    const command = this.options.command ?? process.env.SCIAGENT_PYTHON ?? "python3";
    const args = this.options.args ?? [join(REPO_ROOT, "bridge", "worker.py")];
    const env: NodeJS.ProcessEnv = { ...process.env, ...this.options.env, PYTHONUNBUFFERED: "1" };
    if (this.options.configPath) env.SCIAGENT_CONFIG = this.options.configPath;

    const child = spawn(command, args, { cwd: this.options.cwd ?? REPO_ROOT, env });
    this.child = child;

    this.ready = new Promise<ReadyFrame>((resolve, reject) => {
      let isReady = false;

      createInterface({ input: child.stdout }).on("line", (line) => {
        if (!line.trim()) return;
        let frame: ResponseFrame | ReadyFrame;
        try {
          frame = JSON.parse(line);
        } catch {
          process.stderr.write(`[bridge] non-protocol line on stdout: ${line.slice(0, 200)}\n`);
          return;
        }
        if ("ready" in frame) {
          isReady = true;
          resolve(frame);
          return;
        }
        this.dispatch(frame);
      });

      createInterface({ input: child.stderr }).on("line", (line) => {
        this.stderrTail.push(line);
        if (this.stderrTail.length > STDERR_TAIL_LINES) this.stderrTail.shift();
        if (this.options.forwardStderr !== false) process.stderr.write(`${line}\n`);
      });

      const fail = (code: string, detail: string) => {
        const error = new BridgeError(code, detail);
        if (!isReady) reject(error);
        for (const pending of this.pending.values()) pending.reject(error);
        this.pending.clear();
        this.child = null;
        this.ready = null;
      };

      child.on("error", (error) => fail("worker_spawn_failed", `${command}: ${error.message}`));
      child.on("exit", (code, signal) => {
        if (this.closing && this.pending.size === 0) {
          this.child = null;
          this.ready = null;
          return;
        }
        const tail = this.stderrTail.join("\n");
        fail("worker_exited", `python worker exited (code ${code}, signal ${signal})${tail ? `\n${tail}` : ""}`);
      });
    });
    return this.ready;
  }

  /** Invoke one bridge op. Calls are answered in order; the worker runs one at a time. */
  async call<T = unknown>(op: string, params: JsonObject = {}, onEvent?: EventHandler): Promise<T> {
    await this.start();
    const child = this.child;
    if (!child) throw new BridgeError("worker_not_running", "the python worker is not running");

    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, onEvent });
      child.stdin.write(`${JSON.stringify({ id, op, params })}\n`);
    });
  }

  /** Stop the worker. Pending calls reject; a later call starts a fresh worker. */
  async close(): Promise<void> {
    const child = this.child;
    if (!child) return;
    this.closing = true;
    await new Promise<void>((resolve) => {
      child.once("exit", () => resolve());
      child.stdin.end();
      setTimeout(() => child.kill(), 2_000).unref();
    });
  }

  private dispatch(frame: ResponseFrame): void {
    if (frame.id === null || frame.id === undefined) {
      process.stderr.write(`[bridge] ${frame.error}: ${frame.detail}\n`);
      return;
    }
    const pending = this.pending.get(frame.id);
    if (!pending) return;

    if (frame.event !== undefined) {
      try {
        pending.onEvent?.(frame.event);
      } catch {
        // An observer is for watching a call; a broken one must not end it.
      }
      return;
    }
    this.pending.delete(frame.id);
    if (frame.error !== undefined) pending.reject(new BridgeError(frame.error, frame.detail ?? ""));
    else pending.resolve(frame.result);
  }
}
