import { afterEach, describe, expect, it } from "vitest";

import { BridgeError, type PythonBridge } from "../app/bridge/client.js";
import type { JsonObject } from "../app/bridge/types.js";
import { stubBridge } from "./helpers.js";

let bridge: PythonBridge;

afterEach(async () => {
  await bridge?.close();
});

describe("PythonBridge", () => {
  it("waits for the ready frame and lists the worker's ops", async () => {
    bridge = stubBridge();
    const ready = await bridge.start();
    expect(ready.ready).toBe(true);
    expect(ready.ops).toContain("ask");
    expect(ready.ops).toContain("embed_passages");
  });

  it("resolves an op's result", async () => {
    bridge = stubBridge();
    await expect(bridge.call("ping")).resolves.toEqual({ ok: true, python: "stub" });
  });

  it("answers concurrent calls by id", async () => {
    bridge = stubBridge();
    const [a, b] = await Promise.all([
      bridge.call<JsonObject>("explore_graph", { entity_id: "a", depth: 1 }),
      bridge.call<JsonObject>("explore_graph", { entity_id: "b", depth: 2 }),
    ]);
    expect(a.entity_id).toBe("a");
    expect(b.entity_id).toBe("b");
  });

  it("streams events to onEvent in order, all before the result resolves", async () => {
    bridge = stubBridge();
    const seen: string[] = [];
    const result = await bridge.call<JsonObject>("events_then_result", { n: 5 }, (event) => {
      seen.push(`tick${String(event.i)}`);
    });
    seen.push("result");
    expect(result).toEqual({ done: true });
    expect(seen).toEqual(["tick0", "tick1", "tick2", "tick3", "tick4", "result"]);
  });

  it("delivers the agent's loop events in kind order", async () => {
    bridge = stubBridge();
    const kinds: unknown[] = [];
    await bridge.call("ask", { question: "what is attention?" }, (event) => kinds.push(event.kind));
    expect(kinds).toEqual(["thinking", "tool_start", "tool_end", "answering"]);
  });

  it("rejects with BridgeError on an error frame", async () => {
    bridge = stubBridge();
    const error = await bridge.call("no_such_op").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(BridgeError);
    expect((error as BridgeError).code).toBe("unknown_op");
    expect((error as BridgeError).detail).toContain("no_such_op");
  });

  it("reports an op crash as op_crashed, and the worker stays usable", async () => {
    bridge = stubBridge();
    const error = await bridge.call("ask", { question: "bridge-error" }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(BridgeError);
    expect((error as BridgeError).code).toBe("op_crashed");
    await expect(bridge.call("ping")).resolves.toMatchObject({ ok: true });
  });

  it("resolves tool-level failures as dicts rather than rejecting", async () => {
    bridge = stubBridge();
    const result = await bridge.call<JsonObject>("analyze_corpus", { operation: "bogus" });
    expect(result.error).toBe("unknown_operation");
  });

  it("rejects every pending call with worker_exited when the worker dies", async () => {
    bridge = stubBridge();
    await bridge.start();
    const dying = bridge.call("exit", { code: 3 });
    const queued = bridge.call("ping");
    const [a, b] = await Promise.allSettled([dying, queued]);
    for (const outcome of [a, b]) {
      expect(outcome.status).toBe("rejected");
      const reason = (outcome as PromiseRejectedResult).reason as BridgeError;
      expect(reason).toBeInstanceOf(BridgeError);
      expect(reason.code).toBe("worker_exited");
      expect(reason.detail).toContain("code 3");
      expect(reason.detail).toContain("stub worker exiting on request");
    }
  });

  it("starts a fresh worker on the call after a crash", async () => {
    bridge = stubBridge();
    await bridge.call("exit").catch(() => undefined);
    await expect(bridge.call("ping")).resolves.toMatchObject({ ok: true });
  });

  it("does not let a throwing onEvent callback break the call", async () => {
    bridge = stubBridge();
    let calls = 0;
    const result = await bridge.call("events_then_result", { n: 3 }, () => {
      calls += 1;
      throw new Error("observer bug");
    });
    expect(result).toEqual({ done: true });
    expect(calls).toBe(3);
  });

  it("reports a missing executable as worker_spawn_failed", async () => {
    const { PythonBridge } = await import("../app/bridge/client.js");
    bridge = new PythonBridge({ command: "/nonexistent/python-for-tests", args: [], forwardStderr: false });
    const error = await bridge.call("ping").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(BridgeError);
    expect((error as BridgeError).code).toBe("worker_spawn_failed");
  });

  it("close() stops the worker, is idempotent, and a later call restarts it", async () => {
    bridge = stubBridge();
    await bridge.call("ping");
    await bridge.close();
    await bridge.close();
    await expect(bridge.call("ping")).resolves.toMatchObject({ ok: true });
  });

  it("close() before any call is a no-op", async () => {
    bridge = stubBridge();
    await expect(bridge.close()).resolves.toBeUndefined();
  });
});
