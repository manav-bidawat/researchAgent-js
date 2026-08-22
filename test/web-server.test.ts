import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import request from "supertest";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { PythonBridge } from "../app/bridge/client.js";
import type { EnginePaths } from "../app/bridge/types.js";
import { createApp } from "../app/web/server.js";
import { stubBridge, writeManifest } from "./helpers.js";

interface SseFrame {
  event: string;
  data: Record<string, unknown>;
}

/** Parse a complete text/event-stream body into (event, JSON data) frames. */
function parseSse(body: string): SseFrame[] {
  return body.split("\n\n").filter((block) => block.trim()).map((block) => {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    return { event, data: JSON.parse(data) as Record<string, unknown> };
  });
}

/** supertest does not buffer text/event-stream by default; collect it as a string. */
function sseText(res: NodeJS.ReadableStream, callback: (err: Error | null, body: string) => void): void {
  let body = "";
  res.setEncoding("utf-8");
  res.on("data", (chunk: string) => { body += chunk; });
  res.on("end", () => callback(null, body));
}

const dir = mkdtempSync(join(tmpdir(), "sciagent-web-"));
let bridge: PythonBridge;
let paths: EnginePaths;

beforeAll(async () => {
  bridge = stubBridge();
  paths = { ...(await bridge.call<EnginePaths>("paths")), manifest: join(dir, "manifest.json") };
  writeManifest(paths.manifest);
});

afterAll(async () => {
  await bridge.close();
  rmSync(dir, { recursive: true, force: true });
});

async function ask(q?: string): Promise<{ status: number; type: string; frames: SseFrame[] }> {
  const req = request(createApp(bridge, paths)).get("/api/ask");
  if (q !== undefined) void req.query({ q });
  const res = await req.buffer(true).parse(sseText as never);
  return { status: res.status, type: String(res.headers["content-type"]), frames: parseSse(res.body as string) };
}

describe("web server", () => {
  it("serves the page at /", async () => {
    const res = await request(createApp(bridge, paths)).get("/");
    expect(res.status).toBe(200);
    expect(res.headers["content-type"]).toMatch(/text\/html/);
    expect(res.text).toMatch(/<html/i);
    expect(res.headers["x-powered-by"]).toBeUndefined();
  });

  it("serves the page's script and stylesheet", async () => {
    const app = createApp(bridge, paths);
    const js = await request(app).get("/static/app.js");
    expect(js.status).toBe(200);
    expect(js.headers["content-type"]).toMatch(/javascript/);
    const css = await request(app).get("/static/style.css");
    expect(css.status).toBe(200);
    expect(css.headers["content-type"]).toMatch(/css/);
  });

  it("serves the corpus at /api/papers", async () => {
    const res = await request(createApp(bridge, paths)).get("/api/papers");
    expect(res.status).toBe(200);
    expect(res.body.papers.map((p: { paper_id: string }) => p.paper_id)).toEqual(["p1", "p2", "p3"]);
    expect(res.body.topics).toHaveLength(2);
  });

  it("reports a missing manifest as an error field, not a 500", async () => {
    const res = await request(createApp(bridge, { ...paths, manifest: join(dir, "nope.json") })).get("/api/papers");
    expect(res.status).toBe(200);
    expect(res.body.error).toMatch(/no index yet/);
  });

  it("answers an unknown route with a plain-text 404", async () => {
    const app = createApp(bridge, paths);
    for (const path of ["/nope", "/static/../config.yaml", "/static/index.html"]) {
      const res = await request(app).get(path);
      expect(res.status).toBe(404);
      expect(res.headers["content-type"]).toMatch(/text\/plain/);
      expect(res.text).toBe("no route here");
    }
  });

  it("answers an empty question with failed then done", async () => {
    for (const q of [undefined, "   "]) {
      const { status, type, frames } = await ask(q);
      expect(status).toBe(200);
      expect(type).toMatch(/text\/event-stream/);
      expect(frames.map((f) => f.event)).toEqual(["failed", "done"]);
      expect(frames[0]!.data.error).toBe("empty_question");
    }
  });

  it("streams step frames, then the answer, then done", async () => {
    const { frames } = await ask("what is attention?");
    expect(frames.map((f) => f.event)).toEqual(["step", "step", "step", "step", "answer", "done"]);
    expect(frames.slice(0, 4).map((f) => f.data.kind)).toEqual(["thinking", "tool_start", "tool_end", "answering"]);
    expect(frames[4]!.data).toEqual({
      answer: "Transformers use self-attention [p1__c0]. They scale well [p2__c3].",
      iterations: 2,
      tool_calls: 1,
      context_tokens: 1234,
      stopped_because: "answered",
      run_id: "run-stub-1",
    });
    expect(frames[5]!.data).toEqual({});
  });

  it("reports an ask failure as a failed frame carrying the partial answer", async () => {
    const { frames } = await ask("fail");
    expect(frames.map((f) => f.event)).toEqual(["failed", "done"]);
    expect(frames[0]!.data).toEqual({
      error: "llm_unavailable",
      detail: "the provider timed out",
      answer: "partial answer so far",
    });
  });

  it("turns a bridge crash into a loop_crashed failed frame", async () => {
    const { frames } = await ask("bridge-error");
    expect(frames.map((f) => f.event)).toEqual(["failed", "done"]);
    expect(frames[0]!.data.error).toBe("loop_crashed");
    expect(String(frames[0]!.data.detail)).toContain("op_crashed");
    expect(frames[0]!.data.answer).toBe("");
  });
});
