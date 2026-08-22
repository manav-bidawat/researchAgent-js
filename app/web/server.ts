/**
 * Express demo server: one page that asks a question and shows the corpus.
 *
 * In:  HTTP requests. Out: the static page, /api/papers from the manifest, and /api/ask as
 *      a Server-Sent Events stream of the agent's progress followed by its answer.
 *      Every engine call goes through the Python bridge; nothing in src/ knows this exists.
 */

import express, { type Express, type Response } from "express";
import { join } from "node:path";

import { BridgeError, type PythonBridge } from "../bridge/client.js";
import { isToolError, type AskFailure, type AskResult, type EnginePaths } from "../bridge/types.js";
import { readCorpus } from "../corpus.js";
import { REPO_ROOT } from "../paths.js";

const STATIC_DIR = join(REPO_ROOT, "web", "static");

function sse(res: Response, event: string, payload: unknown): boolean {
  if (res.writableEnded || res.destroyed) return false;
  return res.write(`event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`);
}

export function createApp(bridge: PythonBridge, paths: EnginePaths): Express {
  const app = express();
  app.disable("x-powered-by");

  // An explicit map, not a static directory: a request can only ever reach these files.
  app.get("/", (_req, res) => res.sendFile(join(STATIC_DIR, "index.html")));
  app.get("/static/style.css", (_req, res) => res.sendFile(join(STATIC_DIR, "style.css")));
  app.get("/static/app.js", (_req, res) => res.sendFile(join(STATIC_DIR, "app.js")));

  app.get("/api/papers", (_req, res) => {
    try {
      res.json(readCorpus(paths.manifest));
    } catch (error) {
      res.status(500).json({ papers: [], topics: [], error: "manifest_unreadable", detail: String(error) });
    }
  });

  app.get("/api/ask", async (req, res) => {
    const question = String(req.query.q ?? "").trim();
    res.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "close",
    });

    if (!question) {
      sse(res, "failed", { error: "empty_question", detail: "type a question first" });
      sse(res, "done", {});
      res.end();
      return;
    }

    let alive = true;
    res.on("close", () => {
      alive = false;
    });

    let result: AskResult | AskFailure;
    try {
      result = await bridge.call<AskResult | AskFailure>("ask", { question }, (event) => {
        if (alive) sse(res, "step", event);
      });
    } catch (error) {
      // The loop is meant not to raise; a page must not 500 when the worker does.
      const detail = error instanceof BridgeError ? error.message : String(error);
      result = { error: "loop_crashed", detail };
    }
    if (!alive) return;

    if (isToolError(result)) {
      const failure = result as AskFailure;
      sse(res, "failed", {
        error: failure.error,
        detail: failure.detail ?? "",
        answer: failure.partial?.answer ?? "",
      });
    } else {
      const answer = result as AskResult;
      sse(res, "answer", {
        answer: answer.answer,
        iterations: answer.iterations,
        tool_calls: answer.tool_calls,
        context_tokens: answer.context_tokens,
        stopped_because: answer.stopped_because,
        run_id: answer.run_id,
      });
    }
    sse(res, "done", {});
    res.end();
  });

  app.use((_req, res) => {
    res.status(404).type("text/plain").send("no route here");
  });
  return app;
}

/** Warm the engine, then answer requests until interrupted. */
export async function serve(bridge: PythonBridge, host: string, port: number): Promise<void> {
  console.log("loading models and the index ...");
  const paths = await bridge.call<EnginePaths>("paths");
  // Load the models before the first question rather than inside it.
  await bridge.call("warm");
  const corpus = readCorpus(paths.manifest);
  console.log(`ready — ${corpus.papers.length} paper(s) indexed`);

  const server = createApp(bridge, paths).listen(port, host, () => {
    console.log(`open http://${host}:${port}`);
  });
  const stop = () => {
    console.log("\nstopped");
    server.close();
    void bridge.close().then(() => process.exit(0));
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
}
