/**
 * MCP server exposing the research corpus to compatible hosts, over stdio or Streamable HTTP.
 *
 * In:  MCP tool calls: retrieve_evidence, analyze_corpus, explore_knowledge_graph, plus the
 *      sciagent://corpus/summary resource. Out: the engine's structured dicts, fetched via
 *      the Python bridge. In stdio mode stdout is the wire protocol; diagnostics go to stderr.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import type { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

import type { PythonBridge } from "../bridge/client.js";
import { isToolError, type JsonObject } from "../bridge/types.js";

const INSTRUCTIONS =
  "Search the local scientific-paper corpus before answering factual questions. " +
  "Use analyze_corpus for aggregate corpus statistics and explore_knowledge_graph " +
  "for paper, topic, chunk, and figure relationships.";

function toolResult(result: unknown): CallToolResult {
  const structured = (result ?? {}) as JsonObject;
  return {
    content: [{ type: "text", text: JSON.stringify(structured, null, 2) }],
    structuredContent: structured,
    isError: isToolError(result),
  };
}

/** Drop undefined optionals so the engine applies its own config defaults. */
function defined(params: Record<string, unknown>): JsonObject {
  return Object.fromEntries(Object.entries(params).filter(([, value]) => value !== undefined));
}

export function createMcpServer(bridge: PythonBridge): McpServer {
  const server = new McpServer({ name: "research-agent", version: "0.1.0" }, { instructions: INSTRUCTIONS });

  server.registerTool(
    "retrieve_evidence",
    {
      description: "Retrieve cited paper passages, figures, and tables relevant to a query.",
      inputSchema: {
        query: z.string(),
        k: z.number().int().positive().optional(),
        topic_filter: z.string().optional(),
        chunk_types: z.array(z.string()).optional(),
      },
    },
    // MCP calls are independent requests, unlike calls inside one agent conversation, so
    // the retriever's dedup state is reset for each (reset: true).
    async (args) => toolResult(await bridge.call("retrieve_evidence", defined({ ...args, reset: true }))),
  );

  server.registerTool(
    "analyze_corpus",
    {
      description: "Compute reproducible corpus statistics, timelines, clusters, or topic comparisons.",
      inputSchema: {
        operation: z.string(),
        topic_filter: z.string().optional(),
        params: z.record(z.string(), z.unknown()).optional(),
      },
    },
    async (args) => toolResult(await bridge.call("analyze_corpus", defined(args))),
  );

  server.registerTool(
    "explore_knowledge_graph",
    {
      description: "Traverse Neo4j relationships around a paper id, chunk id, topic, or figure id.",
      inputSchema: {
        entity_id: z.string(),
        depth: z.number().int().positive().default(1),
      },
    },
    async (args) => toolResult(await bridge.call("explore_graph", defined(args))),
  );

  server.registerResource(
    "corpus_summary",
    "sciagent://corpus/summary",
    {
      description: "Current corpus counts, without loading the embedding or reranking models.",
      mimeType: "application/json",
    },
    async (uri) => ({
      contents: [{
        uri: uri.href,
        mimeType: "application/json",
        text: JSON.stringify(await bridge.call("analyze_corpus", { operation: "stats" })),
      }],
    }),
  );

  return server;
}

export async function serveStdio(bridge: PythonBridge): Promise<void> {
  const server = createMcpServer(bridge);
  await server.connect(new StdioServerTransport());
  process.stderr.write("research-agent MCP server on stdio\n");
}

/** Stateless Streamable HTTP: a fresh server and transport per request, sharing one bridge. */
export async function serveHttp(bridge: PythonBridge, host: string, port: number): Promise<void> {
  const app = createMcpExpressApp({ host });

  app.post("/mcp", async (req, res) => {
    const server = createMcpServer(bridge);
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    res.on("close", () => {
      void transport.close();
      void server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  });

  const methodNotAllowed = (_req: unknown, res: { status: (code: number) => { json: (body: unknown) => void } }) =>
    res.status(405).json({ jsonrpc: "2.0", error: { code: -32000, message: "Method not allowed." }, id: null });
  app.get("/mcp", methodNotAllowed);
  app.delete("/mcp", methodNotAllowed);

  app.listen(port, host, () => {
    process.stderr.write(`research-agent MCP server on http://${host}:${port}/mcp\n`);
  });
}
