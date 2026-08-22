import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import type { PythonBridge } from "../app/bridge/client.js";
import { createMcpServer } from "../app/mcp/server.js";
import { stubBridge } from "./helpers.js";

let bridge: PythonBridge;
let client: Client;

beforeAll(async () => {
  bridge = stubBridge();
  const server = createMcpServer(bridge);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  client = new Client({ name: "test-client", version: "0.0.0" });
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
});

afterAll(async () => {
  await client.close();
  await bridge.close();
});

type Structured = Record<string, unknown> & { received?: Record<string, unknown> };

describe("MCP server", () => {
  it("advertises exactly the three corpus tools", async () => {
    const { tools } = await client.listTools();
    expect(new Set(tools.map((tool) => tool.name))).toEqual(
      new Set(["retrieve_evidence", "analyze_corpus", "explore_knowledge_graph"]),
    );
  });

  it("carries the server instructions", () => {
    expect(client.getInstructions()).toMatch(/analyze_corpus/);
  });

  it("forwards retrieve_evidence with reset: true and drops unset optionals", async () => {
    const result = await client.callTool({ name: "retrieve_evidence", arguments: { query: "attention", k: 3 } });
    expect(result.isError).toBe(false);
    const structured = result.structuredContent as Structured;
    expect(structured.received).toEqual({ query: "attention", k: 3, reset: true });
    const [content] = result.content as { type: string; text: string }[];
    expect(content!.type).toBe("text");
    expect(JSON.parse(content!.text)).toEqual(structured);
  });

  it("marks a tool-error dict as isError", async () => {
    const result = await client.callTool({ name: "analyze_corpus", arguments: { operation: "bogus" } });
    expect(result.isError).toBe(true);
    expect((result.structuredContent as Structured).error).toBe("unknown_operation");
  });

  it("maps explore_knowledge_graph to the explore_graph op with depth defaulted", async () => {
    const result = await client.callTool({ name: "explore_knowledge_graph", arguments: { entity_id: "p1" } });
    expect(result.isError).toBe(false);
    expect(result.structuredContent).toMatchObject({ entity_id: "p1", depth: 1 });
  });

  it("reads the corpus summary resource", async () => {
    const { resources } = await client.listResources();
    expect(resources.map((r) => r.uri)).toContain("sciagent://corpus/summary");
    const read = await client.readResource({ uri: "sciagent://corpus/summary" });
    const [content] = read.contents as { uri: string; mimeType: string; text: string }[];
    expect(content!.uri).toBe("sciagent://corpus/summary");
    expect(content!.mimeType).toBe("application/json");
    expect(JSON.parse(content!.text)).toMatchObject({ operation: "stats", n_papers: 2 });
  });
});
