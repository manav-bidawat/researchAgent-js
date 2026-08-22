import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";

import { readCorpus } from "../app/corpus.js";
import { writeManifest } from "./helpers.js";

const dir = mkdtempSync(join(tmpdir(), "sciagent-corpus-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("readCorpus", () => {
  it("returns an error field when there is no manifest", () => {
    const view = readCorpus(join(dir, "missing.json"));
    expect(view.papers).toEqual([]);
    expect(view.topics).toEqual([]);
    expect(view.error).toMatch(/no index yet/);
  });

  it("sorts papers case-insensitively by title and counts papers per topic", () => {
    const path = join(dir, "manifest.json");
    writeManifest(path);
    const view = readCorpus(path);
    expect(view.error).toBeUndefined();
    // p3 has no title, so its paper_id stands in.
    expect(view.papers.map((p) => p.paper_id)).toEqual(["p1", "p2", "p3"]);
    expect(view.papers.map((p) => p.title)).toEqual(["Alpha paper", "beta paper", "p3"]);

    const p2 = view.papers.find((p) => p.paper_id === "p2")!;
    expect(p2.pdf_name).toBe("p2.pdf");
    expect(p2.n_chunks).toBe(7);
    const p1 = view.papers.find((p) => p.paper_id === "p1")!;
    expect(p1).toMatchObject({ arxiv_id: null, authors: [], year: null, pdf_name: "", n_figures: 0 });

    expect(view.topics).toEqual([
      { tag: "topic_a", label: "Topic A as typed", arxiv_query: "all:a", count: 2 },
      { tag: "topic_b", label: "topic b", arxiv_query: "", count: 2 },
    ]);
    expect(view.embedding_model).toBe("BAAI/bge-small-en-v1.5");
    expect(view.updated_at).toBe("2026-09-01T00:00:00Z");
  });
});
