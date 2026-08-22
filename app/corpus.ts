/**
 * The indexed papers and the topics they were collected under, read from the manifest.
 *
 * In:  the manifest path (from the bridge's `paths` op).
 * Out: a sorted paper list and per-topic counts for the web corpus view. Read fresh on
 *      every call, so indexing from the CLI while the server is up shows on next reload.
 */

import { existsSync, readFileSync } from "node:fs";
import { basename } from "node:path";

export interface PaperSummary {
  paper_id: string;
  arxiv_id: string | null;
  title: string;
  authors: string[];
  year: number | null;
  categories: string[];
  topic_tags: string[];
  n_chunks: number;
  n_figures: number;
  pdf_name: string;
  pdf_url: string;
  parse_status: string | null;
}

export interface TopicSummary {
  tag: string;
  label: string;
  arxiv_query: string;
  count: number;
}

export interface CorpusView {
  papers: PaperSummary[];
  topics: TopicSummary[];
  embedding_model?: string;
  updated_at?: string;
  error?: string;
}

type RawRecord = Record<string, unknown>;

export function readCorpus(manifestPath: string): CorpusView {
  if (!existsSync(manifestPath)) {
    return { papers: [], topics: [], error: 'no index yet — run: npm run cli -- index "<topic>"' };
  }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as RawRecord;
  const rawTopics = (manifest.topics ?? {}) as Record<string, RawRecord | null>;
  const rawPapers = (manifest.papers ?? {}) as Record<string, RawRecord>;

  const papers: PaperSummary[] = Object.values(rawPapers).map((record) => {
    const pdfPath = String(record.pdf_path ?? "");
    return {
      paper_id: String(record.paper_id),
      arxiv_id: (record.arxiv_id as string) ?? null,
      title: String(record.title || record.paper_id),
      authors: (record.authors as string[]) ?? [],
      year: (record.year as number) ?? null,
      categories: (record.categories as string[]) ?? [],
      topic_tags: (record.topic_tags as string[]) ?? [],
      n_chunks: Number(record.n_chunks ?? 0),
      n_figures: Number(record.n_figures ?? 0),
      pdf_name: pdfPath ? basename(pdfPath) : "",
      pdf_url: String(record.pdf_url ?? ""),
      parse_status: (record.parse_status as string) ?? null,
    };
  });
  papers.sort((a, b) => a.title.toLowerCase().localeCompare(b.title.toLowerCase()));

  const topics: TopicSummary[] = Object.entries(rawTopics).map(([tag, body]) => ({
    tag,
    // The topic as it was typed, which reads better than the slug it became.
    label: String(body?.query || tag.replaceAll("_", " ")),
    arxiv_query: String(body?.arxiv_query ?? ""),
    count: papers.filter((paper) => paper.topic_tags.includes(tag)).length,
  }));
  topics.sort((a, b) => b.count - a.count);

  return {
    papers,
    topics,
    embedding_model: String(manifest.embedding_model ?? ""),
    updated_at: String(manifest.updated_at ?? ""),
  };
}
