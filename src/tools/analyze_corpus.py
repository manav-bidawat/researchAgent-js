"""
`analyze_corpus`: statistics over corpus metadata. No retrieval, no LLM call.

In:  an operation ('stats' | 'timeline' | 'cluster' | 'compare_topics'), optional
     topic_filter and params.
Out: {"operation", "result", "summary"} or an error dict. `summary` is templated from
     the numbers in `result`, never generated, so it cannot disagree with them.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config import CFG, Config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest, ManifestError
from common.storage import read_json, write_json_atomic

OPERATIONS = ("stats", "timeline", "cluster", "compare_topics")


def _error(code: str, detail: str, partial: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"error": code, "detail": detail, "partial": partial}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


class CorpusAnalyzer:
    """Pandas and scikit-learn over the manifest and chunk metadata.

    Deliberately has no access to an LLM. The `summary` field is produced by string
    templating over the computed numbers, one template per operation, which keeps the
    tool reproducible for the eval and makes it impossible for the prose to say something
    the numbers do not.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.config = config

    # ---- shared loading --------------------------------------------------------

    def _load(self, topic_filter: Optional[str]):
        try:
            manifest = Manifest.load(self.config, strict_model_check=False)
        except ManifestError as exc:
            return None, None, _error("manifest_unreadable", str(exc))

        papers = list(manifest.papers.values())
        if topic_filter:
            papers = [p for p in papers if topic_filter in (p.get("topic_tags") or [])]
        if not papers:
            detail = (
                f"no papers carry the topic tag {topic_filter!r}"
                if topic_filter else "the corpus is empty; fetch literature first"
            )
            return None, None, _error("empty_corpus", detail)
        return manifest, papers, None

    # ---- operations ------------------------------------------------------------

    def stats(self, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """Papers per topic, chunk and figure counts, date range."""
        manifest, papers, failure = self._load(topic_filter)
        if failure:
            return failure

        paper_ids = {p["paper_id"] for p in papers}
        chunks = [c for c in ChunkStore(config=self.config) if c["paper_id"] in paper_ids]
        types = Counter(c["chunk_type"] for c in chunks)
        per_topic = Counter(tag for p in papers for tag in (p.get("topic_tags") or []))
        years = sorted(int(p["year"]) for p in papers if p.get("year"))

        result = {
            "n_papers": len(papers),
            "n_chunks": len(chunks),
            "n_text_chunks": types.get("text", 0),
            "n_figure_chunks": types.get("figure", 0),
            "n_table_chunks": types.get("table", 0),
            "papers_per_topic": dict(sorted(per_topic.items())),
            "year_range": [years[0], years[-1]] if years else None,
            "parse_status": dict(Counter(p.get("parse_status", "unknown") for p in papers)),
            "mean_chunks_per_paper": round(len(chunks) / len(papers), 1) if papers else 0.0,
        }
        span = (
            f"published between {result['year_range'][0]} and {result['year_range'][1]}"
            if result["year_range"] else "with no recorded publication years"
        )
        topics = ", ".join(f"{tag} ({count})" for tag, count in result["papers_per_topic"].items())
        result_summary = (
            f"The corpus holds {_plural(result['n_papers'], 'paper')} {span}, "
            f"split into {_plural(result['n_chunks'], 'chunk')} "
            f"({result['n_text_chunks']} text, {result['n_figure_chunks']} figure, "
            f"{result['n_table_chunks']} table), averaging "
            f"{result['mean_chunks_per_paper']} chunks per paper. "
            f"Topic coverage: {topics or 'none recorded'}."
        )
        return {"operation": "stats", "result": result, "summary": result_summary}

    def timeline(self, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """Per-topic year histogram, median year, recency."""
        manifest, papers, failure = self._load(topic_filter)
        if failure:
            return failure

        by_year = Counter(int(p["year"]) for p in papers if p.get("year"))
        per_topic: Dict[str, Dict[str, Any]] = {}
        for paper in papers:
            for tag in paper.get("topic_tags") or []:
                entry = per_topic.setdefault(tag, {"years": []})
                if paper.get("year"):
                    entry["years"].append(int(paper["year"]))
        for tag, entry in per_topic.items():
            years = sorted(entry["years"])
            entry["histogram"] = dict(sorted(Counter(years).items()))
            entry["median_year"] = int(np.median(years)) if years else None
            entry["newest"] = years[-1] if years else None
            entry.pop("years")

        all_years = sorted(int(p["year"]) for p in papers if p.get("year"))
        result = {
            "histogram": dict(sorted(by_year.items())),
            "median_year": int(np.median(all_years)) if all_years else None,
            "newest_year": all_years[-1] if all_years else None,
            "oldest_year": all_years[0] if all_years else None,
            "per_topic": per_topic,
        }
        summary = (
            f"The {_plural(len(papers), 'paper')} span "
            f"{result['oldest_year']}–{result['newest_year']} with a median year of "
            f"{result['median_year']}. Per-year counts: "
            + ", ".join(f"{year}: {count}" for year, count in result["histogram"].items())
            + "."
        ) if all_years else "No publication years are recorded for these papers."
        return {"operation": "timeline", "result": result, "summary": summary}

    def compare_topics(self, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """Vocabulary overlap and centroid distance between topic groups."""
        manifest, papers, failure = self._load(None)
        if failure:
            return failure

        by_topic: Dict[str, List[str]] = {}
        for paper in papers:
            for tag in paper.get("topic_tags") or []:
                by_topic.setdefault(tag, []).append(paper["paper_id"])
        if len(by_topic) < 2:
            return _error(
                "too_few_topics",
                f"comparison needs at least two topics, the corpus has {len(by_topic)}",
            )

        chunks = list(ChunkStore(config=self.config))
        text_by_topic: Dict[str, List[str]] = {tag: [] for tag in by_topic}
        for chunk in chunks:
            for tag in chunk.get("topic_tags") or []:
                if tag in text_by_topic:
                    text_by_topic[tag].append(chunk["text"])

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            return _error("sklearn_missing", str(exc))

        tags = sorted(text_by_topic)
        documents = [" ".join(text_by_topic[tag]) for tag in tags]
        if not any(document.strip() for document in documents):
            return _error("no_chunks", "the papers have no chunks; run extraction first")

        vectoriser = TfidfVectorizer(stop_words="english", max_features=4000)
        matrix = vectoriser.fit_transform(documents)
        vocabulary = np.asarray(vectoriser.get_feature_names_out())

        pairs = []
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                left = set(vocabulary[matrix[i].toarray()[0] > 0])
                right = set(vocabulary[matrix[j].toarray()[0] > 0])
                overlap = len(left & right) / len(left | right) if (left | right) else 0.0
                a, b = matrix[i].toarray()[0], matrix[j].toarray()[0]
                denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
                similarity = float(a @ b / denominator) if denominator else 0.0
                pairs.append({
                    "topics": [tags[i], tags[j]],
                    "jaccard_vocabulary_overlap": round(overlap, 4),
                    "centroid_cosine": round(similarity, 4),
                    "distinctive_terms": {
                        tags[i]: vocabulary[np.argsort(a - b)[-6:][::-1]].tolist(),
                        tags[j]: vocabulary[np.argsort(b - a)[-6:][::-1]].tolist(),
                    },
                })

        result = {"topics": tags, "pairs": pairs,
                  "papers_per_topic": {tag: len(ids) for tag, ids in by_topic.items()}}
        lines = [
            f"{p['topics'][0]} vs {p['topics'][1]}: vocabulary overlap "
            f"{p['jaccard_vocabulary_overlap']:.2f}, centroid cosine {p['centroid_cosine']:.2f}"
            for p in pairs
        ]
        return {"operation": "compare_topics", "result": result,
                "summary": "Topic comparison — " + "; ".join(lines) + "."}

    def cluster(
        self, topic_filter: Optional[str] = None, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """KMeans over chunk embeddings, k by silhouette, validated against topic_tags.

        The validation is the point. Cluster labels alone are decorative; comparing the
        assignments against the topic tags we already know via adjusted Rand index and
        purity says whether the embeddings carry the semantic signal retrieval depends
        on. If the clusters fail to recover a known topic split, that is a real finding
        about the retrieval stack, and it gets reported rather than buried.
        """
        params = params or {}
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import adjusted_rand_score, silhouette_score
        except ImportError as exc:
            return _error("sklearn_missing", str(exc))

        try:
            manifest = Manifest.load(self.config, strict_model_check=False)
        except ManifestError as exc:
            return _error("manifest_unreadable", str(exc))

        cache_key = manifest.data.get("manifest_hash") or manifest.compute_hash()
        scope = topic_filter or "__all__"
        cache = read_json(self.config.paths.clusters, default={}) or {}
        cached = cache.get(cache_key, {}).get(scope)
        if cached and not params:
            return {**cached, "cached": True}

        chunk_ids = list(manifest.data.get("faiss_id_map") or [])
        if not chunk_ids or not self.config.paths.embeddings.is_file():
            return _error("no_index", "no embeddings on disk; build the index first")

        vectors = np.load(self.config.paths.embeddings).astype(np.float32)
        if vectors.shape[0] != len(chunk_ids):
            return _error(
                "index_inconsistent",
                f"embeddings.npy has {vectors.shape[0]} rows but faiss_id_map has "
                f"{len(chunk_ids)}",
            )

        by_id = {c["chunk_id"]: c for c in ChunkStore(config=self.config)}
        rows, labels, texts, kept_ids = [], [], [], []
        for position, chunk_id in enumerate(chunk_ids):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            tags = chunk.get("topic_tags") or []
            if topic_filter and topic_filter not in tags:
                continue
            rows.append(vectors[position])
            # The first tag is the label. A chunk reached through two topics has two, but
            # ARI needs one true label per point, and the first is the topic that
            # introduced the paper.
            labels.append(tags[0] if tags else "untagged")
            texts.append(chunk["text"])
            kept_ids.append(chunk_id)

        if len(rows) < 4:
            return _error("too_few_chunks",
                          f"clustering needs at least 4 chunks, found {len(rows)}")

        matrix = np.vstack(rows)
        low, high = (int(x) for x in self.config.analysis.cluster_k_range)
        high = min(high, len(rows) - 1)
        forced_k = params.get("k")
        candidates = [int(forced_k)] if forced_k else list(range(max(2, low), max(3, high) + 1))

        scored = []
        for k in candidates:
            if k >= len(rows):
                continue
            model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(matrix)
            score = float(silhouette_score(matrix, model.labels_)) if k > 1 else -1.0
            scored.append({"k": k, "silhouette": round(score, 4), "model": model})
        if not scored:
            return _error("no_valid_k", "no usable k in the configured range")

        best = max(scored, key=lambda entry: entry["silhouette"])
        assignments = best["model"].labels_

        # TF-IDF terms per cluster, as human-readable labels.
        vectoriser = TfidfVectorizer(stop_words="english", max_features=4000)
        tfidf = vectoriser.fit_transform(texts)
        vocabulary = np.asarray(vectoriser.get_feature_names_out())
        top_n = int(self.config.analysis.tfidf_top_terms)
        cluster_labels: Dict[str, List[str]] = {}
        for cluster in range(best["k"]):
            mask = assignments == cluster
            if not mask.any():
                cluster_labels[str(cluster)] = []
                continue
            centroid = np.asarray(tfidf[mask].mean(axis=0)).ravel()
            cluster_labels[str(cluster)] = vocabulary[np.argsort(centroid)[-top_n:][::-1]].tolist()

        # Validation against the tags we already know.
        unique_tags = sorted(set(labels))
        truth = np.array([unique_tags.index(tag) for tag in labels])
        ari = float(adjusted_rand_score(truth, assignments))
        purity_total = 0
        for cluster in range(best["k"]):
            mask = assignments == cluster
            if mask.any():
                purity_total += Counter(np.asarray(labels)[mask]).most_common(1)[0][1]
        purity = purity_total / len(labels)

        result = {
            "k": best["k"],
            "k_selected_by": "forced" if forced_k else "silhouette",
            "silhouette": best["silhouette"],
            "silhouette_by_k": {str(e["k"]): e["silhouette"] for e in scored},
            "cluster_labels": cluster_labels,
            "cluster_sizes": {str(c): int((assignments == c).sum()) for c in range(best["k"])},
            "adjusted_rand_index": round(ari, 4),
            "purity": round(purity, 4),
            "known_topics": unique_tags,
            "n_chunks": len(rows),
            "membership": {chunk_id: int(cluster)
                           for chunk_id, cluster in zip(kept_ids, assignments)},
        }

        verdict = (
            "the clusters closely recover the known topic split"
            if ari >= 0.5 else
            "the clusters partly recover the known topic split"
            if ari >= 0.2 else
            "the clusters do NOT recover the known topic split, which suggests the "
            "embeddings are not separating these topics"
        )
        summary = (
            f"KMeans over {_plural(len(rows), 'chunk')} selected k={best['k']} by silhouette "
            f"({best['silhouette']:.3f}). Validated against {len(unique_tags)} known topic "
            f"tag(s): adjusted Rand index {ari:.3f}, purity {purity:.3f} — {verdict}. "
            + "; ".join(
                f"cluster {c} ({result['cluster_sizes'][c]} chunks): "
                + ", ".join(terms[:4])
                for c, terms in cluster_labels.items()
            )
            + "."
        )

        payload = {"operation": "cluster", "result": result, "summary": summary}
        if not params:
            cache.setdefault(cache_key, {})[scope] = payload
            write_json_atomic(self.config.paths.clusters, cache)
        return {**payload, "cached": False}

    def analyze(
        self,
        operation: str,
        topic_filter: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch to one operation. This is the tool entry point."""
        operation = (operation or "").strip().lower()
        if operation not in OPERATIONS:
            return _error(
                "unknown_operation",
                f"unknown operation {operation!r}; expected one of {list(OPERATIONS)}",
            )
        if operation == "stats":
            return self.stats(topic_filter)
        if operation == "timeline":
            return self.timeline(topic_filter)
        if operation == "compare_topics":
            return self.compare_topics(topic_filter)
        return self.cluster(topic_filter, params)


ANALYZE_CORPUS_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": list(OPERATIONS),
            "description": (
                "stats: counts and coverage. timeline: years and recency. "
                "cluster: sub-themes, validated against known topic tags. "
                "compare_topics: vocabulary overlap between topic groups."
            ),
        },
        "topic_filter": {
            "type": "string",
            "description": "Restrict the analysis to papers carrying this topic tag.",
        },
        "params": {
            "type": "object",
            "description": "Operation options, e.g. {'k': 4} to force a cluster count.",
        },
    },
    "required": ["operation"],
    "additionalProperties": False,
}
