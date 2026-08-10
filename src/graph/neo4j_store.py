"""Neo4j projection of papers, chunks, topics, and figures.

FAISS remains the retrieval index: it is the right structure for semantic nearest-
neighbour search. This module adds the relationships that FAISS intentionally does
not model, so an MCP client can traverse a paper, its chunks, its topics, and figures.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import CFG, Config
from corpus.chunk_store import ChunkStore
from corpus.manifest import Manifest, ManifestError


class GraphStoreError(RuntimeError):
    """Neo4j is unavailable or rejected a graph operation."""


def _batches(rows: Sequence[Dict[str, Any]], size: int = 250) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


class GraphStore:
    """Projects the file-backed corpus into Neo4j and reads local graph neighbourhoods.

    The driver is imported only when a graph operation is requested. This keeps the
    normal FAISS-only CLI usable when Neo4j is not installed or configured.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.config = config
        graph_config = getattr(config, "graph", None)
        self.uri = os.environ.get(
            "NEO4J_URI", graph_config.get("uri", "") if graph_config else ""
        ).strip()
        self.username = os.environ.get(
            "NEO4J_USERNAME", graph_config.get("username", "neo4j") if graph_config else "neo4j"
        ).strip()
        self.password = os.environ.get(
            "NEO4J_PASSWORD", graph_config.get("password", "") if graph_config else ""
        ).strip()
        self.database = os.environ.get(
            "NEO4J_DATABASE", graph_config.get("database", "neo4j") if graph_config else "neo4j"
        ).strip() or "neo4j"

    @property
    def configured(self) -> bool:
        return bool(self.uri and self.username and self.password)

    def _driver(self):
        if not self.configured:
            raise GraphStoreError(
                "Neo4j is not configured; set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD"
            )
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise GraphStoreError("the neo4j package is not installed") from exc
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            driver.verify_connectivity()
            return driver
        except Exception as exc:
            raise GraphStoreError(f"could not connect to Neo4j at {self.uri}: {exc}") from exc

    @staticmethod
    def _run(session: Any, query: str, rows: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        if rows is None:
            session.run(query).consume()
            return
        for batch in _batches(rows):
            session.run(query, rows=list(batch)).consume()

    def sync(self) -> Dict[str, Any]:
        """Upsert the current manifest and chunk store into the graph.

        This operation never replaces the database wholesale: `MERGE` preserves any
        analyst-added graph annotations while refreshing corpus-owned properties.
        """
        try:
            manifest = Manifest.load(self.config, strict_model_check=False)
        except ManifestError as exc:
            return {"error": "manifest_unreadable", "detail": str(exc), "partial": None}

        papers = list(manifest.papers.values())
        chunks = list(ChunkStore(config=self.config))
        topics = sorted({tag for paper in papers for tag in paper.get("topic_tags") or []})
        figures = list(manifest.figures.values())
        chunks_by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_paper[str(chunk["paper_id"])].append(chunk)

        next_rows: List[Dict[str, str]] = []
        for paper_chunks in chunks_by_paper.values():
            ordered = sorted(paper_chunks, key=lambda item: int(item.get("position", 0)))
            next_rows.extend(
                {"left": left["chunk_id"], "right": right["chunk_id"]}
                for left, right in zip(ordered, ordered[1:])
            )

        try:
            driver = self._driver()
            with driver:
                with driver.session(database=self.database) as session:
                    for statement in (
                        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (node:Paper) REQUIRE node.paper_id IS UNIQUE",
                        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (node:Chunk) REQUIRE node.chunk_id IS UNIQUE",
                        "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (node:Topic) REQUIRE node.name IS UNIQUE",
                        "CREATE CONSTRAINT figure_id IF NOT EXISTS FOR (node:Figure) REQUIRE node.figure_id IS UNIQUE",
                    ):
                        self._run(session, statement)

                    self._run(session, """
                        UNWIND $rows AS row
                        MERGE (paper:Paper {paper_id: row.paper_id})
                        SET paper.arxiv_id = row.arxiv_id, paper.title = row.title,
                            paper.authors = row.authors, paper.abstract = row.abstract,
                            paper.year = row.year, paper.published = row.published,
                            paper.categories = row.categories, paper.parse_status = row.parse_status
                    """, papers)
                    self._run(session, "UNWIND $rows AS row MERGE (:Topic {name: row.name})",
                              [{"name": topic} for topic in topics])
                    self._run(session, """
                        UNWIND $rows AS row
                        MATCH (paper:Paper {paper_id: row.paper_id})
                        UNWIND row.topic_tags AS topic_name
                        MATCH (topic:Topic {name: topic_name})
                        MERGE (paper)-[:TAGGED_WITH]->(topic)
                    """, papers)
                    self._run(session, """
                        UNWIND $rows AS row
                        MERGE (chunk:Chunk {chunk_id: row.chunk_id})
                        SET chunk.paper_id = row.paper_id, chunk.paper_title = row.paper_title,
                            chunk.text = row.text, chunk.chunk_type = row.chunk_type,
                            chunk.page = row.page, chunk.section = row.section,
                            chunk.position = row.position, chunk.n_tokens = row.n_tokens
                    """, chunks)
                    self._run(session, """
                        UNWIND $rows AS row
                        MATCH (paper:Paper {paper_id: row.paper_id})
                        MATCH (chunk:Chunk {chunk_id: row.chunk_id})
                        MERGE (paper)-[:HAS_CHUNK]->(chunk)
                        WITH chunk, row
                        UNWIND row.topic_tags AS topic_name
                        MATCH (topic:Topic {name: topic_name})
                        MERGE (chunk)-[:TAGGED_WITH]->(topic)
                    """, chunks)
                    self._run(session, """
                        UNWIND $rows AS row
                        MATCH (left:Chunk {chunk_id: row.left})
                        MATCH (right:Chunk {chunk_id: row.right})
                        MERGE (left)-[:NEXT]->(right)
                    """, next_rows)
                    self._run(session, """
                        UNWIND $rows AS row
                        MERGE (figure:Figure {figure_id: row.figure_id})
                        SET figure.paper_id = row.paper_id, figure.kind = row.kind,
                            figure.page = row.page, figure.caption = row.caption,
                            figure.description = row.description
                        WITH figure, row
                        MATCH (paper:Paper {paper_id: row.paper_id})
                        MERGE (paper)-[:HAS_FIGURE]->(figure)
                        WITH figure, row
                        OPTIONAL MATCH (chunk:Chunk {chunk_id: row.chunk_id})
                        FOREACH (_ IN CASE WHEN chunk IS NULL THEN [] ELSE [1] END |
                            MERGE (chunk)-[:DESCRIBES]->(figure))
                    """, figures)
        except GraphStoreError as exc:
            return {"error": "graph_unavailable", "detail": str(exc), "partial": None}
        except Exception as exc:
            return {"error": "graph_sync_failed", "detail": str(exc), "partial": None}

        return {
            "papers": len(papers), "chunks": len(chunks), "topics": len(topics),
            "figures": len(figures), "next_relationships": len(next_rows),
        }

    def neighbourhood(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """Return graph paths around one paper, chunk, topic, or figure identifier."""
        entity_id = (entity_id or "").strip()
        if not entity_id:
            return {"error": "empty_entity_id", "detail": "an entity_id is required", "partial": None}
        if depth < 1 or depth > 3:
            return {"error": "bad_depth", "detail": "depth must be between 1 and 3", "partial": None}

        query = f"""
            MATCH (root)
            WHERE root.paper_id = $entity_id OR root.chunk_id = $entity_id
               OR root.figure_id = $entity_id OR root.name = $entity_id
            OPTIONAL MATCH path = (root)-[*1..{int(depth)}]-(neighbor)
            RETURN labels(root) AS root_labels, properties(root) AS root,
                   [node IN nodes(path) | {{labels: labels(node), properties: properties(node)}}] AS nodes,
                   [rel IN relationships(path) | {{type: type(rel), start: elementId(startNode(rel)),
                       end: elementId(endNode(rel)), properties: properties(rel)}}] AS relationships
            LIMIT 50
        """
        try:
            driver = self._driver()
            with driver:
                with driver.session(database=self.database) as session:
                    records = list(session.run(query, entity_id=entity_id))
        except GraphStoreError as exc:
            return {"error": "graph_unavailable", "detail": str(exc), "partial": None}
        except Exception as exc:
            return {"error": "graph_query_failed", "detail": str(exc), "partial": None}

        if not records:
            return {
                "error": "unknown_graph_entity",
                "detail": f"no paper, chunk, topic, or figure has id {entity_id!r}; run graph sync first",
                "partial": None,
            }
        paths = [record.data() if hasattr(record, "data") else dict(record) for record in records]
        return {"entity_id": entity_id, "depth": depth, "paths": paths}
