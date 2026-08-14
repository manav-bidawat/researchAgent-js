"""
`explore_graph`: graph neighbourhood traversal.

In: an entity_id (paper_id, chunk_id, figure_id, or topic name) and optional depth.
Out: graph paths radiating from that entity.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import CFG, Config
from graph.neo4j_store import GraphStore

EXPLORE_GRAPH_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity_id": {
            "type": "string",
            "description": "The exact ID of the paper, chunk, figure, or the exact name of the topic.",
        },
        "depth": {
            "type": "integer",
            "description": "How many steps to traverse from the entity. Defaults to 1. Max 3.",
        },
    },
    "required": ["entity_id"],
    "additionalProperties": False,
}

class GraphExplorer:
    def __init__(self, config: Config = CFG) -> None:
        self.config = config
        self.store = GraphStore(config)

    def explore(self, entity_id: str, depth: Optional[int] = None) -> Dict[str, Any]:
        """Return the graph neighbourhood of `entity_id`."""
        d = int(depth) if depth is not None else 1
        return self.store.neighbourhood(entity_id, depth=d)
