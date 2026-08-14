"""
Registry of the agent-facing tools: their JSON schemas and dispatch by name.

In:  tool specs (name, description file, parameter schema, callable).
Out: the OpenAI `tools=[...]` payload, and `dispatch(name, args)` returning the tool's
     dict. Descriptions load from src/prompts/tools/ so they can be diffed like prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from config import CFG, Config


@dataclass
class ToolSpec:
    """One tool as the model sees it, plus the callable that actually runs it."""

    name: str
    parameters: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    description: str = ""

    def schema(self) -> Dict[str, Any]:
        """The OpenAI function-calling schema entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the tools the agent may call, and routes a call to the right one.

    Tool descriptions are loaded from files rather than written inline: they are the
    exact text the model reads, they get iterated on, and a diff of a prompt change is
    only readable if the prompt lives in a file.
    """

    def __init__(self, config: Config = CFG) -> None:
        self.config = config
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Dict[str, Any]],
        description: Optional[str] = None,
    ) -> None:
        """Add a tool. `description` defaults to src/prompts/tools/{name}.md."""
        if description is None:
            description = self.config.prompt(f"tools/{name}").strip()
        self._tools[name] = ToolSpec(
            name=name, parameters=parameters, handler=handler, description=description
        )

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """The `tools` payload for a completion request."""
        return [self._tools[name].schema() for name in sorted(self._tools)]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run a tool by name. Always returns a dict, never raises.

        An unknown name means the model hallucinated a tool. That is recoverable — it
        gets told what actually exists and can pick again — so it is an error result,
        not an exception that would end the run.
        """
        spec = self._tools.get(name)
        if spec is None:
            return {
                "error": "unknown_tool",
                "detail": f"no tool named {name!r}. Available tools: {', '.join(self.names)}",
                "partial": None,
            }
        try:
            result = spec.handler(**arguments)
        except TypeError as exc:
            # Wrong or missing arguments: the model can correct this on the next turn.
            return {
                "error": "bad_arguments",
                "detail": f"{name} rejected these arguments: {exc}",
                "partial": None,
            }
        except Exception as exc:
            # A tool that raises would kill the loop. docs/TOOLS.md requires that tools
            # never raise; this is the backstop for one that does anyway.
            return {
                "error": "tool_crashed",
                "detail": f"{name} raised {type(exc).__name__}: {exc}",
                "partial": None,
            }

        if not isinstance(result, dict):
            return {
                "error": "bad_tool_result",
                "detail": f"{name} returned {type(result).__name__}, expected a dict",
                "partial": None,
            }
        return result


RETRIEVE_EVIDENCE_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "What to search the indexed papers for, in natural language.",
        },
        "k": {
            "type": "integer",
            "description": "How many chunks to return. Defaults to 5.",
        },
        "topic_filter": {
            "type": "string",
            "description": "Restrict results to papers carrying this topic tag.",
        },
        "chunk_types": {
            "type": "array",
            "items": {"type": "string", "enum": ["text", "figure", "table"]},
            "description": "Restrict results to these chunk types.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def build_default_registry(retriever, config: Config = CFG) -> ToolRegistry:
    """Retrieval only — the M5 registry, kept for tests that want one tool."""
    registry = ToolRegistry(config)
    registry.register("retrieve_evidence", RETRIEVE_EVIDENCE_PARAMETERS, retriever.retrieve)
    return registry


def build_full_registry(
    retriever=None,
    config: Config = CFG,
    client=None,
) -> "ToolRegistry":
    """All five tools, per docs/TOOLS.md.

    `search_literature` is given a callback that clears the retriever's cached view of
    the corpus. The retriever holds the index and chunk table in memory for the life of a
    conversation, so papers added mid-conversation would otherwise be invisible to every
    later retrieval — silently, since a stale index still returns plausible results.
    The callback keeps control flow in the agent: no tool calls another tool.
    """
    from tools.analyze_corpus import ANALYZE_CORPUS_PARAMETERS, CorpusAnalyzer
    from tools.check_evidence_consistency import (
        CHECK_CONSISTENCY_PARAMETERS,
        ConsistencyChecker,
    )
    from tools.inspect_figure import INSPECT_FIGURE_PARAMETERS, FigureInspector
    from tools.retrieve_evidence import EvidenceRetriever
    from tools.search_literature import SEARCH_LITERATURE_PARAMETERS, LiteratureSearcher
    from tools.explore_graph import EXPLORE_GRAPH_PARAMETERS, GraphExplorer

    retriever = retriever if retriever is not None else EvidenceRetriever(config)
    registry = ToolRegistry(config)

    registry.register("retrieve_evidence", RETRIEVE_EVIDENCE_PARAMETERS, retriever.retrieve)
    registry.register(
        "search_literature",
        SEARCH_LITERATURE_PARAMETERS,
        LiteratureSearcher(config, client=client, on_corpus_change=retriever.invalidate).search,
    )
    registry.register(
        "analyze_corpus", ANALYZE_CORPUS_PARAMETERS, CorpusAnalyzer(config).analyze
    )
    registry.register(
        "explore_graph", EXPLORE_GRAPH_PARAMETERS, GraphExplorer(config).explore
    )
    registry.register(
        "inspect_figure", INSPECT_FIGURE_PARAMETERS, FigureInspector(config).inspect
    )
    registry.register(
        "check_evidence_consistency",
        CHECK_CONSISTENCY_PARAMETERS,
        ConsistencyChecker(config, client=client).check,
    )
    return registry
