"""
The message array the loop re-sends on every call — the system's only memory.

In:  a system prompt, the user question, assistant turns, and tool results.
Out: `messages()`, the full conversation in provider wire format. Enforces the pairing
     rules the API requires, and elides old tool content when the array outgrows budget.

The provider is stateless. /chat/completions remembers nothing between requests, so the
model "recalls" its own earlier tool call only because that call and its result are
sitting in the array we re-send. This class owns that array and nothing else does.
"""

from __future__ import annotations

import copy
import json
from typing import Any, List

from common.tokenization import TokenCounter
from config import CFG, Config
from llm_client import LLMResponse, Message


class ConversationError(RuntimeError):
    """The message array is in a state the API would reject."""


def _serialise(result: Any) -> str:
    """Tool results are dicts; the wire format wants a string."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"error": "unserialisable_result", "detail": repr(result)[:500]})


class Conversation:
    """Builds and holds the messages array, guarding the rules that break silently.

    Three of them are structural and the API rejects the request if they are violated:
    every assistant tool_call needs exactly one tool message, the tool_call_id must match
    exactly, and the assistant turn must be appended before its results. The fourth —
    keeping the array inside a token budget — is ours to manage.
    """

    def __init__(self, system_prompt: str, question: str, config: Config = CFG) -> None:
        self.config = config
        self.counter = TokenCounter(config)
        self._messages: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        self._pending: List[str] = []  # tool_call_ids awaiting a result
        self.elided = 0

    # ---- reading ---------------------------------------------------------------

    def messages(self) -> List[Message]:
        """The whole conversation, ready to send. Every call re-sends all of it."""
        if self._pending:
            raise ConversationError(
                f"tool_call_id(s) {self._pending} have no result yet; the API rejects an "
                "assistant tool_call that is not answered by a tool message"
            )
        return [copy.deepcopy(message) for message in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def token_estimate(self) -> int:
        """Approximate size of the array, for the elision decision."""
        total = 0
        for message in self._messages:
            content = message.get("content")
            if isinstance(content, str):
                total += self.counter.count(content)
            for call in message.get("tool_calls") or []:
                total += self.counter.count(str((call.get("function") or {}).get("arguments", "")))
        return total

    def final_text(self) -> str:
        """The last assistant message's text — the answer, once the loop stops."""
        for message in reversed(self._messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    # ---- writing ---------------------------------------------------------------

    def add_assistant(self, response: LLMResponse) -> Message:
        """Append the assistant turn exactly as the provider returned it.

        Verbatim, not reconstructed. The tool_call ids in this message are the only link
        to the results that follow, so re-deriving the message risks changing an id and
        silently breaking the pairing.
        """
        message = response.assistant_message()
        self._messages.append(message)
        self._pending = [call.id for call in response.tool_calls]
        return message

    def add_tool_result(self, tool_call_id: str, result: Any) -> Message:
        """Append one tool result, answering the tool_call with this exact id."""
        if tool_call_id not in self._pending:
            raise ConversationError(
                f"no unanswered tool_call with id {tool_call_id!r}; "
                f"awaiting {self._pending or 'nothing'}"
            )
        message: Message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _serialise(result),
        }
        self._messages.append(message)
        self._pending.remove(tool_call_id)
        return message

    def add_user(self, content: Any) -> Message:
        """Append a user turn. Images ride here, never in a tool result."""
        message: Message = {"role": "user", "content": content}
        self._messages.append(message)
        return message

    # ---- context budget --------------------------------------------------------

    def elide_if_needed(self) -> int:
        """Shrink the array to fit the budget. Returns how many results were elided.

        Elides the *content* of the oldest tool results and leaves the messages in place.
        Deleting them instead would orphan the assistant tool_calls they answer, and the
        API rejects a request whose tool_calls are unanswered — so a deletion strategy
        fails outright rather than degrading.

        The placeholder still tells the agent that search already happened, so it does
        not simply repeat it.
        """
        budget = int(self.config.agent.max_context_tokens)
        elided = 0
        for message in self._messages:
            if self.token_estimate() <= budget:
                break
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or content.startswith("[earlier result elided"):
                continue
            message["content"] = self._placeholder(content)
            elided += 1
        self.elided += elided
        return elided

    def _placeholder(self, content: str) -> str:
        """A short stand-in naming what was dropped, so the turn still carries meaning."""
        limit = int(self.config.agent.elision_placeholder_chars)
        try:
            payload = json.loads(content)
        except ValueError:
            return f"[earlier result elided: {len(content)} characters]"[:limit]

        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if isinstance(chunks, list) and chunks:
            papers = sorted({str(chunk.get("paper_id", "?")) for chunk in chunks})
            return (
                f"[earlier result elided: {len(chunks)} chunk(s) from {', '.join(papers)}. "
                "You already ran this search.]"
            )[:limit]
        return "[earlier result elided. You already ran this search.]"[:limit]
