"""
Thin OpenRouter client — the only module in the system that talks to an LLM provider.

In:  OpenAI-format `messages`, optional tool schemas, and a role ('agent' | 'vision' |
     'utility') that selects the model from config.
Out: a normalised LLMResponse (text, parsed tool calls, usage, raw payload). Raises
     LLMError on failure; tool wrappers catch it and return the documented error dict.
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from config import CFG, Config

Message = Dict[str, Any]
ImageSource = Union[str, Path, bytes]

# Transient by nature: rate limits, gateway hiccups, upstream timeouts.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
DEFAULT_IMAGE_MIME = "image/png"

# Capacity and rate-limit failures arrive inside an HTTP 200 body as often as they do
# as a status code — free endpoints in particular answer 200 with
# "ResourceExhausted: Worker local total request limit reached". These are transient and
# must be retried; a permanent error (bad model id, malformed request) must not be.
TRANSIENT_BODY_MARKERS = (
    "resourceexhausted",
    "resource exhausted",
    "rate limit",
    "rate-limit",
    "ratelimit",
    "too many requests",
    "overloaded",
    "temporarily unavailable",
    "capacity",
    "try again",
    "timeout",
    "timed out",
    "request limit reached",
)


class LLMError(RuntimeError):
    """A provider call failed. `code` is machine-ish, `detail` is safe to show the model.

    llm_client raises; it is not a tool. Tool wrappers catch this and return
    {"error": code, "detail": detail} so the agent loop degrades instead of crashing.
    """

    def __init__(
        self, code: str, detail: str, status: Optional[int] = None, retryable: bool = False
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class ToolCall:
    """One function call the model asked for, with its arguments already parsed."""

    id: str
    name: str
    arguments: Dict[str, Any]
    arguments_raw: str
    parse_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """False when the model emitted arguments that were not valid JSON."""
        return self.parse_error is None


@dataclass(frozen=True)
class LLMResponse:
    """One completion, normalised. `raw` keeps the untouched provider payload."""

    text: str
    tool_calls: List[ToolCall]
    finish_reason: str
    model: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def assistant_message(self) -> Message:
        """The assistant turn to append to the conversation, in provider wire format.

        Returned verbatim from the payload so tool_call ids round-trip exactly; the
        agent loop must append this before appending any tool result.
        """
        message = self.raw.get("choices", [{}])[0].get("message")
        if not isinstance(message, dict):
            return {"role": "assistant", "content": self.text}
        return copy.deepcopy(message)


def encode_image(image: ImageSource, mime_type: Optional[str] = None) -> str:
    """Turn a path or raw bytes into a `data:` URL for an image_url content part."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise LLMError("image_not_found", f"no image at {path}")
        payload = path.read_bytes()
        mime_type = mime_type or mimetypes.guess_type(path.name)[0] or DEFAULT_IMAGE_MIME
    elif isinstance(image, (bytes, bytearray)):
        payload = bytes(image)
        mime_type = mime_type or DEFAULT_IMAGE_MIME
    else:
        raise LLMError("bad_image", f"unsupported image type {type(image).__name__}")

    return f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"


def attach_images(messages: Sequence[Message], images: Sequence[ImageSource]) -> List[Message]:
    """Copy `messages` with `images` appended to the last user turn, as image_url parts.

    Images must ride on a user turn: the OpenAI-compatible schema has no image block
    inside a tool result, so a tool that produces an image hands it back through here.
    """
    conversation = [copy.deepcopy(message) for message in messages]
    if not images:
        return conversation

    parts = [{"type": "image_url", "image_url": {"url": encode_image(image)}} for image in images]

    target = next((m for m in reversed(conversation) if m.get("role") == "user"), None)
    if target is None:
        conversation.append({"role": "user", "content": parts})
        return conversation

    content = target.get("content")
    if isinstance(content, str):
        target["content"] = [{"type": "text", "text": content}, *parts]
    elif isinstance(content, list):
        target["content"] = [*content, *parts]
    else:
        target["content"] = parts
    return conversation


class LLMClient:
    """Talks to one OpenAI-compatible endpoint (OpenRouter) over plain HTTP.

    Model, timeout and retry policy come from config; nothing here is hardcoded, so
    swapping provider or model is a config edit rather than a code change.
    """

    def __init__(self, config: Config = CFG, session: Optional[requests.Session] = None) -> None:
        self._config = config
        self._session = session if session is not None else requests.Session()

    def model_for(self, role: str) -> str:
        """Model id configured for an LLM role."""
        return self._config.model_for(role)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        role: str = "agent",
        model: Optional[str] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> LLMResponse:
        """One chat completion, optionally offering `tools` for the model to call.

        `timeout` and `max_retries` override the config for this call. Short, retryable
        tasks should shorten both: transport retries multiply with any retry loop the
        caller runs on top, and the product is the real worst-case latency.
        """
        payload: Dict[str, Any] = {
            "model": model or self.model_for(role),
            "messages": [copy.deepcopy(message) for message in messages],
            "temperature": self._config.llm.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = list(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        return _parse_response(self._post(payload, timeout=timeout, max_retries=max_retries))

    def complete_vision(
        self,
        messages: Sequence[Message],
        images: Sequence[ImageSource],
        role: str = "vision",
        **kwargs: Any,
    ) -> LLMResponse:
        """A completion that shows the model actual images, appended to the last user turn."""
        return self.complete(attach_images(messages, images), role=role, **kwargs)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.require_api_key()}",
            "Content-Type": "application/json",
            "X-Title": self._config.app_name,
        }

    def _post(
        self,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """POST to /chat/completions, retrying transient failures with backoff."""
        url = f"{str(self._config.llm.base_url).rstrip('/')}/chat/completions"
        headers = self._headers()
        attempts = max(1, int(self._config.llm.max_retries if max_retries is None else max_retries))
        base_delay = float(self._config.llm.retry_base_delay_s)
        request_timeout = float(self._config.llm.timeout_s if timeout is None else timeout)
        last_error: Optional[LLMError] = None

        for attempt in range(attempts):
            retry_after: Optional[str] = None
            try:
                response = self._session.post(
                    url, headers=headers, json=payload, timeout=request_timeout
                )
            except requests.Timeout as exc:
                last_error = LLMError("llm_timeout", f"request timed out: {exc}")
            except requests.RequestException as exc:
                last_error = LLMError("llm_unreachable", f"could not reach {url}: {exc}")
            else:
                if response.status_code in RETRYABLE_STATUS:
                    retry_after = (response.headers or {}).get("Retry-After")
                    last_error = LLMError(
                        "llm_transient", _body_snippet(response), response.status_code
                    )
                elif response.status_code >= 400:
                    # 4xx other than the retryable ones is our bug: bad key, bad payload.
                    raise LLMError(
                        "llm_request_rejected", _body_snippet(response), response.status_code
                    )
                else:
                    try:
                        return _decode_body(response)
                    except LLMError as exc:
                        if not exc.retryable:
                            raise
                        last_error = exc

            if attempt < attempts - 1:
                time.sleep(_backoff_delay(base_delay, attempt, retry_after))

        raise last_error if last_error else LLMError("llm_failed", "no attempts were made")


def _backoff_delay(base_delay: float, attempt: int, retry_after: Optional[str] = None) -> float:
    """Exponential backoff with jitter, or the server's Retry-After when it sent one."""
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except (TypeError, ValueError):
            pass
    return base_delay * (2**attempt) + random.uniform(0.0, base_delay)


def _body_snippet(response: Any, limit: int = 400) -> str:
    text = (getattr(response, "text", "") or "").strip().replace("\n", " ")
    return f"HTTP {getattr(response, 'status_code', '?')}: {text[:limit]}"


def _decode_body(response: Any) -> Dict[str, Any]:
    """JSON body, or an LLMError. OpenRouter can return an error inside an HTTP 200."""
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMError("llm_bad_response", f"response was not JSON: {exc}") from None

    if not isinstance(body, dict):
        raise LLMError("llm_bad_response", f"expected a JSON object, got {type(body).__name__}")

    error = body.get("error")
    if error:
        detail = str(error.get("message") if isinstance(error, dict) else error)
        lowered = detail.lower()
        transient = any(marker in lowered for marker in TRANSIENT_BODY_MARKERS)
        raise LLMError("llm_provider_error", detail, retryable=transient)

    return body


def _parse_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    """Normalise the provider's tool_calls, keeping malformed JSON as a parse_error."""
    calls: List[ToolCall] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_arguments)
            parse_error = None
        except (TypeError, ValueError) as exc:
            parsed, parse_error = {}, f"arguments were not valid JSON: {exc}"
        if not isinstance(parsed, dict):
            parsed, parse_error = {}, "arguments did not decode to an object"

        calls.append(
            ToolCall(
                id=str(call.get("id") or f"call_{index}"),
                name=str(function.get("name") or ""),
                arguments=parsed,
                arguments_raw=str(raw_arguments),
                parse_error=parse_error,
            )
        )
    return calls


def _parse_response(body: Dict[str, Any]) -> LLMResponse:
    """Provider payload -> LLMResponse."""
    choices = body.get("choices") or []
    if not choices:
        raise LLMError("llm_empty_response", "provider returned no choices")

    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        # Some providers return content as parts even for plain text.
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

    return LLMResponse(
        text=content or "",
        tool_calls=_parse_tool_calls(message),
        finish_reason=str(choice.get("finish_reason") or ""),
        model=str(body.get("model") or ""),
        usage=body.get("usage") or {},
        raw=body,
    )
