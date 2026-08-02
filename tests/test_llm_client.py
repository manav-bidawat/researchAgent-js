"""
Offline tests for the OpenRouter wrapper: response parsing, image attachment, retries.

In:  a fake requests.Session that returns canned payloads — no network, no API key.
Out: assertions on LLMResponse shape, tool-call parsing, and error classification.
"""

import json

import pytest
import requests

from config import CFG
from llm_client import (
    LLMClient,
    LLMError,
    LLMResponse,
    attach_images,
    encode_image,
)


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=None, headers=None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else json.dumps(body)
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    """Replays a queue of responses (or exceptions) and records what was posted."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "payload": json, "timeout": timeout})
        result = self.responses.pop(0) if self.responses else FakeResponse(500, text="exhausted")
        if isinstance(result, Exception):
            raise result
        return result


def _text_payload(text="hello"):
    return {
        "model": "test/model",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 7},
    }


def _tool_payload(name="retrieve_evidence", arguments='{"query": "graph transformers", "k": 5}'):
    return {
        "model": "test/model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _client(*responses):
    session = FakeSession(*responses)
    return LLMClient(session=session), session


def test_text_completion_is_normalised():
    client, session = _client(FakeResponse(200, _text_payload("hi there")))
    result = client.complete([{"role": "user", "content": "hi"}])

    assert isinstance(result, LLMResponse)
    assert result.text == "hi there"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 7
    assert session.calls[0]["url"].endswith("/chat/completions")
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_role_selects_the_configured_model():
    client, session = _client(FakeResponse(200, _text_payload()))
    client.complete([{"role": "user", "content": "hi"}], role="utility")
    assert session.calls[0]["payload"]["model"] == CFG.llm.utility_model


def test_tool_call_round_trip_is_well_formed():
    client, session = _client(FakeResponse(200, _tool_payload()))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_evidence",
                "description": "retrieve passages",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"],
                },
            },
        }
    ]
    result = client.complete([{"role": "user", "content": "what do the papers say?"}], tools=tools)

    assert result.has_tool_calls
    call = result.tool_calls[0]
    assert (call.id, call.name, call.ok) == ("call_abc", "retrieve_evidence", True)
    assert call.arguments == {"query": "graph transformers", "k": 5}
    assert session.calls[0]["payload"]["tools"] == tools

    # The assistant turn must round-trip verbatim so tool_call ids still match.
    assert result.assistant_message()["tool_calls"][0]["id"] == "call_abc"


def test_malformed_tool_arguments_are_flagged_not_raised():
    client, _ = _client(FakeResponse(200, _tool_payload(arguments="{not json")))
    call = client.complete([{"role": "user", "content": "x"}]).tool_calls[0]
    assert not call.ok
    assert call.arguments == {}
    assert "not valid JSON" in call.parse_error


def test_transient_status_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("llm_client.time.sleep", lambda _seconds: None)
    client, session = _client(
        FakeResponse(429, text="rate limited", headers={"Retry-After": "0"}),
        FakeResponse(200, _text_payload("recovered")),
    )
    assert client.complete([{"role": "user", "content": "x"}]).text == "recovered"
    assert len(session.calls) == 2


def test_network_failure_raises_a_typed_error(monkeypatch):
    monkeypatch.setattr("llm_client.time.sleep", lambda _seconds: None)
    client, _ = _client(*[requests.ConnectionError("down")] * int(CFG.llm.max_retries))
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "llm_unreachable"


def test_client_error_is_not_retried():
    client, session = _client(FakeResponse(401, text="no credentials"))
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "llm_request_rejected"
    assert excinfo.value.status == 401
    assert len(session.calls) == 1


def test_error_inside_http_200_is_surfaced():
    client, _ = _client(FakeResponse(200, {"error": {"message": "upstream is down"}}))
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "llm_provider_error"


def test_transient_error_inside_http_200_is_retried(monkeypatch):
    """Free endpoints answer 200 with a capacity error. That is transient, not fatal."""
    monkeypatch.setattr("llm_client.time.sleep", lambda _seconds: None)
    exhausted = {"error": {"message": "Upstream error from Nvidia: ResourceExhausted: "
                                      "Worker local total request limit reached (16/16)"}}
    client, session = _client(
        FakeResponse(200, exhausted),
        FakeResponse(200, _text_payload("recovered")),
    )
    assert client.complete([{"role": "user", "content": "x"}]).text == "recovered"
    assert len(session.calls) == 2


def test_permanent_error_inside_http_200_is_not_retried():
    client, session = _client(FakeResponse(200, {"error": {"message": "model not found"}}))
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "llm_provider_error"
    assert excinfo.value.retryable is False
    assert len(session.calls) == 1, "a permanent error must not burn the retry budget"


def test_empty_choices_is_an_error():
    client, _ = _client(FakeResponse(200, {"choices": []}))
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert excinfo.value.code == "llm_empty_response"


def test_encode_image_from_bytes_and_path(tmp_path):
    assert encode_image(b"\x89PNG\r\n").startswith("data:image/png;base64,")

    jpeg = tmp_path / "fig.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff")
    assert encode_image(jpeg).startswith("data:image/jpeg;base64,")

    with pytest.raises(LLMError, match="image_not_found"):
        encode_image(tmp_path / "missing.png")


def test_attach_images_rides_on_the_last_user_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look at this"},
    ]
    attached = attach_images(messages, [b"\x89PNG"])

    assert messages[1]["content"] == "look at this", "input must not be mutated"
    content = attached[1]["content"]
    assert content[0] == {"type": "text", "text": "look at this"}
    assert content[1]["type"] == "image_url"


def test_attach_images_appends_a_user_turn_when_there_is_none():
    attached = attach_images([{"role": "system", "content": "sys"}], [b"\x89PNG"])
    assert attached[-1]["role"] == "user"
    assert attached[-1]["content"][0]["type"] == "image_url"


def test_complete_vision_uses_the_vision_model():
    client, session = _client(FakeResponse(200, _text_payload("a scatter plot")))
    result = client.complete_vision([{"role": "user", "content": "describe"}], [b"\x89PNG"])

    assert result.text == "a scatter plot"
    assert session.calls[0]["payload"]["model"] == CFG.llm.vision_model
    assert session.calls[0]["payload"]["messages"][0]["content"][1]["type"] == "image_url"
