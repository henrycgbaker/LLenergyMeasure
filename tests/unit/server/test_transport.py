"""Unit tests for the serving HTTP transport (host-only, no server, no GPU).

Covers the three things the transport is responsible for: the lazy ``httpx``
import behind the ``server`` extra, client-side streaming token counting, and
preserving a partially streamed completion when the stream aborts mid-flight.

Also pins the structural conformance the layering depends on: the issuer's
``Transport`` seam lives one layer above, so :class:`HttpxTransport` must satisfy
it WITHOUT importing or inheriting from it. The assignment in
:func:`_as_transport` is the type-checked half of that assertion; the isinstance
check is the runtime half.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from llenergymeasure.harness.traffic import Transport
from llenergymeasure.serving.transport import (
    PARTIAL_COMPLETION_ATTR,
    CompletionResult,
    HttpxTransport,
    RequestShape,
    require_httpx,
)

# ---------------------------------------------------------------------------
# Structural conformance to the issuer's Transport seam (no upward import)
# ---------------------------------------------------------------------------


def _as_transport(transport: HttpxTransport) -> Transport:
    """Type-checked assertion that HttpxTransport satisfies Transport structurally.

    The type checker rejects this return the moment the two call signatures
    diverge, which is the only thing holding the seam together: the transport
    cannot import the protocol (it sits below it), so nothing else would catch a
    drift.
    """
    return transport


def test_httpx_transport_satisfies_the_transport_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructed transport is accepted as a Transport at runtime too."""
    monkeypatch.setattr(
        "llenergymeasure.serving.transport.require_httpx",
        lambda: SimpleNamespace(AsyncClient=lambda **kw: _FakeAsyncClient([], [], **kw)),
    )
    transport = HttpxTransport(base_url="http://x", path="/v1/completions")
    assert isinstance(_as_transport(transport), Transport)


# ---------------------------------------------------------------------------
# [server] extra: lazy httpx import with an actionable error
# ---------------------------------------------------------------------------


def test_require_httpx_names_the_server_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """When httpx is absent, the error names the extra to install."""
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError) as excinfo:
        require_httpx()
    message = str(excinfo.value)
    assert "httpx" in message
    assert "llenergymeasure[server]" in message


def test_httpx_transport_construction_requires_server_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HttpxTransport is the lazy-import use site: constructing it without httpx errors."""
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError) as excinfo:
        HttpxTransport(base_url="http://localhost:8000")
    assert "llenergymeasure[server]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Client-side streaming token counting - a fake httpx stream
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _FakeStreamResponse:
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeAsyncClient:
    """Minimal fake of httpx.AsyncClient.stream over canned SSE lines."""

    def __init__(self, lines: list[str], payloads: list[dict], **_kw: object) -> None:
        self._lines = lines
        self._payloads = payloads

    def stream(self, method: str, path: str, *, json: dict) -> _FakeStreamCtx:
        self._payloads.append(json)
        return _FakeStreamCtx(self._lines)

    async def aclose(self) -> None:
        return None


def test_httpx_transport_streams_and_counts_client_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transport counts streamed deltas client-side; server usage is auxiliary."""
    lines = [
        'data: {"choices":[{"text":"Hel","finish_reason":null}]}',
        'data: {"choices":[{"text":"lo","finish_reason":null}]}',
        # An empty finish delta is a step with no text - not a counted token.
        'data: {"choices":[{"text":"","finish_reason":"length"}]}',
        # A usage-only terminal chunk (choices empty) carries the auxiliary count.
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    payloads: list[dict] = []
    fake_httpx = SimpleNamespace(AsyncClient=lambda **kw: _FakeAsyncClient(lines, payloads, **kw))
    monkeypatch.setattr("llenergymeasure.serving.transport.require_httpx", lambda: fake_httpx)

    transport = HttpxTransport(base_url="http://x", path="/v1/completions")
    result = asyncio.run(
        transport(RequestShape(index=0, payload={"model": "gpt2", "prompt": "hi"}))
    )

    assert isinstance(result, CompletionResult)
    # Two non-empty deltas -> two client-counted tokens (the empty finish delta is not one).
    assert len(result.output_token_times) == 2
    assert result.text == "Hello"
    assert result.first_token_at == result.output_token_times[0]
    assert result.finish_reason == "length"
    # The engine's self-reported usage rides as auxiliary provenance only.
    assert result.server_prompt_tokens == 4
    assert result.server_completion_tokens == 2
    # Streaming was requested with the usage option, and the original payload is preserved.
    assert payloads[0]["stream"] is True
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert payloads[0]["prompt"] == "hi"


class _FailingStreamResponse:
    def __init__(self, lines: list[str], fail_after: int) -> None:
        self._lines = lines
        self._fail_after = fail_after

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for i, line in enumerate(self._lines):
            if i == self._fail_after:
                raise ConnectionError("stream reset mid-response")
            yield line


class _FailingStreamCtx:
    def __init__(self, lines: list[str], fail_after: int) -> None:
        self._lines = lines
        self._fail_after = fail_after

    async def __aenter__(self) -> _FailingStreamResponse:
        return _FailingStreamResponse(self._lines, self._fail_after)

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FailingAsyncClient:
    def __init__(self, lines: list[str], fail_after: int, **_kw: object) -> None:
        self._lines = lines
        self._fail_after = fail_after

    def stream(self, method: str, path: str, *, json: dict) -> _FailingStreamCtx:
        return _FailingStreamCtx(self._lines, self._fail_after)

    async def aclose(self) -> None:
        return None


def test_httpx_transport_preserves_partial_on_mid_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that dies mid-response attaches the tokens delivered so far."""
    lines = [
        'data: {"choices":[{"text":"Hel","finish_reason":null}]}',
        'data: {"choices":[{"text":"lo","finish_reason":null}]}',
        # The reset strikes here, before any finish / usage / [DONE] chunk.
        'data: {"choices":[{"text":"!","finish_reason":null}]}',
    ]
    fake_httpx = SimpleNamespace(AsyncClient=lambda **kw: _FailingAsyncClient(lines, 2, **kw))
    monkeypatch.setattr("llenergymeasure.serving.transport.require_httpx", lambda: fake_httpx)

    transport = HttpxTransport(base_url="http://x", path="/v1/completions")
    with pytest.raises(ConnectionError) as excinfo:
        asyncio.run(transport(RequestShape(index=0, payload={"prompt": "hi"})))

    partial = getattr(excinfo.value, PARTIAL_COMPLETION_ATTR)
    assert isinstance(partial, CompletionResult)
    # The two deltas delivered before the reset are preserved (not discarded),
    # so their in-span tokens can still count toward the energy denominator.
    assert len(partial.output_token_times) == 2
    assert partial.text == "Hello"
    assert partial.finish_reason is None  # the stream never finished
