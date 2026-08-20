"""HTTP transport for online-serving traffic: one request out, one completion back.

The wire half of the serving layer. It owns what goes onto the wire for a single
request (:class:`RequestShape`), what comes back off it
(:class:`CompletionResult`), and the production streaming client that turns one
into the other (:class:`HttpxTransport`).

The transport is a plain callable, deliberately owning no schedule and no
concurrency policy: those belong to the load issuer above it, which injects a
base-URL-bound transport and records whatever the call returns. The issuer's
``Transport`` seam is satisfied STRUCTURALLY - :class:`HttpxTransport` matches
its call signature without inheriting from or importing anything above this
layer, which is what lets a lower layer supply a higher layer's dependency
without an upward import. Conformance tests inject fakes in its place.

``httpx`` ships only in the optional ``server`` extra and is imported lazily at
the use site (:func:`require_httpx`), so importing this module costs nothing on
a host that never serves.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PARTIAL_COMPLETION_ATTR",
    "CompletionResult",
    "HttpxTransport",
    "RequestShape",
    "require_httpx",
]


@dataclass(frozen=True)
class RequestShape:
    """One request's payload, supplied by the injected request-shape source.

    Deliberately opaque here: the transport only needs an index and an optional
    payload to put on the wire, and the issuer above only needs to carry it. The
    per-engine request encoding (prompt, token budget, sampling params) is owned
    by later work and rides ``payload``.
    """

    index: int
    payload: Any = None


def require_httpx() -> Any:
    """Import ``httpx`` or raise an actionable error naming the ``server`` extra.

    ``httpx`` is a pure-Python client and ships only in the optional ``server``
    extra (server mode is not needed for offline measurement), so it is imported
    lazily at the use site rather than at module import.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "Server-mode traffic generation requires the 'httpx' HTTP client, which is "
            "not installed. Install the server extra: pip install 'llenergymeasure[server]'."
        ) from exc
    return httpx


@dataclass
class CompletionResult:
    """One streamed completion's client-observed facts (the transport's product).

    Stored as ``RequestRecord.result``. Timestamps share the issuer's
    ``time.monotonic`` basis. ``output_token_times`` is the CLIENT-SIDE canonical
    token receipt series - one monotonic timestamp per streamed content delta,
    counted identically for every OpenAI-compatible engine in this callback.
    Its length is the canonical output-token count that feeds the energy
    denominator and the stability gate; ``first_token_at`` is its first entry
    (None when nothing streamed). ``server_prompt_tokens`` /
    ``server_completion_tokens`` are the engine's self-reported usage block -
    AUXILIARY provenance only, None when the engine reported none (e.g. a stream
    without ``include_usage`` support), NEVER the denominator.

    The client count assumes the server streams ONE token per content delta, which
    vLLM and TRT-LLM OpenAI-compatible ``/v1/completions`` streaming does by
    default (one decode step per SSE chunk). An engine that coalesces multiple
    tokens into one delta would make the client count an under-count; the
    self-reported usage rides alongside precisely so any such divergence is
    visible per request rather than hidden as a silent approximation.

    Receipt timestamps carry client-loop jitter: each ``output_token_times`` entry
    is stamped when the async event loop resumes this reader after the bytes
    arrive, not at the socket, so under high concurrency the loop's scheduling
    delay is folded into the receipt time. TTFT and per-window aggregates absorb
    it; sub-millisecond inter-token-latency claims from consecutive receipts do
    not (a downstream consumer should treat fine-grained ITL as approximate).
    """

    text: str
    output_token_times: list[float]
    first_token_at: float | None
    server_prompt_tokens: int | None
    server_completion_tokens: int | None
    finish_reason: str | None = None


#: Attribute name under which a streaming transport attaches its partially
#: accumulated :class:`CompletionResult` to the exception that aborts a stream.
#: The AbortedLevel precedent: the exception is re-raised unchanged and the
#: issuer reads this off it, so a mid-stream failure still preserves the tokens
#: actually delivered (they count toward the in-span energy denominator).
PARTIAL_COMPLETION_ATTR = "llem_partial_completion"


@dataclass
class HttpxTransport:
    """Production streaming HTTP transport for the issuer (the ``httpx`` use site).

    Lazily imports ``httpx`` (the ``server`` extra) and holds an async client
    bound to the engine server's ``base_url``. The server session sets
    ``base_url`` from the launched server and ``path`` to the engine's
    OpenAI-compatible serving endpoint (e.g. ``/v1/completions``), with each
    request's ``payload`` the JSON body.

    Each call POSTs the payload with ``stream: true`` and counts the streamed
    response deltas CLIENT-SIDE: one output-token receipt timestamp per
    content delta, measured identically for every engine here so the J/token
    denominator is engine-agnostic. It returns a :class:`CompletionResult`
    carrying those receipts (for TTFT / ITL / the denominator) plus the engine's
    self-reported usage as auxiliary provenance. Call :meth:`aclose` to release
    the connection pool.
    """

    base_url: str
    timeout: float = 60.0
    #: Serving endpoint each request is POSTed to. Defaults to root; the server
    #: session sets it to the engine's OpenAI completions path.
    path: str = "/"
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        httpx = require_httpx()
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def __call__(self, request: RequestShape) -> CompletionResult:
        """Stream one completion and return its client-observed facts.

        Streams the OpenAI-compatible completions response and timestamps each
        content delta with ``time.monotonic`` (the issuer's clock), so the return
        carries the client-side token receipts the denominator and stability gate
        consume. The engine's self-reported ``usage`` (when it sends the final
        ``include_usage`` chunk) rides as auxiliary provenance only.
        """
        payload = dict(request.payload or {})
        payload["stream"] = True
        # Request the terminal usage chunk where the engine honours it (vLLM);
        # engines that ignore it simply never send usage and the auxiliary fields
        # stay None - the client-side delta count is the denominator regardless.
        payload["stream_options"] = {"include_usage": True}

        token_times: list[float] = []
        text_parts: list[str] = []
        first_token_at: float | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        finish_reason: str | None = None

        def snapshot() -> CompletionResult:
            # Built from the mutable accumulators, so it reflects whatever streamed
            # up to this point - the clean return AND the mid-stream-failure partial.
            return CompletionResult(
                text="".join(text_parts),
                output_token_times=token_times,
                first_token_at=first_token_at,
                server_prompt_tokens=prompt_tokens,
                server_completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )

        try:
            async with self._client.stream("POST", self.path, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    data = self._sse_data(line)
                    if data is None:
                        continue
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices") or []
                    if choices:
                        choice = choices[0]
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                        # Completions API streams the incremental text under "text".
                        delta = choice.get("text") or ""
                        if delta:
                            now = time.monotonic()
                            if first_token_at is None:
                                first_token_at = now
                            token_times.append(now)
                            text_parts.append(delta)
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = _usage_int(usage.get("prompt_tokens"), prompt_tokens)
                        completion_tokens = _usage_int(
                            usage.get("completion_tokens"), completion_tokens
                        )
        except BaseException as exc:
            # A connection reset, read-timeout between deltas, malformed chunk, or
            # a drain cancellation must not discard the tokens already delivered:
            # attach the partial and re-raise unchanged (both Exception and
            # BaseException/cancellation paths). The issuer stashes it on the record.
            setattr(exc, PARTIAL_COMPLETION_ATTR, snapshot())
            raise

        return snapshot()

    @staticmethod
    def _sse_data(line: str) -> str | None:
        """Extract one SSE ``data:`` line's payload, or None for non-data lines.

        Blank keep-alive lines and non-``data:`` fields (``event:``, ``id:``,
        comments) are skipped; the returned string is a chunk's JSON or the
        ``[DONE]`` sentinel.
        """
        if not line or not line.startswith("data:"):
            return None
        return line[len("data:") :].strip()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def _usage_int(value: Any, current: int | None) -> int | None:
    """Coerce a usage-block token count to int, keeping ``current`` when unusable.

    Guards the auxiliary usage fields against a missing / null / bool / non-int
    value (bool is an int subclass, so it is rejected explicitly).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return current
    return int(value)
