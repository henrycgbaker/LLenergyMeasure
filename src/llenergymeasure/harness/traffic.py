"""Open-loop traffic generation for server-mode measurement.

This module owns the ``TrafficSource`` seam - the single window-manager-facing
interface for driving online-serving load - and the built-in async Poisson/gamma
issuer that implements it. It is the load-generation half of server mode; the
window manager, warmup, and engine server lifecycle live in later slices and
consume this seam rather than re-implementing it.

Semantic cluster (stated explicitly, per the loop-semantics research). Four
conventions exist in the wild - MLPerf LoadGen (schedule-anchored latency),
vllm-bench / ML.ENERGY (admission-anchored, cap-queue time invisible),
perf_analyzer (mutually-exclusive rate/concurrency axes), and k6
(drop-on-exhaustion). LLenergyMeasure adopts the **MLPerf schedule-anchored**
convention:

- The issuance schedule is precomputed from the arrival process and the seed,
  independent of how the system-under-test progresses (OPEN-LOOP). A stalled
  transport never slows issuance - the failure mode that hides stalls from tail
  percentiles (coordinated omission; Tene / wrk2).
- Each request records three timestamps: ``issued_at`` (its ideal scheduled
  time, the LATENCY ANCHOR), ``dispatched_at`` (when it was actually handed to
  the transport, after any concurrency-cap wait), and ``completed_at``.
  Percentiles are anchored at ``issued_at`` so cap-induced queue time counts
  against the SUT.
- ``concurrency_cap`` gates DISPATCH of already-issued requests; it never slows
  the schedule (the loop is semi-open under a cap, pure-open without one).
- A materially binding cap is DISCLOSED, not silently absorbed: the report
  carries ``cap_bound_fraction`` (the fraction of requests whose dispatch was
  delayed beyond a small tolerance) for result provenance. MLPerf fails such a
  run outright; LLenergyMeasure keeps the cap legal (it is a deliberate,
  hashed user choice) but stamps its effect.

Design source: ``.product/designs/server-mode-implementation-plan-2026-07-23.md``
section 12 (as amended 2026-07-27); evidence base in
``.product/research/wave2-sut-and-loop-semantics-2026-07-27/``.

The real HTTP transport (:class:`HttpxTransport`) lazily imports ``httpx`` from
the ``server`` extra; the seam itself imports nothing beyond the standard
library and numpy, so the window manager can consume it without the extra
installed. Later slices inject a base-URL-bound transport; conformance tests
inject fakes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from llenergymeasure.config.models import TrafficConfig

#: Dispatch delay (seconds) above which a request counts as cap-bound. Absorbs
#: trivial event-loop scheduling jitter so ``cap_bound_fraction`` reflects real
#: queueing behind the concurrency cap, not sub-millisecond task hand-off.
_CAP_BOUND_TOLERANCE_S = 0.010


@dataclass(frozen=True)
class RequestShape:
    """One request's payload, supplied by the injected request-shape source.

    Deliberately opaque at this layer: SM5 only needs an index and an optional
    payload to hand to the transport. The per-engine request encoding (prompt,
    token budget, sampling params) is owned by later slices and rides
    ``payload``.
    """

    index: int
    payload: Any = None


#: A request-shape source maps a request index to its shape. Built-in default
#: mints index-only shapes; later slices inject a dataset-backed source.
ShapeSource = Callable[[int], RequestShape]


@runtime_checkable
class Transport(Protocol):
    """Async sink for a single issued request (injected, never owned here).

    The issuer is transport-agnostic: production wires an HTTP transport bound to
    the engine server's base URL, conformance tests inject fakes. The return
    value is recorded as bookkeeping only and never influences the schedule.
    """

    async def __call__(self, request: RequestShape) -> Any: ...


@dataclass
class RequestRecord:
    """Per-request bookkeeping. Timestamps share a ``time.monotonic`` basis.

    ``issued_at`` is the ideal scheduled time (the latency anchor), not the
    wall-clock instant the loop reached the request. ``dispatched_at`` is when
    the transport was actually invoked (after any cap wait); ``completed_at`` is
    when it returned. ``dispatched_at`` / ``completed_at`` stay ``None`` when a
    request is cancelled before dispatch or before completion (e.g. a drain
    timeout under a stalled transport).
    """

    index: int
    issued_at: float
    request: RequestShape
    dispatched_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: BaseException | None = None

    @property
    def dispatch_delay_s(self) -> float | None:
        """``dispatched_at - issued_at`` (cap + scheduling delay), or None."""
        if self.dispatched_at is None:
            return None
        return self.dispatched_at - self.issued_at


@dataclass(frozen=True)
class ArrivalSchedule:
    """A precomputed open-loop issuance schedule.

    ``offsets`` are seconds from the run origin (cumulative inter-arrival times),
    strictly the arrival process - independent of any transport progress.
    """

    offsets: tuple[float, ...]
    rate: float
    arrival: str

    @property
    def count(self) -> int:
        return len(self.offsets)

    def interarrivals(self) -> npt.NDArray[np.float64]:
        """Inter-arrival gaps (the first gap is ``offsets[0]``)."""
        return np.diff(np.asarray(self.offsets, dtype=np.float64), prepend=0.0)

    def coefficient_of_variation(self) -> float:
        """CV (std / mean) of the inter-arrival gaps. ~1 for Poisson."""
        gaps = self.interarrivals()
        mean = float(gaps.mean())
        if mean <= 0.0:
            return float("inf")
        return float(gaps.std() / mean)


@dataclass(frozen=True)
class IssuerReport:
    """Outcome of an issuer run - the source's summary / bookkeeping surface.

    Attributes:
        records: Per-request bookkeeping in issue order.
        issued_count: Requests placed on the schedule and issued.
        completed_count: Requests whose transport call returned.
        cap_bound_fraction: Fraction of issued requests whose dispatch was
            delayed beyond ``_CAP_BOUND_TOLERANCE_S`` (or never dispatched)
            because of the concurrency cap. 0.0 when uncapped. Provenance for
            the cap-binding validity disclosure.
        issuance_duration_s: Wall-clock span of the issue loop. Tracks the
            schedule span when the loop is non-blocking; a stalled transport
            does NOT inflate it (the open-loop guarantee, measured).
        concurrency_cap: The cap in force (None = uncapped).
    """

    records: list[RequestRecord]
    issued_count: int
    completed_count: int
    cap_bound_fraction: float
    issuance_duration_s: float
    concurrency_cap: int | None


def _default_shape_source(index: int) -> RequestShape:
    return RequestShape(index=index)


def _draw_interarrivals(
    rng: np.random.Generator, config: TrafficConfig, count: int
) -> npt.NDArray[np.float64]:
    """Draw ``count`` inter-arrival gaps for the configured arrival process.

    Poisson: exponential gaps at rate lambda (CV = 1). Gamma: shape / scale
    chosen so the mean is ``1 / rate`` and the CV equals ``burstiness`` (CV = 1
    reproduces Poisson, > 1 is burstier, < 1 smoother).
    """
    rate = config.rate
    if config.arrival == "gamma":
        cv = config.burstiness if config.burstiness is not None else 1.0
        shape = 1.0 / (cv * cv)
        scale = (cv * cv) / rate
        return rng.gamma(shape, scale, size=count)
    return rng.exponential(1.0 / rate, size=count)


def build_schedule(
    config: TrafficConfig, *, seed: int | None = None, count: int | None = None
) -> ArrivalSchedule:
    """Build the deterministic open-loop issuance schedule for ``config``.

    The schedule depends only on ``(rate, arrival, burstiness, seed)`` and the
    resolved length - never on transport progress - so the same inputs always
    yield an identical schedule (the reproducibility guarantee that rides the
    hashed traffic identity).

    Length resolution: ``count`` overrides everything (used by conformance
    tests that want a large sample). Otherwise ``window_requests`` fixes the
    length; a ``window_seconds`` window draws gaps until the cumulative offset
    reaches the duration.
    """
    resolved_seed = seed if seed is not None else config.seed
    rng = np.random.default_rng(resolved_seed)

    # ``count`` overrides everything; otherwise ``window_requests`` fixes the
    # length (None falls through to the duration-bounded ``window_seconds`` path).
    fixed_count = count if count is not None else config.window_requests
    if fixed_count is not None:
        gaps = _draw_interarrivals(rng, config, fixed_count)
        offsets = np.cumsum(gaps)
    else:
        # Duration-bounded (window_seconds): draw in deterministic batches until
        # the cumulative schedule covers the window, then truncate.
        assert config.window_seconds is not None  # enforced by TrafficConfig
        horizon = config.window_seconds
        batch = max(16, int(config.rate * horizon * 1.2) + 16)
        collected: list[npt.NDArray[np.float64]] = []
        total = 0.0
        while total < horizon:
            gaps = _draw_interarrivals(rng, config, batch)
            collected.append(gaps)
            total += float(gaps.sum())
        all_offsets = np.cumsum(np.concatenate(collected))
        offsets = all_offsets[all_offsets <= horizon]

    return ArrivalSchedule(
        offsets=tuple(float(x) for x in offsets),
        rate=config.rate,
        arrival=config.arrival,
    )


@runtime_checkable
class TrafficSource(Protocol):
    """The single window-manager-facing surface for driving online load.

    A traffic source is built from ``(TrafficConfig, seed, request-shape
    source)`` and driven by exactly one call - :meth:`run` - against an injected
    transport. Completions are bookkeeping only and never influence the
    schedule. This is the D14 / O2 plugin point: a LoadGen-backed source can
    arrive later behind the same interface.
    """

    async def run(
        self, transport: Transport, *, drain_timeout: float | None = None
    ) -> IssuerReport: ...


class OpenLoopPoissonSource:
    """Built-in async open-loop Poisson / gamma issuer (implements TrafficSource).

    Precomputes the schedule at construction, then :meth:`run` iterates it in
    real time, firing each request at its scheduled instant as a detached task.
    The issue loop never awaits a transport call, so a stall or a binding
    concurrency cap cannot slow issuance. The cap, when set, gates entry into
    the transport call INSIDE each detached task (never the loop).
    """

    def __init__(
        self,
        config: TrafficConfig,
        *,
        seed: int | None = None,
        shape_source: ShapeSource | None = None,
    ) -> None:
        self._config = config
        self._seed = seed if seed is not None else config.seed
        self._shape_source = shape_source if shape_source is not None else _default_shape_source
        self._schedule = build_schedule(config, seed=self._seed)

    @property
    def schedule(self) -> ArrivalSchedule:
        return self._schedule

    async def run(
        self, transport: Transport, *, drain_timeout: float | None = None
    ) -> IssuerReport:
        """Issue the whole schedule against ``transport`` and report bookkeeping.

        The schedule is issued open-loop and non-blocking; after issuance the
        outstanding dispatch tasks are drained. ``drain_timeout`` bounds that
        drain (``None`` waits for every dispatch to finish); requests still
        pending at the timeout are cancelled and left with ``completed_at``
        unset so a stalled transport cannot hang the run.
        """
        cap = self._config.concurrency_cap
        semaphore = asyncio.Semaphore(cap) if cap is not None else None
        records: list[RequestRecord] = []
        tasks: list[asyncio.Task[None]] = []

        origin = time.monotonic()
        for i, offset in enumerate(self._schedule.offsets):
            delay = (origin + offset) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            record = RequestRecord(
                index=i,
                issued_at=origin + offset,  # ideal scheduled time = latency anchor
                request=self._shape_source(i),
            )
            records.append(record)
            tasks.append(asyncio.create_task(self._dispatch(record, transport, semaphore)))
        issuance_duration = time.monotonic() - origin

        await self._drain(tasks, drain_timeout)
        return self._build_report(records, issuance_duration)

    async def _dispatch(
        self,
        record: RequestRecord,
        transport: Transport,
        semaphore: asyncio.Semaphore | None,
    ) -> None:
        """Dispatch one already-issued request, gated by the cap when present."""
        if semaphore is not None:
            async with semaphore:
                await self._invoke(record, transport)
        else:
            await self._invoke(record, transport)

    async def _invoke(self, record: RequestRecord, transport: Transport) -> None:
        record.dispatched_at = time.monotonic()
        try:
            record.result = await transport(record.request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # bookkeeping only; a failed call never stops issuance
            record.error = exc
        record.completed_at = time.monotonic()

    @staticmethod
    async def _drain(tasks: list[asyncio.Task[None]], drain_timeout: float | None) -> None:
        if not tasks:
            return
        if drain_timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
            return
        _, pending = await asyncio.wait(tasks, timeout=drain_timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _build_report(self, records: list[RequestRecord], issuance_duration: float) -> IssuerReport:
        cap = self._config.concurrency_cap
        completed = sum(1 for r in records if r.completed_at is not None)
        if cap is None:
            cap_bound_fraction = 0.0
        else:
            bound = sum(1 for r in records if self._is_cap_bound(r))
            cap_bound_fraction = bound / len(records) if records else 0.0
        return IssuerReport(
            records=records,
            issued_count=len(records),
            completed_count=completed,
            cap_bound_fraction=cap_bound_fraction,
            issuance_duration_s=issuance_duration,
            concurrency_cap=cap,
        )

    @staticmethod
    def _is_cap_bound(record: RequestRecord) -> bool:
        delay = record.dispatch_delay_s
        if delay is None:
            return True  # never dispatched: the cap held it back for the whole run
        return delay > _CAP_BOUND_TOLERANCE_S


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
    counted identically for every OpenAI-compatible engine in this callback (O8).
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
    """

    text: str
    output_token_times: list[float]
    first_token_at: float | None
    server_prompt_tokens: int | None
    server_completion_tokens: int | None
    finish_reason: str | None = None


@dataclass
class HttpxTransport:
    """Production streaming HTTP transport for the issuer (the ``httpx`` use site).

    Lazily imports ``httpx`` (the ``server`` extra) and holds an async client
    bound to the engine server's ``base_url``. The server session sets
    ``base_url`` from the launched server and ``path`` to the engine's
    OpenAI-compatible serving endpoint (e.g. ``/v1/completions``), with each
    request's ``payload`` the JSON body.

    Each call POSTs the payload with ``stream: true`` and counts the streamed
    response deltas CLIENT-SIDE (O8): one output-token receipt timestamp per
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

        return CompletionResult(
            text="".join(text_parts),
            output_token_times=token_times,
            first_token_at=first_token_at,
            server_prompt_tokens=prompt_tokens,
            server_completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

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
