"""Server-mode measurement session - the C1/C3 sibling of the offline sessions.

A :class:`ServerSession` is one online-serving measurement session, the
one-dispatch:N-results consumer the F6 ``ExperimentSession`` seam was built for
(constraint C3). Its context-manager lifetime mirrors the offline
``SubprocessSession`` / ``DockerSession``:

- ``__enter__`` LAUNCHES the engine server (SM6 ``ServerCapable.launch``) and
  drives it to READY (``await_ready`` with a real probe request whose SHAPE is
  drawn from the measured traffic distribution, SM8 ``build_probe_request``),
  then marks the experiment running. A failure DURING acquisition releases the
  partially-launched server and re-raises, so a failed launch never leaks.
- ``run()`` drives the SM7 :class:`WindowManager` over the level(s) the session
  was built for, per level: the SM8 ``ServerWarmup`` hook runs once, the ramp is
  excluded prospectively, ``windows_per_level`` contiguous measured windows
  produce one result each, and the traffic drains for latency. It returns a
  :class:`ServerSessionResult` carrying the N window results (one per window)
  with each level's warmup outcome stamped in - the "N results over one lifetime"
  shape (C3). A rate sweep maps to multiple levels (C4); v0.7 feeds one level per
  dispatch (session grouping of a rate sweep is SM10, not foreclosed here).
- ``__exit__`` SHUTS the server down (idempotent, leak-free) and runs the
  drain-finalize, exactly once on the normal, SIGINT, and exception paths alike
  (the F6 session-hardening invariant).

The measurement loop runs IN-PROCESS on the host (the traffic issuer + the
host-side NVML energy/thermal sampling); only the engine SERVER runs
out-of-process (a sibling container or a host subprocess). No serialized config
crosses a process boundary for the measurement, so the R7W resolved-warmup
side-channel (an ``ExperimentConfig`` PrivateAttr that JSON would drop) is read
directly in-process via ``resolved_server_warmup()`` - the SM9/SM12 serialization
contract holds structurally, and the server itself never consumes the resolved
warmup (warmup traffic is host-driven).

Contract notes satisfied here (banked from SM7/SM8/R7W delivery):

1. ISSUANCE HORIZON: the level's traffic source issues across the whole level -
   ``ramp_exclusion + windows_per_level * window_seconds`` - as one continuous
   run (the manager owns window timing and never resizes the schedule).
2. TOKEN RECEIPTS: client-side counts flow through the SM7 ``TokenReceiptFn``
   seam so the energy denominator and the stability gate receive counts. The
   canonical count is llem's OWN count of the streamed response deltas
   (``client_token_receipts`` reading each transport ``CompletionResult``),
   measured identically for every engine (O8); the engine's self-reported usage
   rides as auxiliary provenance in ``requests.parquet`` and the per-window
   provenance, never as the denominator. The mechanism is disclosed in
   ``token_counting`` (``TOKEN_COUNTING_CLIENT_STREAMED``).
3. ABORTED LEVEL: a level failure carries its partial state on the propagating
   exception as ``llem_aborted_level``; the driver catches it at the DIRECT
   ``run_level`` await site (an outer Task boundary would substitute a fresh
   CancelledError and lose it). A ``WarmupTrafficError`` ABORTS the session (the
   transport is dead, dooming later levels); any other level failure is recorded
   invalid-with-reason and the session continues to the next level.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llenergymeasure.domain.experiment import (
    TOKEN_COUNTING_CLIENT_STREAMED,
    ServerWarmupProvenance,
    ServerWindowProvenance,
)
from llenergymeasure.harness.server_warmup import (
    ServerWarmup,
    ServerWarmupResult,
    WarmupTrafficError,
    build_probe_request,
    describe_server_warmup_protocol,
)
from llenergymeasure.harness.traffic import CompletionResult, OpenLoopPoissonSource, RequestShape
from llenergymeasure.harness.window_manager import (
    ABORTED_LEVEL_ATTR,
    ATTRIBUTION_STEADY_STATE_SPAN,
    AbortedLevel,
    BracketEnergySink,
    LevelOutcome,
    LevelPlan,
    LevelValidation,
    WarmupContext,
    WindowManager,
    WindowRecord,
    WindowSpec,
)
from llenergymeasure.results.request_log import (
    REQUEST_STATUS_ERROR,
    REQUEST_STATUS_OK,
    REQUEST_STATUS_TIMEOUT,
    RequestLogRow,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, TrafficConfig
    from llenergymeasure.config.runner_spec import RunnerSpec
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.experiment import ExperimentResult
    from llenergymeasure.domain.session import SessionBlock
    from llenergymeasure.engines.protocol import ServerCapable
    from llenergymeasure.harness.bracket import MeasuredWindowCore, MeasurementBracket
    from llenergymeasure.harness.traffic import RequestRecord, ShapeSource, TrafficSource, Transport
    from llenergymeasure.infra.server_lifecycle import ServerHandle, ServerPlacement
    from llenergymeasure.results.bundle import BundleWriter
    from llenergymeasure.study.runner import StudyRunner

logger = logging.getLogger(__name__)

#: The OpenAI-compatible completions endpoint every v0.7 server engine exposes.
#: vLLM (``/v1/completions``) and TRT-LLM (``/v1/completions``) agree, and
#: transformers is E5-gated out of server mode, so a session-layer constant is
#: the minimal faithful encoding; a future non-OpenAI engine would promote this
#: to a ``ServerCapable`` member (flagged).
SERVING_COMPLETIONS_PATH = "/v1/completions"

#: Interrupt-watcher poll cadence (seconds): how promptly a mid-session SIGINT
#: (which the study runner's handler records on its interrupt event) is noticed
#: and cancels the driving task.
_INTERRUPT_POLL_INTERVAL_S = 0.2

__all__ = [
    "SERVING_COMPLETIONS_PATH",
    "TOKEN_COUNTING_CLIENT_STREAMED",
    "ServerCell",
    "ServerLevelResult",
    "ServerSession",
    "ServerSessionError",
    "ServerSessionResult",
    "ServerWindowResult",
    "build_request_rows_by_window",
    "client_token_receipts",
    "partition_server_groups",
]


# ---------------------------------------------------------------------------
# Result types (the session's product - the N window results + provenance).
# SM10 persists these as per-window bundles; here they are the in-memory shape.
# ---------------------------------------------------------------------------


@dataclass
class ServerWindowResult:
    """One measured window's result, with the level's warmup provenance stamped in.

    The per-window unit SM10 turns into one bundle. ``warmup`` /
    ``pre_window_protocol`` are the D6 divergence label (SM12/SM14 render the
    offline-vs-server pre-window difference); they are identical across a level's
    windows (one warmup per level) but stamped per window so the persistence layer
    (which never learns sessions exist) has them locally.
    """

    level_index: int
    window: WindowRecord
    warmup: ServerWarmupResult | None
    pre_window_protocol: str


@dataclass
class ServerLevelResult:
    """One rate level's product: its windows + steady-state verdict + warmup outcome.

    ``validation`` is the SM7 window-to-window J/token gate verdict, or ``None``
    when the level failed before it could run (aborted / warmup-failed).
    ``invalid_reason`` names the failure site when the level did not complete
    cleanly (never dropped - recorded invalid-with-reason, contract 3).
    ``completed_cores`` holds the RAW measured cores of a failed level's
    cleanly-closed windows (O7.4): full per-window bookkeeping was lost with the
    abort, so no synthetic ``ServerWindowResult`` is built, but the measured GPU
    energy is preserved and counted in the session total. SM10 decides their
    bundle fate (drain-fields-null semantics).
    """

    level_index: int
    spec: WindowSpec | None
    windows: list[ServerWindowResult]
    validation: LevelValidation | None
    warmup: ServerWarmupResult | None
    invalid_reason: str | None
    aborted_window_index: int | None = None
    completed_cores: list[MeasuredWindowCore | None] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None and self.validation is not None and self.validation.valid


@dataclass
class ServerSessionResult:
    """One server session's product: the N window results over one server lifetime.

    The session's internal return type. ``valid`` is True iff at least one level
    passed its stability gate. ``token_counting`` names the client-side counting
    mechanism whose count is the J/token denominator (``TOKEN_COUNTING_CLIENT_STREAMED``,
    O8). ``experiment_results`` / ``result_files`` are the per-window bundles the
    session persisted (SM10): the mapped ExperimentResults that enter
    ``StudyResult.experiments`` and their on-disk paths - orchestration consumes
    these (there is no experiments=None side-channel any more).
    """

    engine: str
    config_hash: str
    cycle: int
    index: int
    serving_mode: str
    levels: list[ServerLevelResult]
    token_counting: str
    total_window_energy_j: float | None
    elapsed_s: float
    aborted: bool
    abort_reason: str | None
    experiment_results: list[ExperimentResult] = field(default_factory=list)
    result_files: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return any(level.valid for level in self.levels)

    @property
    def window_count(self) -> int:
        return sum(len(level.windows) for level in self.levels)


# ---------------------------------------------------------------------------
# Token-receipt seam wiring (contract 2, O8). The CANONICAL denominator is the
# client-side count of the streamed response deltas the transport captured on
# each RequestRecord.result (a CompletionResult), one receipt timestamp per
# streamed token. Token-granular, so the window manager counts tokens RECEIVED
# within the measured span (the ratified energy-denominator rule) and the k=4
# sub-window J/token diagnostic is formable. The engine's self-reported usage is
# auxiliary provenance only (it rides in requests.parquet), never the denominator.
# ---------------------------------------------------------------------------


def client_token_receipts(record: RequestRecord) -> Sequence[float]:
    """Return a request's client-counted output-token receipt timestamps (O8).

    The monotonic receipt time of each streamed content delta, as the transport
    counted them in its own callback (identical across engines). Receipts are
    returned for a request whose ``result`` carries them REGARDLESS of terminal
    status (H1): a request that streamed tokens in-span then died still delivered
    that compute, so its tokens must count in the denominator (physics: J/token is
    energy over tokens DELIVERED in-span). The transport preserves the partial
    receipts of a mid-stream failure on the record, so error / timeout requests
    contribute their real delivered tokens. The window manager clips receipts to
    ``[span_start, span_end]``, so post-span drain-tail tokens are still excluded.
    Returns ``()`` only when nothing streamed (no CompletionResult, or an empty
    one) - the gate then degrades to invalid-with-reason rather than lying.
    """
    result = record.result
    if not isinstance(result, CompletionResult):
        return ()
    return tuple(result.output_token_times)


def _request_status(record: RequestRecord) -> str:
    """Classify a request's outcome for the request log (ok / error / timeout)."""
    if record.error is not None:
        return REQUEST_STATUS_ERROR
    if record.completed_at is None:
        return REQUEST_STATUS_TIMEOUT
    return REQUEST_STATUS_OK


def _attribute_window(issued_at: float, span_ends: Sequence[float]) -> int:
    """Index of the window that owns a request issued at ``issued_at`` (by span_end).

    The first window whose measured span closes at or after the issue time owns it;
    a request issued past the last span (schedule overrun; normally none) falls to
    the last window. This partitions the level timeline so every record lands in
    exactly one window - none is dropped.
    """
    for i, end in enumerate(span_ends):
        if issued_at <= end:
            return i
    return len(span_ends) - 1


def _build_request_row(
    record: RequestRecord, *, window: WindowRecord, level_index: int, ramp_boundary: float
) -> RequestLogRow:
    """Build one request-log row from a request record and its owning window (D7).

    Raw-record discipline: every column states its physical fact when it exists and
    is null only when it does not - the row does not filter or judge by status
    (that is SM12's consumer-side job). The client receipt series
    (``output_token_times`` / ``client_output_tokens``) is carried for ALL statuses
    (H1: a mid-stream failure still delivered those tokens, preserved on the
    record). ``first_token_at`` / ``ttft_ms`` are the real first-token receipt and
    latency whenever a token physically arrived (so an error / timeout row that
    streamed keeps them, matching ``output_token_times[0]``); ``finish_reason`` is
    the real terminal reason when a finish chunk physically arrived, else null (a
    mid-stream death never finished). ``completed_at`` / ``e2e_latency_ms`` are the
    time-to-terminal: an error row carries its to-failure latency, a timeout row
    (no completion) leaves them null.
    """
    boundaries = window.boundaries
    completion = record.result if isinstance(record.result, CompletionResult) else None
    token_times = list(completion.output_token_times) if completion is not None else []
    completed_at = record.completed_at
    issued_at = record.issued_at
    in_window = boundaries.span_start <= issued_at <= boundaries.span_end
    # Raw physical facts, no status gate: first_token_at is real whenever a token
    # arrived (it IS output_token_times[0]); finish_reason is real only when a
    # finish chunk arrived (null on a mid-stream death). SM12 filters by status.
    first_token_at = completion.first_token_at if completion is not None else None
    finish_reason = completion.finish_reason if completion is not None else None
    return RequestLogRow(
        request_index=record.index,
        issued_at=issued_at,
        dispatched_at=record.dispatched_at,
        first_token_at=first_token_at,
        completed_at=completed_at,
        ttft_ms=((first_token_at - issued_at) * 1000.0) if first_token_at is not None else None,
        e2e_latency_ms=((completed_at - issued_at) * 1000.0) if completed_at is not None else None,
        client_output_tokens=len(token_times),
        server_prompt_tokens=completion.server_prompt_tokens if completion is not None else None,
        server_completion_tokens=(
            completion.server_completion_tokens if completion is not None else None
        ),
        status=_request_status(record),
        finish_reason=finish_reason,
        level_index=level_index,
        window_index=window.window_index,
        in_measurement_window=in_window,
        # Only window 0 carries the level's prospective ramp (issued before its span).
        is_ramp=issued_at < ramp_boundary,
        # D7 straddler: issued in-span, completing in the post-span drain tail.
        completed_in_drain=(
            in_window and completed_at is not None and completed_at > boundaries.span_end
        ),
        output_token_times=token_times,
    )


def build_request_rows_by_window(
    records: Sequence[RequestRecord],
    windows: Sequence[WindowRecord],
    *,
    level_index: int,
) -> list[list[RequestLogRow]]:
    """Partition a level's issued requests into per-window request-log rows (D7).

    Returns one row list per window, aligned to ``windows`` order. Every record is
    attributed to EXACTLY ONE window by issue time so none is dropped: window 0
    owns everything issued up to its span_end (so the level's prospective ramp
    requests ride in window 0 flagged ``is_ramp``); window ``i`` owns records
    issued after window ``i-1``'s span_end through its own span_end. Per-row flags
    follow the D7 boundary policies: ``in_measurement_window`` (issued in the
    window's [span_start, span_end]), ``is_ramp`` (issued before window 0's
    span_start), and ``completed_in_drain`` (issued in-span, completing past
    span_end). Timestamps are the issuer's ``time.monotonic`` basis (SM12 derives
    every latency as a difference).
    """
    if not windows:
        return []
    span_ends = [w.boundaries.span_end for w in windows]
    ramp_boundary = windows[0].boundaries.span_start
    row_lists: list[list[RequestLogRow]] = [[] for _ in windows]
    for record in records:
        w_idx = _attribute_window(record.issued_at, span_ends)
        row_lists[w_idx].append(
            _build_request_row(
                record,
                window=windows[w_idx],
                level_index=level_index,
                ramp_boundary=ramp_boundary,
            )
        )
    return row_lists


def _sum_server_completion_tokens(rows: Sequence[RequestLogRow]) -> int | None:
    """Sum the auxiliary server-reported completion tokens over a window's rows.

    None when no row carried a server usage count (never a false 0), so the
    provenance field distinguishes 'no engine reported usage' from 'zero tokens'.
    """
    total = 0
    seen = False
    for row in rows:
        if row.server_completion_tokens is not None:
            total += row.server_completion_tokens
            seen = True
    return total if seen else None


# ---------------------------------------------------------------------------
# Request-shape source: OpenAI completions payloads drawn from the config's
# dataset prompts (the minimal faithful shape so warmup + measurement + probe all
# drive the path they measure; the streaming flags + client-side token counting
# are added by the transport, SM11, not the shape).
# ---------------------------------------------------------------------------


class _CompletionsShapeSource:
    """Maps a request index to an OpenAI-completions request shape.

    Cycles through the dataset prompts by index so the measured/warmup traffic and
    the readiness probe carry representative bodies. ``temperature=0`` keeps the
    request deterministic; ``max_tokens`` comes from the task's output budget.
    """

    def __init__(self, prompts: Sequence[str], *, model: str, max_tokens: int) -> None:
        # A non-empty prompt list is guaranteed by the caller (load_prompts).
        self._prompts = list(prompts)
        self._model = model
        self._max_tokens = max_tokens

    def __call__(self, index: int) -> RequestShape:
        prompt = self._prompts[index % len(self._prompts)] if self._prompts else "ready?"
        payload = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
        }
        return RequestShape(index=index, payload=payload)


def _level_traffic_source(
    base: TrafficConfig,
    *,
    rate: float,
    arrival: str,
    horizon_seconds: float,
    shape_source: ShapeSource,
) -> TrafficSource:
    """Build an open-loop source whose schedule covers ``horizon_seconds`` (contract 1).

    The schedule must span the WHOLE level (ramp + every measured window) as one
    continuous run, so the horizon-sized ``window_seconds`` is projected onto a
    copy of the level's traffic config (carrying its concurrency cap, burstiness,
    and seed). The window manager owns per-window timing; this only sizes the
    issuance schedule.
    """
    level_config = base.model_copy(
        update={
            "rate": rate,
            "arrival": arrival,
            "window_seconds": horizon_seconds,
            "window_requests": None,
        }
    )
    return OpenLoopPoissonSource(level_config, seed=base.seed, shape_source=shape_source)


# ---------------------------------------------------------------------------
# The level driver (contract 3). A standalone coroutine so the abort / continue /
# session-abort logic is unit-testable with fake managers; ServerSession composes
# it. Catches AbortedLevel at the DIRECT run_level await site - never through an
# outer Task boundary that would drop the attribute.
# ---------------------------------------------------------------------------


@dataclass
class _LevelFailure:
    """A recorded level failure (never dropped): reason + preserved partial state."""

    level_index: int
    reason: str
    aborted_window_index: int | None
    completed_cores: list[MeasuredWindowCore | None]

    @classmethod
    def from_aborted(cls, aborted: AbortedLevel) -> _LevelFailure:
        return cls(
            level_index=aborted.level_index,
            reason=aborted.reason,
            aborted_window_index=aborted.aborted_window_index,
            completed_cores=list(aborted.completed_cores),
        )


async def _watch_interrupt(
    interrupt_event: Any,
    target: asyncio.Task[Any],
    *,
    poll_interval: float,
    sleep: Callable[[float], Awaitable[None]],
) -> None:
    """Cancel ``target`` once ``interrupt_event`` is set (the SIGINT bridge).

    The study runner's SIGINT handler records the interrupt on a threading event;
    this watcher polls it and cancels the driving task so a mid-window / mid-warmup
    interrupt unwinds cleanly (the window manager aborts the open window and the
    server is reaped in ``__exit__``). Cancelling the DRIVING task (which awaits
    ``run_level`` inline) delivers the CancelledError into ``run_level`` itself, so
    its AbortedLevel attaches to the exception the driver already catches inline.
    """
    while not interrupt_event.is_set():
        await sleep(poll_interval)
    target.cancel()


async def _drive_levels(
    manager: WindowManager,
    level_plans: Sequence[LevelPlan],
    outcomes_out: list[LevelOutcome],
    failures_out: list[_LevelFailure],
    *,
    interrupt_event: Any | None = None,
    cooldown_seconds: float = 0.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    poll_interval: float = _INTERRUPT_POLL_INTERVAL_S,
    on_level: Callable[[LevelOutcome], None] | None = None,
    on_level_start: Callable[[int], None] | None = None,
) -> str:
    """Drive the levels, catching per-level aborts inline. Returns a session status.

    Appends each clean :class:`LevelOutcome` to ``outcomes_out`` and each recorded
    failure to ``failures_out`` AS IT GOES, so an interrupt that re-raises still
    leaves the partial state visible to the caller. Returns ``"ok"`` when every
    level was attempted, or ``"warmup_aborted"`` when a ``WarmupTrafficError``
    aborted the session (a dead transport dooms later levels). Only an interrupt
    (CancelledError / KeyboardInterrupt) propagates; every other level failure is
    recorded invalid-with-reason and the session continues (contract 3), so one
    bad level never crashes the study (mirrors the offline DockerError path).

    ``on_level`` is invoked with each clean outcome AS the level closes (SM10: a
    level's window bundles are persisted at level close so a later interrupt cannot
    lose them). It runs between levels while no traffic is in flight; a failure in
    it is logged and never crashes the drive.
    """
    target = asyncio.current_task()
    watcher: asyncio.Task[None] | None = None
    if interrupt_event is not None and target is not None:
        watcher = asyncio.create_task(
            _watch_interrupt(interrupt_event, target, poll_interval=poll_interval, sleep=sleep)
        )
    try:
        for level_index, level in enumerate(level_plans):
            if level_index > 0 and cooldown_seconds > 0.0:
                await sleep(cooldown_seconds)
            if on_level_start is not None:
                # Per-cell manifest lifecycle: mark this level's cell running as its
                # level opens (SM10). A hook failure must not crash the drive.
                try:
                    on_level_start(level_index)
                except Exception:
                    logger.exception("Level-start hook failed for level %d", level_index)
            try:
                # INLINE await: the AbortedLevel the manager attaches to a failing
                # level's exception survives, because there is no intervening Task
                # boundary to substitute a fresh CancelledError (contract 3).
                outcome = await manager.run_level(level_index, level)
                outcomes_out.append(outcome)
                if on_level is not None:
                    # Persist this level's window bundles at level close (SM10). A
                    # persistence failure must not crash the drive; it is logged.
                    try:
                        on_level(outcome)
                    except Exception:
                        logger.exception("Persisting level %d window bundles failed", level_index)
            except WarmupTrafficError as exc:
                # The warmup traffic mechanism is dead: no window opened, and the
                # measured window's traffic on the same transport would fail too.
                # Abort the whole session (do not silently skip the level).
                failures_out.append(
                    _LevelFailure(
                        level_index=level_index,
                        reason=f"warmup failed: {exc}",
                        aborted_window_index=None,
                        completed_cores=[],
                    )
                )
                return "warmup_aborted"
            except (asyncio.CancelledError, KeyboardInterrupt) as exc:
                # Interrupt (SIGINT via the watcher, or a genuine cancellation):
                # preserve whatever the manager measured, then propagate so the
                # session's __exit__ reaps the server.
                aborted = getattr(exc, ABORTED_LEVEL_ATTR, None)
                if aborted is not None:
                    failures_out.append(_LevelFailure.from_aborted(aborted))
                else:
                    # No AbortedLevel: the interrupt landed before any window opened
                    # (e.g. mid-warmup or mid-ramp). Record a traceable partial so
                    # the level is not indistinguishable from nothing-attempted.
                    failures_out.append(
                        _LevelFailure(
                            level_index=level_index,
                            reason="interrupted (warmup)",
                            aborted_window_index=None,
                            completed_cores=[],
                        )
                    )
                raise
            except Exception as exc:
                # Any other level failure is recorded and the session continues
                # (SystemExit / GeneratorExit are BaseException-not-Exception and
                # propagate; CancelledError / KeyboardInterrupt are handled above).
                # An AbortedLevel (window / close / drain failure) carries the
                # cleanly-closed cores to preserve; a pure ramp-phase failure has
                # nothing to preserve but still failed the level - record it too.
                aborted = getattr(exc, ABORTED_LEVEL_ATTR, None)
                if aborted is not None:
                    failures_out.append(_LevelFailure.from_aborted(aborted))
                else:
                    failures_out.append(
                        _LevelFailure(
                            level_index=level_index,
                            reason=f"level failed: {exc!r}",
                            aborted_window_index=None,
                            completed_cores=[],
                        )
                    )
                continue
        return "ok"
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(BaseException):
                await watcher


# ---------------------------------------------------------------------------
# ServerSession - the C1/C3 sibling context manager.
# ---------------------------------------------------------------------------


@dataclass
class ServerCell:
    """One grid point folded into a session: a config + its identity (O7.3).

    A session groups grid points that are identical except ``server.traffic.rate``
    (rate is identity per C4). Each cell drives ONE rate level of the session and
    owns its own declared-config hash and cycle, so its window bundles land under
    its own grid-point hash and its manifest entry resolves per-cell.
    """

    config: ExperimentConfig
    config_hash: str
    cycle: int


class ServerSession:
    """Online-serving measurement session: launch -> warm up -> windows -> shutdown.

    Constructed like the offline sessions (runner + config + identity + the
    resolved runner spec); ``engine`` is injectable for tests, else resolved from
    the config. A session drives ONE or MORE grid points (cells) over one server
    lifetime: a single-cell session is one grid point; a grouped session folds
    consecutive rate-only-varying cells (O7.3) into one launch with a rate level
    per cell. The seam-building methods (``_make_shape_source`` / ``_make_transport``
    / ``_make_energy_sink`` / ``_make_sampler_factory``) are small and overridable
    so unit tests exercise ``run()`` with fakes.
    """

    def __init__(
        self,
        runner: StudyRunner,
        config: ExperimentConfig,
        spec: RunnerSpec | None,
        *,
        config_hash: str,
        cycle: int,
        index: int,
        engine: ServerCapable | None = None,
        cells: list[ServerCell] | None = None,
    ) -> None:
        # A single-cell session is the degenerate group of one (O7.1). ``config`` /
        # ``config_hash`` / ``cycle`` remain the FIRST cell's for the shared launch
        # (model/engine are identical across members; only rate differs) and for the
        # existing single-cell call sites; ``cells`` carries the per-rate-level grid
        # points a grouped session drives.
        self._cells = cells if cells is not None else [ServerCell(config, config_hash, cycle)]
        self._runner = runner
        self.config = self._cells[0].config
        self.spec = spec
        self.config_hash = self._cells[0].config_hash
        self.cycle = self._cells[0].cycle
        self.index = index
        self._engine = engine
        # Cells whose level opened and were marked running (per-cell manifest
        # lifecycle): guards against a double mark and drives interrupt downgrade.
        self._running_cells: set[int] = set()
        self._torn_down = False
        self._exp_start = 0.0
        # Session identity + phase raws (the SessionBlock stamped into every window
        # bundle of this session; O7.2). One session id per session realisation.
        self._session_id = uuid.uuid4().hex
        self._level_count = 0
        self._window_count = 0
        self._launch_duration_s: float | None = None
        self._launch_energy_j: float | None = None
        self._drain_duration_s: float | None = None
        self._drain_energy_j: float | None = None
        self._interrupted = False
        # Per-window bundle writers, held from level-close write until session close
        # so the drain raws patch + finalize sweep run once per bundle (O7.4).
        self._pending_writers: list[BundleWriter] = []
        # The mapped per-window ExperimentResults + their on-disk paths, surfaced on
        # the session result so orchestration enters them into StudyResult (point 6).
        self._experiment_results: list[ExperimentResult] = []
        self._result_files: list[str] = []
        # Each level's representative bundle rel-path for the manifest result_file.
        self._level_result_file: dict[int, str] = {}
        # Per-cell resolved-config hash cache (the config.json sidecar's R7W
        # realised-protocol provenance; computed once per grid point).
        self._resolved_hash_cache: dict[str, str | None] = {}
        # Runner provenance is fixed for the session lifetime (self.spec is), so it
        # is built once and reused across every window bundle + abort core.
        self._runner_provenance_cache: Any = None
        self._env_snapshot: EnvironmentSnapshot | None = None
        # Populated in __enter__.
        self._handle: ServerHandle | None = None
        self._transport: Transport | None = None
        self._shape_source: ShapeSource | None = None
        self._warmup: ServerWarmup | None = None

    @classmethod
    def for_group(
        cls,
        runner: StudyRunner,
        cells: list[ServerCell],
        spec: RunnerSpec | None,
        *,
        index: int,
        engine: ServerCapable | None = None,
    ) -> ServerSession:
        """Build a session over a grid-point group (O7.3): one rate level per cell."""
        first = cells[0]
        return cls(
            runner,
            first.config,
            spec,
            config_hash=first.config_hash,
            cycle=first.cycle,
            index=index,
            engine=engine,
            cells=list(cells),
        )

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> ServerSession:
        runner = self._runner
        config = self.config
        try:
            self._exp_start = time.monotonic()
            # begin_experiment first so a launch/readiness failure still balances
            # with the end_experiment_fail the runner emits on the failure path.
            self._begin_progress()
            engine = self._resolve_engine()

            shape_source = self._make_shape_source()
            self._shape_source = shape_source
            placement = self._make_placement()

            # Launch-to-ready is ONE instrumented phase (model load rides inside it):
            # bracket the launch + readiness with the SAME MeasurementBracket the
            # windows use (C2), so the session block carries the launch duration and
            # GPU energy. Host-side NVML samples the GPU regardless of whether the
            # server is a sibling container or a host subprocess (co-located v0.7).
            def _launch_to_ready() -> None:
                self._handle = engine.launch(config, placement)
                probe = build_probe_request(
                    shape_source, path=SERVING_COMPLETIONS_PATH, method="POST"
                )
                engine.await_ready(self._handle, probe, timeout=self._readiness_timeout())

            self._launch_duration_s, self._launch_energy_j = self._measure_phase(
                "server launch + readiness", _launch_to_ready
            )
            assert self._handle is not None  # set inside _launch_to_ready on success

            self._transport = self._make_transport(self._handle.base_url)

            # The server is up: mark the FIRST cell running (a launch/readiness
            # failure above never reaches here, so a failed launch is recorded as
            # failed, not left running). Grouped-session cells 1..N-1 are marked
            # running as their levels open (the on_level_start hook).
            runner.manifest.mark_running(self.config_hash, self.cycle)
            self._running_cells.add(0)
        except BaseException:
            # Acquisition failed before the `with` body: release the partially
            # launched server and re-raise (a failed launch never leaks). No bundles
            # were written, so there is no drain to measure and nothing to finalize.
            self._torn_down = True
            self._cleanup(clean=False)
            raise
        return self

    def run(self) -> ServerSessionResult | dict[str, Any]:
        """Drive the level(s) to N window results; return the session product.

        Returns a :class:`ServerSessionResult` on success (or partial / interrupt),
        or a failure dict when the session aborts (``WarmupTrafficError`` or no
        valid window). The dict mirrors the offline failure shape so the sweep
        loop, circuit breaker, and manifest treat it uniformly.
        """
        assert self._transport is not None and self._shape_source is not None
        interrupt_event = getattr(self._runner, "_interrupt_event", None)

        self._warmup = self._make_warmup(self._shape_source, self._transport)
        manager = self._make_manager(self._make_energy_sink(), self._warmup)
        level_plans = self._make_level_plans(self._shape_source, self._transport)
        self._level_count = len(level_plans)

        outcomes: list[LevelOutcome] = []
        failures: list[_LevelFailure] = []
        interrupted = False
        status = "ok"
        try:
            status = asyncio.run(
                _drive_levels(
                    manager,
                    level_plans,
                    outcomes,
                    failures,
                    interrupt_event=interrupt_event,
                    cooldown_seconds=self._cooldown_seconds(),
                    on_level=self._persist_clean_level,
                    on_level_start=self._mark_cell_running,
                )
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            if interrupt_event is not None and interrupt_event.is_set():
                # Interrupted mid-session: the partial state is in outcomes /
                # failures. Leave the manifest 'running' - the sweep loop's
                # mark_interrupted downgrades it, as on the offline path.
                interrupted = True
            else:
                raise

        self._interrupted = interrupted
        return self._finalise(outcomes, failures, status=status, interrupted=interrupted)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
        if self._torn_down:
            return None
        self._torn_down = True
        # Clean close = no exception propagating AND not interrupted: only then is
        # the drain measured and its raws patched into the sibling bundles (O7.4).
        clean = exc_type is None and not self._interrupted
        # Teardown is best-effort AS A WHOLE (F6 session-hardening posture): a fault
        # in cleanup must never convert a completed measurement into a failure by
        # escaping to the dispatch site. It is logged LOUDLY, never swallowed
        # silently. KeyboardInterrupt / SystemExit (BaseException) still propagate.
        try:
            self._cleanup(clean=clean)
        except Exception:
            logger.warning(
                "Server session teardown raised (best-effort); the session result "
                "stands and the on-disk bundles are unaffected.",
                exc_info=True,
            )
        return None

    # -- finalisation --------------------------------------------------------

    def _finalise(
        self,
        outcomes: list[LevelOutcome],
        failures: list[_LevelFailure],
        *,
        status: str,
        interrupted: bool,
    ) -> ServerSessionResult | dict[str, Any]:
        """Build the session result (and mark the manifest) from the driver output."""
        warmup_by_level = self._warmup_results_by_level()
        levels = self._build_level_results(outcomes, failures, warmup_by_level)
        # Clean levels wrote their window bundles at level close (the on_level hook);
        # now flush each failed level's preserved abort cores as degraded-but-truthful
        # bundles (energy core present, boundary/bookkeeping null, O7.4).
        self._persist_abort_cores(levels)
        # Attach a uniform session block (final totals, drain null) to the in-memory
        # mapped results so StudyResult.experiments reads one shape. The drain raws are
        # a post-return session-close measurement (stamped only into the on-disk
        # bundles, which are the system of record - O7.5).
        close_block = self._build_session_block(final=False)
        self._experiment_results = [
            r.model_copy(update={"session": close_block}) for r in self._experiment_results
        ]

        total_energy = _sum_window_energy(levels)
        elapsed = time.monotonic() - self._exp_start
        aborted = status == "warmup_aborted"
        abort_reason = failures[-1].reason if aborted and failures else None

        result = ServerSessionResult(
            engine=_engine_name(self.config),
            config_hash=self.config_hash,
            cycle=self.cycle,
            index=self.index,
            serving_mode=self.config.serving_mode,
            levels=levels,
            token_counting=TOKEN_COUNTING_CLIENT_STREAMED,
            total_window_energy_j=total_energy,
            elapsed_s=elapsed,
            aborted=aborted,
            abort_reason=abort_reason,
            experiment_results=list(self._experiment_results),
            result_files=list(self._result_files),
        )

        if interrupted:
            # Credit cleanly-closed levels now (their outcome is known and their
            # bundles are persisted+finalized), so resume does not re-run them and
            # duplicate their bundles; in-flight / aborted / unreached cells stay
            # running for the sweep loop's mark_interrupted downgrade (M3).
            self._resolve_manifest_per_cell(
                levels, aborted=aborted, abort_reason=abort_reason, interrupted=True
            )
            self._end_progress(result, ok=result.valid)
            return result

        # Resolve each grid point (cell) on its own level's outcome (point 5). The
        # manifest result_file points at that level's validated window bundle (the
        # full sibling set is rediscovered via session_id + window_index fields; the
        # manifest is run state, not a results index).
        self._resolve_manifest_per_cell(levels, aborted=aborted, abort_reason=abort_reason)

        if not result.valid:
            message = abort_reason or self._invalid_message(levels)
            error_type = "WarmupTrafficError" if aborted else "ServerSessionInvalid"
            self._end_progress(result, ok=False)
            return {"type": error_type, "message": message}

        self._end_progress(result, ok=True)
        return result

    def _build_level_results(
        self,
        outcomes: list[LevelOutcome],
        failures: list[_LevelFailure],
        warmup_by_level: dict[int, ServerWarmupResult],
    ) -> list[ServerLevelResult]:
        by_level: dict[int, ServerLevelResult] = {}
        for outcome in outcomes:
            warmup = warmup_by_level.get(outcome.level_index)
            protocol = self._protocol_label(warmup)
            windows = [
                ServerWindowResult(
                    level_index=outcome.level_index,
                    window=window,
                    warmup=warmup,
                    pre_window_protocol=protocol,
                )
                for window in outcome.windows
            ]
            by_level[outcome.level_index] = ServerLevelResult(
                level_index=outcome.level_index,
                spec=outcome.spec,
                windows=windows,
                validation=outcome.validation,
                warmup=warmup,
                invalid_reason=None,
            )
        for failure in failures:
            warmup = warmup_by_level.get(failure.level_index)
            # A failed level keeps its cleanly-closed cores RAW (O7.4): full
            # per-window bookkeeping was lost with the abort (the traffic report is
            # gone once the issuer task was cancelled), so no ServerWindowResult is
            # built, but the measured cores are preserved on the level result and
            # their energy still counts toward the session total. SM10 owns their
            # bundle fate (drain-fields-null).
            by_level[failure.level_index] = ServerLevelResult(
                level_index=failure.level_index,
                spec=None,
                windows=[],
                validation=None,
                warmup=warmup,
                invalid_reason=failure.reason,
                aborted_window_index=failure.aborted_window_index,
                completed_cores=list(failure.completed_cores),
            )
        return [by_level[i] for i in sorted(by_level)]

    def _warmup_results_by_level(self) -> dict[int, ServerWarmupResult]:
        if self._warmup is None:
            return {}
        return {r.level_index: r for r in self._warmup.results}

    def _protocol_label(self, warmup: ServerWarmupResult | None) -> str:
        if warmup is not None:
            return warmup.pre_window_protocol
        cfg = self.config.resolved_server_warmup()
        return describe_server_warmup_protocol(cfg) if cfg is not None else "server warmup"

    def _invalid_message(self, levels: list[ServerLevelResult]) -> str:
        reasons = [level.invalid_reason for level in levels if level.invalid_reason]
        if reasons:
            return "; ".join(reasons)
        gate = [
            level.validation.reason
            for level in levels
            if level.validation is not None and level.validation.reason
        ]
        if gate:
            return "; ".join(gate)
        return "the server session produced no valid measured window."

    # -- persistence (per-window bundles, SM10) ------------------------------

    def _persist_clean_level(self, outcome: LevelOutcome) -> None:
        """Persist one clean level's window bundles at level close (the on_level hook).

        No-op when the runner exposes no ``study_dir`` (the unit-test fakes drive
        the manager + manifest logic without a bundle root). Each window maps to one
        ExperimentResult (serving_mode="server") and is written through the existing
        BundleWriter path; the level's issued requests are partitioned per window
        into requests.parquet (SM11), and finalize is deferred to session close
        (O7.4).
        """
        if getattr(self._runner, "study_dir", None) is None:
            return
        cell = self._cells[outcome.level_index]
        warmup = self._warmup_results_by_level().get(outcome.level_index)
        protocol = self._protocol_label(warmup)
        level_valid = outcome.validation.valid
        invalid_reason = None if level_valid else (outcome.validation.reason or "level invalid")
        rows_by_window = build_request_rows_by_window(
            outcome.issuer_report.records, outcome.windows, level_index=outcome.level_index
        )
        for window, rows in zip(outcome.windows, rows_by_window, strict=True):
            wr = ServerWindowResult(
                level_index=outcome.level_index,
                window=window,
                warmup=warmup,
                pre_window_protocol=protocol,
            )
            result = self._map_window(
                wr,
                cell=cell,
                level_window_count=len(outcome.windows),
                level_valid=level_valid,
                invalid_reason=invalid_reason,
                server_output_tokens=_sum_server_completion_tokens(rows),
            )
            samples = self._window_samples(window.energy)
            self._persist_bundle(
                result,
                samples=samples,
                cell=cell,
                level_index=outcome.level_index,
                request_rows=rows,
                request_span=(window.boundaries.span_start, window.boundaries.span_end),
            )

    def _persist_abort_cores(self, levels: list[ServerLevelResult]) -> None:
        """Flush each failed level's preserved abort cores as degraded bundles.

        A failed level lost its per-window bookkeeping with the abort, so the bundle
        carries the measured energy core only (tokens / boundaries null) with the
        aborted-level disclosure in the server provenance. The per-request log was
        lost with the bookkeeping, so no requests.parquet is written and its
        finalize backstop is suppressed (``request_rows=None``). No-op without a
        study_dir.
        """
        if getattr(self._runner, "study_dir", None) is None:
            return
        protocol_by_level = self._warmup_results_by_level()
        for level in levels:
            if not level.completed_cores:
                continue
            cell = self._cells[level.level_index]
            warmup = protocol_by_level.get(level.level_index)
            protocol = self._protocol_label(warmup)
            for window_index, core in enumerate(level.completed_cores):
                result = self._map_degraded_core(
                    core,
                    cell=cell,
                    level_index=level.level_index,
                    window_index=window_index,
                    level_window_count=len(level.completed_cores),
                    invalid_reason=level.invalid_reason or "level aborted",
                    warmup=warmup,
                    protocol=protocol,
                )
                samples = self._window_samples(core)
                self._persist_bundle(
                    result,
                    samples=samples,
                    cell=cell,
                    level_index=level.level_index,
                    request_rows=None,
                )

    def _persist_bundle(
        self,
        result: ExperimentResult,
        *,
        samples: list[Any],
        cell: ServerCell,
        level_index: int,
        request_rows: list[RequestLogRow] | None,
        request_span: tuple[float, float] | None = None,
    ) -> None:
        """Write one window bundle through the existing writer path; defer finalize.

        No ts_source_dir / rescue analogue applies: the measurement loop is
        host-side by design even when the engine server is a sibling container, so
        there is no in-container staging directory. The runner block still carries
        the image provenance. config.json is written host-side directly (the
        declared config + declared/resolved hashes; the observed half is
        container-boundary state deferred to SM12), and the timeseries parquet -
        when the window core carried samples - is written directly into the bundle
        from those in-memory samples via the same writer the offline harness uses
        (no new sampling machinery).

        ``request_rows`` is the window's per-request log (SM11): a list (possibly
        empty) writes requests.parquet via the registry writer method; ``None`` (a
        degraded abort core, whose bookkeeping was lost) writes none and suppresses
        the finalize backstop for that server bundle. ``request_span`` is the
        window's measured monotonic (span_start, span_end), stored as file metadata
        so the receipt-unclipped rows are re-clippable offline (M1). A parquet-write
        hiccup is swallowed so the finalize sweep reports the missing artefact
        loudly rather than crashing persistence.
        """
        from llenergymeasure.results.bundle import BundleWriter

        self._window_count += 1
        preliminary = self._build_session_block(final=False)
        writer = BundleWriter(
            self._runner.study_dir,
            model_name=result.model_name,
            engine=result.engine,
            config_hash=cell.config_hash,
            cycle=cell.cycle,
            experiment_index=self.index,
        )
        result_path = writer.write_result(
            result, runner_provenance=self._runner_provenance(), session=preliminary
        )
        if result.timeseries and samples:
            self._write_timeseries(writer.bundle_dir, samples, result)
        if request_rows is not None:
            span_start, span_end = request_span if request_span is not None else (None, None)
            with contextlib.suppress(Exception):
                writer.write_requests(
                    request_rows,
                    experiment_id=result.experiment_id,
                    span_start=span_start,
                    span_end=span_end,
                )
        else:
            writer.mark_artefact_absent("requests")
        self._write_config_sidecar(writer.bundle_dir, cell, result)
        writer.write_system(
            host_snapshot=self._host_snapshot(),
            runner=self._runner_provenance(),
            session=preliminary,
        )
        writer.move_config_sidecar()  # no-op host-side; config.json is written above
        self._pending_writers.append(writer)
        self._experiment_results.append(result)
        self._result_files.append(str(result_path))
        with contextlib.suppress(ValueError):
            self._level_result_file[level_index] = str(
                result_path.relative_to(self._runner.study_dir)
            )

    def _write_timeseries(self, bundle_dir: Any, samples: list[Any], result: Any) -> None:
        """Write timeseries.parquet from the window core's in-memory samples."""
        from llenergymeasure.domain.bundle_artefacts import TIMESERIES_FILENAME
        from llenergymeasure.harness.timeseries import write_timeseries_parquet

        with contextlib.suppress(Exception):
            write_timeseries_parquet(
                samples,
                bundle_dir / TIMESERIES_FILENAME,
                experiment_id=result.experiment_id,
                declared_config_hash=result.declared_config_hash,
            )

    def _write_config_sidecar(self, bundle_dir: Any, cell: ServerCell, result: Any) -> None:
        """Write the per-window config.json host-side (declared config + hashes).

        Server bundles carry a config.json like offline bundles so the resolved
        config hash (the R7W realised-protocol provenance the resume guard keys on)
        lands on disk and a config.json glob over the results does not fork on
        serving mode. The OBSERVED half (observed_engine_params /
        observed_sampling_params / observed_config_hash) and the running
        engine_version are container-boundary state the host cannot see for server
        mode; they are OMITTED (not nulled) and land with SM12.
        """
        from llenergymeasure.results.persistence import save_config_sidecar

        with contextlib.suppress(Exception):
            save_config_sidecar(
                bundle_dir,
                experiment_id=result.experiment_id,
                config_hash=cell.config_hash,
                engine=result.engine,
                model_name=result.model_name,
                measurement_methodology="server_windowed",
                resolved_config_hash=self._resolved_config_hash(cell),
                declared_config=cell.config.model_dump(mode="json"),
            )

    def _resolved_config_hash(self, cell: ServerCell) -> str | None:
        """Resolved-config hash for a cell (build_resolved_view + hash_config), cached.

        The same pipeline that stamps the manifest entry's resolved hash, so the
        sidecar and the manifest agree. Best-effort: a failure yields None (the
        sidecar then omits the field).
        """
        if cell.config_hash in self._resolved_hash_cache:
            return self._resolved_hash_cache[cell.config_hash]
        resolved: str | None
        try:
            from llenergymeasure.study.hashing import resolved_config_hash

            resolved = resolved_config_hash(cell.config)
        except Exception:
            resolved = None
        self._resolved_hash_cache[cell.config_hash] = resolved
        return resolved

    # -- per-cell manifest lifecycle (point 5) -------------------------------

    def _mark_cell_running(self, level_index: int) -> None:
        """Mark this level's cell running as its level opens (the on_level_start hook).

        Cell 0 is already marked running in ``__enter__`` (the shared launch marks
        the first grid point); this covers the grouped session's cells 1..N-1.
        """
        if level_index in self._running_cells:
            return
        self._running_cells.add(level_index)
        cell = self._cells[level_index]
        self._runner.manifest.mark_running(cell.config_hash, cell.cycle)

    def _resolve_manifest_per_cell(
        self,
        levels: list[ServerLevelResult],
        *,
        aborted: bool,
        abort_reason: str | None,
        interrupted: bool = False,
    ) -> None:
        """Resolve each grid point (cell) on its own level's outcome (point 5).

        A cell whose level passed the stability gate completes (with its level
        aggregates as the resume-display metrics); a cleanly-closed gate-failed
        level fails. On the CLEAN path every other cell fails too (an abort, or a
        level a warmup-abort doomed before it ran). On the INTERRUPT path only
        cleanly-CLOSED levels (validation resolved) are credited now - their outcome
        is known and their bundles are finalized, so resume must not re-run them;
        in-flight, aborted, and unreached cells stay running for the sweep loop's
        mark_interrupted downgrade (M3).
        """
        levels_by_index = {level.level_index: level for level in levels}
        error_type = "WarmupTrafficError" if aborted else "ServerSessionInvalid"
        for level_index, cell in enumerate(self._cells):
            level = levels_by_index.get(level_index)
            closed = level is not None and level.validation is not None
            if interrupted and not closed:
                # In-flight / aborted / unreached: leave running for mark_interrupted.
                continue
            if level is not None and level.validation is not None and level.valid:
                self._runner.manifest.mark_completed(
                    cell.config_hash,
                    cell.cycle,
                    self._level_result_file.get(level_index, ""),
                    **self._level_summary_metrics(level),
                )
            else:
                reason = (
                    (level.invalid_reason if level is not None else None)
                    or abort_reason
                    or "the server session produced no valid measured window."
                )
                self._runner.manifest.mark_failed(cell.config_hash, cell.cycle, error_type, reason)

    @staticmethod
    def _level_summary_metrics(level: ServerLevelResult) -> dict[str, float | None]:
        """Level aggregates for a completed cell's resume-display metrics (point 7)."""
        energy = 0.0
        tokens = 0
        seen_energy = False
        span_lo: float | None = None
        span_hi: float | None = None
        for wr in level.windows:
            window = wr.window
            if window.window_energy_j is not None:
                energy += window.window_energy_j
                seen_energy = True
            tokens += window.bookkeeping.energy_denominator_tokens
            boundaries = window.boundaries
            span_lo = (
                boundaries.span_start if span_lo is None else min(span_lo, boundaries.span_start)
            )
            span_hi = boundaries.span_end if span_hi is None else max(span_hi, boundaries.span_end)
        elapsed = (span_hi - span_lo) if span_lo is not None and span_hi is not None else None
        energy_j = energy if seen_energy else None
        return {
            "elapsed_seconds": elapsed,
            "inference_seconds": elapsed,
            "energy_joules": energy_j,
            "throughput_tok_s": (tokens / elapsed) if elapsed and elapsed > 0 else None,
            "energy_per_token_mj": (energy / tokens * 1000.0)
            if seen_energy and tokens > 0
            else None,
        }

    # -- mapping (ServerWindowResult / abort core -> ExperimentResult) --------

    def _map_window(
        self,
        wr: ServerWindowResult,
        *,
        cell: ServerCell,
        level_window_count: int,
        level_valid: bool,
        invalid_reason: str | None,
        server_output_tokens: int | None,
    ) -> ExperimentResult:
        """Map one measured window to an ExperimentResult (serving_mode='server').

        Fills the shared-core quantities truthfully (window energy J, client-counted
        output tokens, window duration, J/token) and leaves the derived server
        metrics (goodput, slo_pass, amortised breakdown) None - SM12. ``output_tokens``
        is the client-side canonical count (span-received streamed deltas, O8);
        prefill tokens are 0 (client-side input-token counting needs a host tokenizer
        and is out of scope - the engine's prompt_tokens ride only in requests.parquet).
        The counting mechanism and the auxiliary server-reported total are disclosed in
        the server provenance. Identity is the CELL's grid-point hash (rate is identity
        per C4), so the bundle lands under its own hash.
        """
        from llenergymeasure.domain.experiment import ExperimentResult

        window = wr.window
        bk = window.bookkeeping
        duration = max(window.boundaries.span_end - window.boundaries.span_start, 0.0)
        output_tokens = bk.energy_denominator_tokens
        energy_j = window.window_energy_j or 0.0
        j_per_token = window.window_j_per_token
        start_time, end_time = self._core_times(window.energy)
        server_prov = self._server_provenance(
            level_index=wr.level_index,
            window_index=window.window_index,
            level_window_count=level_window_count,
            level_valid=level_valid,
            intra_window_cov=window.intra_window_cov,
            invalid_reason=invalid_reason,
            warmup=wr.warmup,
            pre_window_protocol=wr.pre_window_protocol,
            attribution_policy=bk.attribution_policy,
            server_reported_output_tokens=server_output_tokens,
        )
        return ExperimentResult(
            experiment_id=self._window_experiment_id(
                cell.config_hash, cell.cycle, wr.level_index, window.window_index
            ),
            declared_config_hash=cell.config_hash,
            serving_mode="server",
            engine=_engine_name(cell.config),
            model_name=cell.config.task.model,
            input_tokens=0,
            output_tokens=output_tokens,
            total_tokens=output_tokens,
            total_energy_j=energy_j,
            total_inference_time_sec=duration,
            avg_tokens_per_second=(output_tokens / duration) if duration > 0 else 0.0,
            avg_energy_per_token_j=j_per_token if j_per_token is not None else 0.0,
            energy_per_token_mj_total=(j_per_token * 1000.0) if j_per_token is not None else None,
            total_flops=0.0,
            start_time=start_time,
            end_time=end_time,
            timeseries=self._timeseries_name(window.energy),
            server=server_prov,
        )

    def _map_degraded_core(
        self,
        core: MeasuredWindowCore | None,
        *,
        cell: ServerCell,
        level_index: int,
        window_index: int,
        level_window_count: int,
        invalid_reason: str,
        warmup: ServerWarmupResult | None,
        protocol: str,
    ) -> ExperimentResult:
        """Map a failed level's preserved abort core to a degraded ExperimentResult.

        The window's bookkeeping was lost with the abort, so tokens / duration are
        null-equivalents (0); only the measured GPU energy stands. The degradation
        and its cause are disclosed in the server provenance and the warnings.
        """
        from llenergymeasure.domain.experiment import ExperimentResult

        start_time, end_time = self._core_times(core)
        server_prov = self._server_provenance(
            level_index=level_index,
            window_index=window_index,
            level_window_count=level_window_count,
            level_valid=False,
            invalid_reason=invalid_reason,
            warmup=warmup,
            pre_window_protocol=protocol,
            attribution_policy=ATTRIBUTION_STEADY_STATE_SPAN,
        )
        return ExperimentResult(
            experiment_id=self._window_experiment_id(
                cell.config_hash, cell.cycle, level_index, window_index
            ),
            declared_config_hash=cell.config_hash,
            serving_mode="server",
            engine=_engine_name(cell.config),
            model_name=cell.config.task.model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            total_energy_j=_core_energy_j(core) or 0.0,
            total_inference_time_sec=0.0,
            avg_tokens_per_second=0.0,
            avg_energy_per_token_j=0.0,
            total_flops=0.0,
            start_time=start_time,
            end_time=end_time,
            timeseries=self._timeseries_name(core),
            measurement_warnings=[f"degraded window bundle (level aborted): {invalid_reason}"],
            server=server_prov,
        )

    @staticmethod
    def _server_provenance(
        *,
        level_index: int,
        window_index: int,
        level_window_count: int,
        level_valid: bool,
        invalid_reason: str | None,
        warmup: ServerWarmupResult | None,
        pre_window_protocol: str,
        attribution_policy: str,
        intra_window_cov: float | None = None,
        server_reported_output_tokens: int | None = None,
    ) -> ServerWindowProvenance:
        return ServerWindowProvenance(
            level_index=level_index,
            window_index=window_index,
            level_window_count=level_window_count,
            level_valid=level_valid,
            intra_window_cov=intra_window_cov,
            invalid_reason=invalid_reason,
            warmup=_warmup_provenance(warmup),
            pre_window_protocol=pre_window_protocol,
            attribution_policy=attribution_policy,
            token_counting=TOKEN_COUNTING_CLIENT_STREAMED,
            server_reported_output_tokens=server_reported_output_tokens,
        )

    @staticmethod
    def _window_experiment_id(
        config_hash: str, cycle: int, level_index: int, window_index: int
    ) -> str:
        # The cycle component keeps cycle 1 and cycle 2 of one grid point distinct
        # (else their result.json + timeseries would collide once a reader keys on
        # experiment_id).
        return f"server-{config_hash}-c{cycle}-L{level_index}-W{window_index}"

    @staticmethod
    def _window_samples(core: MeasuredWindowCore | None) -> list[Any]:
        return list(getattr(core, "timeseries_samples", []) or []) if core is not None else []

    @staticmethod
    def _timeseries_name(core: MeasuredWindowCore | None) -> str | None:
        from llenergymeasure.domain.bundle_artefacts import TIMESERIES_FILENAME

        samples = ServerSession._window_samples(core)
        return TIMESERIES_FILENAME if len(samples) >= 2 else None

    @staticmethod
    def _core_times(core: MeasuredWindowCore | None) -> tuple[Any, Any]:
        from datetime import datetime

        now = datetime.now()
        start = getattr(core, "start_time", None) if core is not None else None
        end = getattr(core, "end_time", None) if core is not None else None
        return (start or now, end or now)

    # -- session block + phase measurement -----------------------------------

    def _build_session_block(self, *, final: bool) -> SessionBlock:
        """Assemble the session facts. ``final`` includes the measured drain raws."""
        from llenergymeasure.domain.session import SessionBlock

        warmups = self._warmup.results if self._warmup is not None else []
        warmup_duration = sum(r.elapsed_s for r in warmups) if warmups else None
        warmup_energy = _sum_optional(r.energy_j for r in warmups)
        return SessionBlock(
            session_id=self._session_id,
            window_count=self._window_count,
            level_count=self._level_count or None,
            launch_duration_s=self._launch_duration_s,
            launch_energy_j=self._launch_energy_j,
            warmup_total_duration_s=warmup_duration,
            warmup_total_energy_j=warmup_energy,
            drain_duration_s=self._drain_duration_s if final else None,
            drain_energy_j=self._drain_energy_j if final else None,
        )

    def _measure_phase(
        self, detail: str, run: Callable[[], None]
    ) -> tuple[float | None, float | None]:
        """Run ``run()`` inside a MeasurementBracket; return (duration_s, energy_j).

        Session-phase energy (launch / warmup / drain) is AUXILIARY provenance, not
        the measurement itself, and the duration is the floor: ``run`` always
        executes and the phase duration is always measured. Energy degrades to None
        when the bracket cannot be constructed / entered / finished (e.g. no GPU /
        energy backend on the host), with one warning - it does NOT abort the phase.
        Only an exception raised by ``run`` itself propagates (a launch/drain failure
        must still fail). This reuses the SAME bracket machinery the windows use
        (C2); windows, unlike these phases, legitimately hard-require the backend.
        """
        start = time.monotonic()
        bracket: MeasurementBracket | None = None
        try:
            bracket = self._make_phase_bracket(detail)
            bracket.__enter__()
        except Exception:
            # Energy backend unavailable: still run the phase, stamp null energy.
            logger.warning(
                "Phase %r energy is unmeasurable (energy bracket unavailable); "
                "stamping null energy (the duration is still measured).",
                detail,
                exc_info=True,
            )
            run()  # a failure in run() still propagates (a launch/drain error is fatal)
            return time.monotonic() - start, None

        # The bracket is live: run the phase inside it.
        try:
            run()
        except BaseException:
            with contextlib.suppress(BaseException):
                bracket.__exit__(None, None, None)
            with contextlib.suppress(BaseException):
                bracket.finish()
            raise
        # Phase completed: close the bracket and read its energy, both best-effort
        # (a teardown / read fault degrades energy to null, never fails the phase).
        energy: float | None = None
        with contextlib.suppress(Exception):
            bracket.__exit__(None, None, None)
        with contextlib.suppress(Exception):
            energy = _phase_energy_j(bracket.finish())
        return time.monotonic() - start, energy

    def _make_phase_bracket(self, detail: str) -> MeasurementBracket:
        """Build the phase-energy bracket (overridable for tests).

        Progress is deliberately None: the phase brackets must not emit the offline
        energy-select / measure step events into the server progress surface.
        """
        from llenergymeasure.harness.bracket import MeasurementBracket

        return MeasurementBracket(
            self.config.measurement,
            self._measurement_gpu_indices(),
            None,
            measure_detail=detail,
        )

    def _host_snapshot(self) -> EnvironmentSnapshot | None:
        """Host environment snapshot for system.json (collected once, reused).

        Prefers the runner's cached study-level snapshot; falls back to a direct
        collection when the runner exposes none (e.g. a test double). Best-effort:
        a collection failure yields None (system.json then records no host block).
        """
        if self._env_snapshot is not None:
            return self._env_snapshot
        getter = getattr(self._runner, "_get_env_snapshot", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                self._env_snapshot = getter()
                return self._env_snapshot
        from llenergymeasure.harness.environment import collect_environment_snapshot

        with contextlib.suppress(Exception):
            self._env_snapshot = collect_environment_snapshot()
        return self._env_snapshot

    def _runner_provenance(self) -> Any:
        """The runner provenance block (image provenance for a container placement).

        Built once and cached: ``self.spec`` is fixed for the session lifetime and
        the mapping is deterministic, so every window bundle reuses one instance.
        """
        if self._runner_provenance_cache is None:
            from llenergymeasure.study.runner import _provenance_from_spec

            self._runner_provenance_cache = _provenance_from_spec(self.spec)
        return self._runner_provenance_cache

    # -- seam construction (overridable for tests) ---------------------------

    def _resolve_engine(self) -> ServerCapable:
        if self._engine is not None:
            return self._engine
        from llenergymeasure.engines import get_engine

        engine = get_engine(_engine_name(self.config))
        if not _is_server_capable(engine):
            raise ServerSessionError(
                f"engine {_engine_name(self.config)!r} does not support server mode "
                "(it does not implement the ServerCapable launch / await_ready / "
                "shutdown protocol). Use serving_mode: offline for this engine."
            )
        self._engine = engine  # type: ignore[assignment]
        return engine  # type: ignore[return-value]

    def _make_shape_source(self) -> ShapeSource:
        from llenergymeasure.datasets.loader import load_prompts

        prompts = load_prompts(self.config)
        max_tokens = self.config.task.max_output_tokens or 128
        return _CompletionsShapeSource(prompts, model=self.config.task.model, max_tokens=max_tokens)

    def _make_transport(self, base_url: str) -> Transport:
        from llenergymeasure.harness.traffic import HttpxTransport

        return HttpxTransport(base_url=base_url, path=SERVING_COMPLETIONS_PATH)

    def _make_energy_sink(self) -> BracketEnergySink:
        return BracketEnergySink.from_measurement_config(
            self.config.measurement,
            self._measurement_gpu_indices(),
            getattr(self._runner, "_progress", None),
        )

    def _make_sampler_factory(self) -> Callable[[], Any]:
        from llenergymeasure.device.power_thermal import PowerThermalSampler

        gpu_indices = self._measurement_gpu_indices()
        return lambda: PowerThermalSampler(gpu_indices=gpu_indices)

    def _make_warmup(self, shape_source: ShapeSource, transport: Transport) -> ServerWarmup:
        traffic = self._traffic()
        warmup_config = self.config.resolved_server_warmup()
        assert warmup_config is not None  # server mode always resolves a warmup

        def traffic_factory(context: WarmupContext, horizon: float) -> TrafficSource:
            return _level_traffic_source(
                traffic,
                rate=context.spec.rate,
                arrival=context.spec.arrival,
                horizon_seconds=max(horizon, 1.0),
                shape_source=shape_source,
            )

        return ServerWarmup(
            warmup_config,
            traffic_factory=traffic_factory,
            transport=transport,
            sampler_factory=self._make_sampler_factory(),
        )

    def _make_manager(self, energy_sink: Any, warmup: ServerWarmup) -> WindowManager:
        return WindowManager(
            energy_sink,
            windows_per_level=self._windows_per_level(),
            warmup_hook=warmup,
            drain_timeout=self._drain_timeout(),
        )

    def _make_level_plans(self, shape_source: ShapeSource, transport: Transport) -> list[LevelPlan]:
        # One level per cell (O7.3): a grouped session's cells differ only in
        # server.traffic.rate, so each drives its own rate level over the shared
        # server lifetime. A single-cell session yields exactly one plan (unchanged).
        plans: list[LevelPlan] = []
        for cell in self._cells:
            assert cell.config.server is not None  # server mode requires the section
            traffic = cell.config.server.traffic
            spec = WindowSpec.from_traffic_config(traffic)
            horizon = spec.ramp_exclusion_seconds + self._windows_per_level() * float(
                spec.duration_seconds or 0.0
            )
            source = _level_traffic_source(
                traffic,
                rate=spec.rate,
                arrival=spec.arrival,
                horizon_seconds=horizon,
                shape_source=shape_source,
            )
            plans.append(
                LevelPlan(
                    spec=spec,
                    traffic_source=source,
                    transport=transport,
                    token_receipt_fn=client_token_receipts,
                )
            )
        return plans

    # -- placement / config accessors ---------------------------------------

    def _make_placement(self) -> ServerPlacement:
        from llenergymeasure.config.ssot import RUNNER_PROCESS
        from llenergymeasure.infra.server_lifecycle import ServerPlacement

        mode = self.spec.mode if self.spec is not None else RUNNER_PROCESS
        image = self.spec.image if self.spec is not None else None
        return ServerPlacement(mode=mode, image=image, gpu_indices=self._placement_gpu_indices())

    def _traffic(self) -> TrafficConfig:
        assert self.config.server is not None  # server mode requires the section
        return self.config.server.traffic

    def _windows_per_level(self) -> int:
        from llenergymeasure.harness.window_manager import DEFAULT_WINDOWS_PER_LEVEL

        return DEFAULT_WINDOWS_PER_LEVEL

    def _cooldown_seconds(self) -> float:
        return self.config.server.cooldown_seconds if self.config.server is not None else 0.0

    def _drain_timeout(self) -> float | None:
        return self._runner.study.study_execution.experiment_timeout_seconds

    def _readiness_timeout(self) -> float:
        return float(self._runner.study.study_execution.experiment_timeout_seconds)

    def _placement_gpu_indices(self) -> list[int] | None:
        # Physical container scoping (mirrors the offline docker path).
        return self._runner.study.study_execution.gpu_indices

    def _measurement_gpu_indices(self) -> list[int] | None:
        # Logical indices the host-side NVML samplers address (mirrors offline).
        from llenergymeasure.device.gpu_info import _resolve_gpu_indices

        return _resolve_gpu_indices(self.config)

    # -- progress + cleanup --------------------------------------------------

    def _begin_progress(self) -> None:
        progress = getattr(self._runner, "_progress", None)
        if progress is None:
            return
        from llenergymeasure.domain.progress import server_steps
        from llenergymeasure.utils.formatting import format_experiment_header

        progress.begin_experiment(
            self.index,
            format_experiment_header(self.config),
            server_steps(),
            runner_info=self.spec.to_runner_info() if self.spec is not None else None,
        )

    def _end_progress(self, result: ServerSessionResult, *, ok: bool) -> None:
        progress = getattr(self._runner, "_progress", None)
        if progress is None:
            return
        elapsed = result.elapsed_s
        if ok:
            progress.end_experiment_ok(
                self.index,
                elapsed,
                energy_j=result.total_window_energy_j,
            )
        else:
            progress.end_experiment_fail(
                self.index, elapsed, error=result.abort_reason or "server session invalid"
            )

    def _cleanup(self, *, clean: bool) -> None:
        """Shut the server down + drain-finalise; runs exactly once from __exit__.

        Idempotent and best-effort (it runs on the normal, SIGINT, and exception
        paths). The engine's ``shutdown`` is itself idempotent + leak-free (SM6),
        so a double invocation is a no-op; the transport's connection pool is
        closed too so a launched-but-never-run session leaks nothing.

        On a CLEAN close the server shutdown IS the session drain phase (D19: the
        drain is an event-delineated phase, not a clock-diff reconstruction), so it
        is bracketed for duration + energy and those raws are patched into every
        sibling bundle before finalize. On the interrupted / exception path the
        server is reaped without measuring a drain (the drain fields stay null) and
        the already-written bundles are finalized as-is.
        """
        handle = self._handle
        if handle is not None:
            if clean and self._pending_writers:
                self._drain_duration_s, self._drain_energy_j = self._drain_shutdown(handle)
            else:
                with contextlib.suppress(Exception):
                    self._shutdown_handle(handle)
        self._finalize_bundles(clean=clean)
        transport = self._transport
        if transport is not None:
            aclose = getattr(transport, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    _run_sync(aclose())

    def _drain_shutdown(self, handle: ServerHandle) -> tuple[float | None, float | None]:
        """Bracket the server shutdown as the drain phase; fall back to plain reap.

        Returns ``(duration_s, energy_j)``. If bracketing itself fails (e.g. no
        sampler), the server is still reaped and the drain raws come back null - the
        server is never leaked over a measurement hiccup.
        """
        try:
            return self._measure_phase("server drain", lambda: self._shutdown_handle(handle))
        except Exception:
            with contextlib.suppress(Exception):
                self._shutdown_handle(handle)
            return None, None

    def _finalize_bundles(self, *, clean: bool) -> None:
        """Patch the final session block (clean close) then finalize every bundle.

        The window bundles were written at level close carrying a PRELIMINARY
        session block (drain null, counts not yet final). On a clean close the drain
        raws and final totals are now known, so the complete block is patched into
        each sibling before its finalize sweep (O7.4/O7.5). On the interrupted path
        the drain patch is skipped (fields stay null) and the bundles are finalized
        as written.
        """
        if not self._pending_writers:
            return
        # Always patch the best-known session block into every sibling before
        # finalize, so all siblings carry the FINAL window/level counts (not the
        # incremental preliminary window_count they were written with). The drain
        # raws are included ONLY on a clean close (final=clean); they stay null on
        # the interrupt path (O7.4). The block build is the one non-trivial call on
        # this otherwise fully-defensive teardown path, so it is guarded: a fault
        # degrades to unpatched-but-finalized bundles rather than escaping (which
        # would flip completed manifest entries to failed).
        final_block: SessionBlock | None = None
        try:
            final_block = self._build_session_block(final=clean)
        except Exception:
            logger.warning(
                "Building the session block failed; window bundles are finalized "
                "with their preliminary session block.",
                exc_info=True,
            )
        if final_block is not None:
            for writer in self._pending_writers:
                with contextlib.suppress(Exception):
                    writer.patch_session_block(final_block)
        for writer in self._pending_writers:
            with contextlib.suppress(Exception):
                writer.finalize()

    def _shutdown_handle(self, handle: ServerHandle) -> None:
        """Reap the launched server via the engine protocol (falls back to infra)."""
        if self._engine is not None:
            self._engine.shutdown(handle)
            return
        from llenergymeasure.infra.server_lifecycle import shutdown

        shutdown(handle)


class ServerSessionError(Exception):
    """A server session could not be set up (e.g. a non-server-capable engine)."""


# ---------------------------------------------------------------------------
# Session grouping (O7.3): partition the ordered study into dispatch units.
# ---------------------------------------------------------------------------


def _group_key(config: ExperimentConfig) -> str:
    """Canonical identity of a server config with server.traffic.rate stripped.

    Two server configs fold into one session iff they are identical except
    ``server.traffic.rate`` (rate is identity per C4), so the group key is the
    JSON-stable declared dump minus that field (and minus ``slo``, already excluded
    from identity as a post-hoc overlay - so an slo-only difference never splits a
    group either).
    """
    import json

    dumped = config.model_dump(mode="json", exclude={"server": {"traffic": {"slo", "rate"}}})
    return json.dumps(dumped, sort_keys=True)


def partition_server_groups(configs: Sequence[ExperimentConfig]) -> list[list[int]]:
    """Partition the ordered configs into dispatch units by session grouping (O7.3).

    Returns a list of units, each a list of indices into ``configs``, in order and
    never reordered. A unit is either a single cell (an offline experiment, or a
    server experiment that folds with no neighbour) or a run of CONSECUTIVE
    server cells whose configs are identical except ``server.traffic.rate`` AND that
    belong to the SAME cycle. Cycle membership is tracked prospectively: the nth
    occurrence of a declared hash across the walk is its cycle n, so folding
    requires both a matching rate-stripped key and an equal prospective cycle. A
    non-server cell, any non-rate difference, or a cycle-number change ends the
    current group and starts a fresh one, so repeat cycles become fresh sessions.

    ORDERING CONSEQUENCE: under the DEFAULT ``sequential`` order the execution
    sequence is ``[A, A, B, B, ...]`` (each grid point's cycles adjacent), so a
    rate sweep's cells are never both consecutive AND same-cycle - sequential
    server sweeps therefore dispatch as SINGLETON sessions (one launch per grid
    point). Under ``interleave`` the sequence is ``[A, B, ..., A, B, ...]``, so
    each pass folds into one session (one launch per sweep per cycle). A
    single-cycle sweep folds under either order.
    """
    from llenergymeasure.domain.experiment import compute_declared_config_hash

    units: list[list[int]] = []
    current: list[int] = []
    current_key: str | None = None
    current_cycle: int | None = None
    occurrence_count: dict[str, int] = {}

    def _flush() -> None:
        nonlocal current, current_key, current_cycle
        if current:
            units.append(current)
        current = []
        current_key = None
        current_cycle = None

    for index, config in enumerate(configs):
        if config.serving_mode != "server":
            _flush()
            units.append([index])
            continue
        declared = compute_declared_config_hash(config)
        prospective_cycle = occurrence_count.get(declared, 0) + 1
        occurrence_count[declared] = prospective_cycle
        key = _group_key(config)
        if current and key == current_key and prospective_cycle == current_cycle:
            current.append(index)
        else:
            _flush()
            current = [index]
            current_key = key
            current_cycle = prospective_cycle
    _flush()
    return units


# ---------------------------------------------------------------------------
# Small free helpers
# ---------------------------------------------------------------------------


def _engine_name(config: ExperimentConfig) -> str:
    from llenergymeasure.config.ssot import engine_str

    return engine_str(config.engine)


def _is_server_capable(engine: Any) -> bool:
    return all(
        callable(getattr(engine, name, None)) for name in ("launch", "await_ready", "shutdown")
    )


def _warmup_provenance(warmup: ServerWarmupResult | None) -> ServerWarmupProvenance | None:
    """Project a level's warmup outcome into the per-window provenance sub-model."""
    if warmup is None:
        return None
    return ServerWarmupProvenance(
        mode=warmup.mode,
        converged=warmup.converged,
        timed_out=warmup.timed_out,
        elapsed_s=warmup.elapsed_s,
    )


def _phase_energy_j(core: MeasuredWindowCore | None) -> float | None:
    """Phase GPU energy (J) from the bracket's energy tracker, or None if unmeasured.

    Uses the energy MEASUREMENT (the NVML-integrated total the tracker returns),
    not the thermal timeseries integral: a short phase (a fast launch or drain)
    yields too few thermal samples to integrate, but the energy tracker still
    reports a total. Never 0.0 stands in for unmeasured - a missing measurement is
    None.
    """
    if core is None:
        return None
    em = getattr(core, "energy_measurement", None)
    total = getattr(em, "total_j", None) if em is not None else None
    return float(total) if total is not None else None


def _sum_optional(values: Iterable[float | None]) -> float | None:
    """Sum the non-None values; None when every value was None (never a false 0.0)."""
    total = 0.0
    seen = False
    for value in values:
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def _core_energy_j(core: MeasuredWindowCore | None) -> float | None:
    """Raw window energy (J) from a measured core's power series (windowing reuse).

    Mirrors the window manager's full-span energy statistic (clean the samples,
    trapezoidally integrate the summed-across-GPU power series), so a failed
    level's preserved-but-unbookkept cores still contribute their measured GPU
    energy to the session total. Reuses windowing.py's cleaner and the nvml
    integrator - the same REUSE BINDING the window manager follows - so no
    integration math is duplicated.
    """
    from llenergymeasure.energy.nvml import integrate_power_samples
    from llenergymeasure.harness.windowing import _clean_samples

    samples = list(getattr(core, "timeseries_samples", []) or []) if core is not None else []
    if len(samples) < 2:
        return None
    cleaned = _clean_samples(samples)
    if len(cleaned) < 2:
        return None
    return sum(integrate_power_samples(cleaned).values())


def _sum_window_energy(levels: list[ServerLevelResult]) -> float | None:
    total = 0.0
    seen = False
    for level in levels:
        for window in level.windows:
            energy = window.window.window_energy_j
            if energy is not None:
                total += energy
                seen = True
        # A failed level's preserved raw cores (no ServerWindowResult) still carry
        # measured GPU energy: fold it in so the session total does not silently
        # drop it (contract 3 / O7.4).
        for core in level.completed_cores:
            energy = _core_energy_j(core)
            if energy is not None:
                total += energy
                seen = True
    return total if seen else None


def _run_sync(awaitable: Awaitable[Any]) -> None:
    """Run a coroutine to completion for best-effort transport teardown.

    Used only on the cleanup path (closing the httpx client). A fresh event loop
    keeps it independent of any loop ``run()`` already finished with. Best-effort:
    a missing loop context (already-closed awaitable, etc.) is swallowed.
    """
    with contextlib.suppress(RuntimeError):
        asyncio.run(_as_none(awaitable))


async def _as_none(awaitable: Awaitable[Any]) -> None:
    await awaitable
