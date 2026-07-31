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
   seam so the energy denominator and the stability gate receive counts. v0.7
   wires the INTERIM count from what the transport exposes (the server-reported
   ``usage.completion_tokens``); canonical client-side tokenisation +
   ``requests.parquet`` are SM11. The limitation is stamped loudly in the result
   (``token_counting``) and the module constant below.
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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llenergymeasure.harness.server_warmup import (
    ServerWarmup,
    ServerWarmupResult,
    WarmupTrafficError,
    build_probe_request,
    describe_server_warmup_protocol,
)
from llenergymeasure.harness.traffic import OpenLoopPoissonSource, RequestShape
from llenergymeasure.harness.window_manager import (
    ABORTED_LEVEL_ATTR,
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

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig, TrafficConfig
    from llenergymeasure.config.runner_spec import RunnerSpec
    from llenergymeasure.engines.protocol import ServerCapable
    from llenergymeasure.harness.bracket import MeasuredWindowCore
    from llenergymeasure.harness.traffic import RequestRecord, ShapeSource, TrafficSource, Transport
    from llenergymeasure.infra.server_lifecycle import ServerHandle, ServerPlacement
    from llenergymeasure.study.runner import StudyRunner

logger = logging.getLogger(__name__)

#: The OpenAI-compatible completions endpoint every v0.7 server engine exposes.
#: vLLM (``/v1/completions``) and TRT-LLM (``/v1/completions``) agree, and
#: transformers is E5-gated out of server mode, so a session-layer constant is
#: the minimal faithful encoding; a future non-OpenAI engine would promote this
#: to a ``ServerCapable`` member (flagged).
SERVING_COMPLETIONS_PATH = "/v1/completions"

#: Loud provenance stamp for the INTERIM v0.7 token denominator: client-side
#: canonical counting + requests.parquet are SM11 (O8). Until then the energy
#: denominator and stability gate consume the SERVER-REPORTED completion-token
#: count (auxiliary per O8), which the driver reads off each response.
TOKEN_COUNTING_SERVER_REPORTED = (
    "server_reported_usage_interim: the J/token denominator and stability gate "
    "consume server-reported completion-token counts (usage.completion_tokens), "
    "an INTERIM stand-in for the SM11 client-side canonical count. Treat J/token "
    "as provisional until SM11 lands client-side tokenisation + requests.parquet."
)

#: Interrupt-watcher poll cadence (seconds): how promptly a mid-session SIGINT
#: (which the study runner's handler records on its interrupt event) is noticed
#: and cancels the driving task.
_INTERRUPT_POLL_INTERVAL_S = 0.2

__all__ = [
    "SERVING_COMPLETIONS_PATH",
    "TOKEN_COUNTING_SERVER_REPORTED",
    "ServerLevelResult",
    "ServerSession",
    "ServerSessionError",
    "ServerSessionResult",
    "ServerWindowResult",
    "server_reported_token_receipts",
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
    """

    level_index: int
    spec: WindowSpec | None
    windows: list[ServerWindowResult]
    validation: LevelValidation | None
    warmup: ServerWarmupResult | None
    invalid_reason: str | None
    aborted_window_index: int | None = None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None and self.validation is not None and self.validation.valid


@dataclass
class ServerSessionResult:
    """One server session's product: the N window results over one server lifetime.

    Not an ``ExperimentResult`` (offline shape) - server metrics derivation is
    SM12 and per-window persistence is SM10 - so this rides the runner return seam
    as its own type. ``valid`` is True iff at least one level passed its stability
    gate. ``token_counting`` carries the loud SM11-interim limitation stamp.
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

    @property
    def valid(self) -> bool:
        return any(level.valid for level in self.levels)

    @property
    def window_count(self) -> int:
        return sum(len(level.windows) for level in self.levels)


# ---------------------------------------------------------------------------
# Interim token-receipt seam wiring (contract 2). Reads the SERVER-REPORTED
# completion-token count off the transport's captured response and stamps every
# token at the request's completion time (E2's completion-timestamp attribution,
# the request-granular TokenReceiptFn form SM7 documents).
# ---------------------------------------------------------------------------


def server_reported_token_receipts(record: RequestRecord) -> Sequence[float]:
    """Return a request's output-token receipt timestamps (interim, server-reported).

    All of a request's ``usage.completion_tokens`` tokens are stamped at its
    ``completed_at`` (request-granular attribution = E2's completion-timestamp
    rule). Returns ``()`` when the request never completed, raised, or the
    response carried no usable ``usage.completion_tokens`` - so an unformable
    denominator degrades to the gate's invalid-with-reason path rather than a lie.
    """
    completed_at = record.completed_at
    result = record.result
    if completed_at is None or record.error is not None or not isinstance(result, dict):
        return ()
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return ()
    n = usage.get("completion_tokens")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        return ()
    return (completed_at,) * n


# ---------------------------------------------------------------------------
# Request-shape source: OpenAI completions payloads drawn from the config's
# dataset prompts (SM11 refines the encoding; this is the minimal faithful shape
# so warmup + measurement + probe all drive the path they measure).
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
            try:
                # INLINE await: the AbortedLevel the manager attaches to a failing
                # level's exception survives, because there is no intervening Task
                # boundary to substitute a fresh CancelledError (contract 3).
                outcome = await manager.run_level(level_index, level)
                outcomes_out.append(outcome)
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
                raise
            except BaseException as exc:
                # Any other level failure is recorded and the session continues.
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


class ServerSession:
    """Online-serving measurement session: launch -> warm up -> windows -> shutdown.

    Constructed like the offline sessions (runner + config + identity + the
    resolved runner spec); ``engine`` is injectable for tests, else resolved from
    the config. The seam-building methods (``_make_shape_source`` /
    ``_make_transport`` / ``_make_energy_sink`` / ``_make_sampler_factory``) are
    small and overridable so unit tests exercise ``run()`` with fakes.
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
    ) -> None:
        self._runner = runner
        self.config = config
        self.spec = spec
        self.config_hash = config_hash
        self.cycle = cycle
        self.index = index
        self._engine = engine
        self._torn_down = False
        self._exp_start = 0.0
        # Populated in __enter__.
        self._handle: ServerHandle | None = None
        self._transport: Transport | None = None
        self._shape_source: ShapeSource | None = None
        self._warmup: ServerWarmup | None = None

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

            self._shape_source = self._make_shape_source()
            placement = self._make_placement()
            self._handle = engine.launch(config, placement)

            probe = build_probe_request(
                self._shape_source, path=SERVING_COMPLETIONS_PATH, method="POST"
            )
            engine.await_ready(self._handle, probe, timeout=self._readiness_timeout())

            self._transport = self._make_transport(self._handle.base_url)

            # The server is up: mark running (a launch/readiness failure above
            # never reaches here, so a failed launch is recorded as failed, not
            # left running).
            runner.manifest.mark_running(self.config_hash, self.cycle)
        except BaseException:
            # Acquisition failed before the `with` body: release the partially
            # launched server and re-raise (a failed launch never leaks).
            self._torn_down = True
            self._cleanup()
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

        return self._finalise(outcomes, failures, status=status, interrupted=interrupted)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
        if self._torn_down:
            return None
        self._torn_down = True
        self._cleanup()
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
            token_counting=TOKEN_COUNTING_SERVER_REPORTED,
            total_window_energy_j=total_energy,
            elapsed_s=elapsed,
            aborted=aborted,
            abort_reason=abort_reason,
        )

        if interrupted:
            # Manifest handling is the sweep loop's (mark_interrupted); do not
            # resolve the entry here.
            self._end_progress(result, ok=result.valid)
            return result

        if not result.valid:
            message = abort_reason or self._invalid_message(levels)
            error_type = "WarmupTrafficError" if aborted else "ServerSessionInvalid"
            self._runner.manifest.mark_failed(self.config_hash, self.cycle, error_type, message)
            self._end_progress(result, ok=False)
            return {"type": error_type, "message": message}

        # Valid session: mark completed. SM10 wires the per-window bundle files;
        # the result_file stays empty until then (no bundle persisted at SM9).
        self._runner.manifest.mark_completed(
            self.config_hash,
            self.cycle,
            "",
            elapsed_seconds=elapsed,
            energy_joules=total_energy,
        )
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
            # A failed level keeps its cleanly-closed cores (O7.4) but not full
            # per-window bookkeeping (the traffic report is gone once the issuer
            # task was cancelled); SM10 owns partial-window persistence.
            by_level[failure.level_index] = ServerLevelResult(
                level_index=failure.level_index,
                spec=None,
                windows=[],
                validation=None,
                warmup=warmup,
                invalid_reason=failure.reason,
                aborted_window_index=failure.aborted_window_index,
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
            self._runner._progress if self._runner is not None else None,
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
        traffic = self._traffic()
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
        return [
            LevelPlan(
                spec=spec,
                traffic_source=source,
                transport=transport,
                token_receipt_fn=server_reported_token_receipts,
            )
        ]

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

    def _cleanup(self) -> None:
        """Shut the server down + drain-finalise; runs exactly once from __exit__.

        Idempotent and best-effort (it runs on the normal, SIGINT, and exception
        paths). The engine's ``shutdown`` is itself idempotent + leak-free (SM6),
        so a double invocation is a no-op; the transport's connection pool is
        closed too so a launched-but-never-run session leaks nothing.
        """
        handle = self._handle
        if handle is not None:
            with contextlib.suppress(Exception):
                self._shutdown_handle(handle)
        transport = self._transport
        if transport is not None:
            aclose = getattr(transport, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    _run_sync(aclose())

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
# Small free helpers
# ---------------------------------------------------------------------------


def _engine_name(config: ExperimentConfig) -> str:
    from llenergymeasure.config.ssot import engine_str

    return engine_str(config.engine)


def _is_server_capable(engine: Any) -> bool:
    return all(
        callable(getattr(engine, name, None)) for name in ("launch", "await_ready", "shutdown")
    )


def _sum_window_energy(levels: list[ServerLevelResult]) -> float | None:
    total = 0.0
    seen = False
    for level in levels:
        for window in level.windows:
            energy = window.window.window_energy_j
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
