"""Server-mode per-window metric derivation and the SLO overlay (SM12).

Derives a window's server-distinct metrics from the SAME per-request records that
feed ``requests.parquet`` (:mod:`llenergymeasure.results.request_log`), with NO
re-sampling: TTFT / ITL / TPOT / e2e latency percentiles, request throughput,
completion / error / timeout counts, and the post-hoc SLO overlay (attainment,
slo_pass, goodput, energy-at-operating-point). It lives in the results layer beside
``request_log`` so both call sites reach it without crossing a layer boundary: the
study layer calls it at persist time, and an offline consumer calls it over a loaded
bundle's ``requests.parquet`` (via :func:`request_log.rows_from_parquet`) to
re-judge a window against any SLO bounds - the O5.3 re-judgeable promise.

Design invariants:

- PURE. Every function here is a pure function of its inputs; it constructs no
  backend, opens no transport, and reads no clock. The metric derivation is a pure
  function of (records, slo bounds): no SLO value leaks into the window's identity
  or its physical measurement, so the measurement fields are byte-stable across
  SLO re-judgement and only the ``slo`` overlay moves (O5.3).

- DENOMINATORS ARE REUSED, NEVER RECOMPUTED. The span-clipped token throughput
  (``token_throughput_tokens_s``, the result's ``avg_tokens_per_second``) and the
  window's J/token operating point are passed in already-correct from the window
  manager's span-clipped bookkeeping; this module never re-clips receipts.

- POPULATION. A window's rows are ISSUE-partitioned and receipt-unclipped
  (request_log CLIPPING SEMANTICS), so the steady-state population is the rows with
  ``in_measurement_window`` set (issued in ``[span_start, span_end]``); this drops
  the level's prospective ramp (window 0's ``is_ramp`` rows). Latency and attainment
  further filter to ``status == "ok"`` (completed) and are DRAIN-INCLUSIVE: a
  straddler that completed past ``span_end`` keeps its full latency (D7). Token and
  energy denominators stay span-clipped (they are passed in, not recomputed here).

ATTAINMENT SEMANTICS (O5.2, verdict-bearing). Attainment is a PER-REQUEST joint
evaluation, not a per-metric percentile: the fraction of COMPLETED (``status ==
"ok"``) requests that met ALL configured bounds AT ONCE - a request counts iff its
TTFT is within ``ttft_ms`` AND its per-request TPOT is within ``tpot_ms``. The
window verdict ``slo_pass`` is ``attainment >= percentile`` (the shared SLO tail
quantile): at least ``percentile`` of served requests inside the bound, the MLPerf
server-scenario reading. This joint per-request form is STRICTER than checking each
metric's marginal percentile separately (a request can pass TTFT yet fail TPOT), and
that strictness is deliberate - the ruling is "meeting ALL slo bounds". Goodput is
the DERIVED column ``attainment x throughput`` over the SAME records, never sampled
apart. ``ttft_at_percentile_ms`` / ``tpot_at_percentile_ms`` are reported for
cross-checking the observed tail against the bound.

FINISH-REASON RULING (v0.7). A completed request whose stream stopped at its
output-token budget (``finish_reason == "length"``) is attainment-ELIGIBLE: it is a
normal completion (``status == "ok"``), and the workload fixes the output budget, so
a length stop is a served request, not a failure - the MLPerf convention constrains
output length in the workload spec rather than disqualifying the query. Such
completions are counted and evaluated against the bounds like any other, and their
tally is DISCLOSED in ``length_truncated_count`` so a caller can re-segment if a
future workload wants a different reading.

TPOT-UNDEFINED RULING. A completed request with fewer than two output tokens has no
inter-token interval, so its per-request TPOT is undefined; it passes the TPOT bound
VACUOUSLY (there is no per-output-token latency to exceed) and contributes no sample
to the TPOT percentiles. A request missing a TTFT under a configured TTFT bound
FAILS (a completed request that never delivered a first token cannot be shown to
have met the bound).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llenergymeasure.domain.experiment import (
    LatencyPercentiles,
    ServerSloEvaluation,
    ServerWindowMetrics,
)
from llenergymeasure.results.request_log import (
    REQUEST_STATUS_ERROR,
    REQUEST_STATUS_OK,
    REQUEST_STATUS_TIMEOUT,
    RequestLogRow,
)

#: OpenAI finish-reason marking an output-budget (max_tokens) truncation.
FINISH_REASON_LENGTH = "length"

__all__ = [
    "FINISH_REASON_LENGTH",
    "SloBounds",
    "derive_server_window_metrics",
    "evaluate_slo",
]


@dataclass(frozen=True)
class SloBounds:
    """The SLO bounds a window is judged against (a plain post-hoc overlay input).

    Mirrors ``config.models.SloConfig`` but is a results-layer plain value so the
    pure derivation needs no config import and an offline re-judger can pass any
    bounds without constructing a config. ``ttft_ms`` / ``tpot_ms`` are None when
    unbounded; ``percentile`` is the shared tail quantile the attainment verdict
    targets.
    """

    ttft_ms: float | None
    tpot_ms: float | None
    percentile: float


def _percentile(sorted_values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation (type-7 / numpy-default) percentile of a SORTED series.

    ``q`` is a fraction in [0, 1]. Returns None for an empty series and the sole
    value for a singleton, so the caller gets a truthful null rather than a zero for
    a window that produced no datum. Kept numpy-free and closed-form so the tail
    percentiles are hand-checkable in tests.
    """
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo >= n - 1:
        return float(sorted_values[-1])
    return float(sorted_values[lo] + frac * (sorted_values[lo + 1] - sorted_values[lo]))


def _percentiles(values: Sequence[float]) -> LatencyPercentiles:
    """Build the p50/p90/p99 block for one latency series (ms)."""
    ordered = sorted(float(v) for v in values)
    return LatencyPercentiles(
        p50_ms=_percentile(ordered, 0.5),
        p90_ms=_percentile(ordered, 0.9),
        p99_ms=_percentile(ordered, 0.99),
        samples=len(ordered),
    )


def _per_request_tpot_ms(row: RequestLogRow) -> float | None:
    """Per-request time-per-output-token (ms): decode span / (tokens - 1).

    None when the request delivered fewer than two tokens (no inter-token interval).
    Receipt times share the issuer's monotonic-seconds basis, so the difference is a
    real duration; scaled to ms.
    """
    times = row.output_token_times
    if len(times) < 2:
        return None
    return (times[-1] - times[0]) / (len(times) - 1) * 1000.0


def _inter_token_latencies_ms(row: RequestLogRow) -> list[float]:
    """Pooled inter-token latencies (ms): consecutive receipt differences.

    Approximate at fine grain (client-loop receipt jitter, request_log M2); reported
    for the tail-shape picture, never as an SLO denominator.
    """
    times = row.output_token_times
    return [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))]


def _request_meets_slo(row: RequestLogRow, bounds: SloBounds) -> bool:
    """Whether one COMPLETED request met ALL configured bounds jointly (O5.2).

    A missing TTFT under a configured TTFT bound FAILS; an undefined TPOT (fewer than
    two tokens) passes the TPOT bound VACUOUSLY (see module docstring).
    """
    if bounds.ttft_ms is not None and (row.ttft_ms is None or row.ttft_ms > bounds.ttft_ms):
        return False
    if bounds.tpot_ms is not None:
        tpot = _per_request_tpot_ms(row)
        if tpot is not None and tpot > bounds.tpot_ms:
            return False
    return True


def _completed_in_window(rows: Sequence[RequestLogRow]) -> list[RequestLogRow]:
    """The steady-state completed population: in-span and status == ok (drain-inclusive)."""
    return [r for r in rows if r.in_measurement_window and r.status == REQUEST_STATUS_OK]


def evaluate_slo(
    rows: Sequence[RequestLogRow],
    bounds: SloBounds,
    *,
    token_throughput_tokens_s: float | None,
    level_valid: bool,
) -> ServerSloEvaluation:
    """Judge a window against ``bounds`` - a PURE overlay over (rows, bounds) (O5.3).

    Usable at persist time and offline over a loaded ``requests.parquet``: the same
    rows judged against different bounds yield a different attainment/verdict while
    every physical measurement is untouched. ``token_throughput_tokens_s`` is the
    window's already-correct span-clipped client-token throughput (the result's
    ``avg_tokens_per_second``); it is only multiplied into goodput, never re-derived.

    An empty completed population (no request finished) yields ``attainment=None``
    (0/0 is undefined, not 0.0), a ``slo_pass=False`` verdict (a window that served
    nothing cannot pass), and ``goodput=None``.
    """
    completed = _completed_in_window(rows)
    n = len(completed)
    if n == 0:
        attainment: float | None = None
        slo_pass = False
        goodput: float | None = None
    else:
        passing = sum(1 for r in completed if _request_meets_slo(r, bounds))
        attainment = passing / n
        slo_pass = attainment >= bounds.percentile
        goodput = (
            attainment * token_throughput_tokens_s
            if token_throughput_tokens_s is not None
            else None
        )
    ttft_vals = sorted(r.ttft_ms for r in completed if r.ttft_ms is not None)
    tpot_vals = sorted(tpot for r in completed if (tpot := _per_request_tpot_ms(r)) is not None)
    return ServerSloEvaluation(
        ttft_bound_ms=bounds.ttft_ms,
        tpot_bound_ms=bounds.tpot_ms,
        percentile=bounds.percentile,
        ttft_at_percentile_ms=_percentile(ttft_vals, bounds.percentile),
        tpot_at_percentile_ms=_percentile(tpot_vals, bounds.percentile),
        attainment_fraction=attainment,
        slo_pass=slo_pass,
        goodput_tokens_s=goodput,
        energy_at_operating_point_valid=bool(slo_pass) and level_valid,
    )


def _sum_server_completion_tokens(rows: Sequence[RequestLogRow]) -> tuple[int, bool]:
    """Sum auxiliary server-reported completion tokens over rows; flag whether any was seen.

    Returns (total, any_reported) so a true 0 is distinguishable from 'no engine
    reported usage' (the divergence ratio is None in the latter case, never a false
    0.0).
    """
    total = 0
    seen = False
    for row in rows:
        if row.server_completion_tokens is not None:
            total += row.server_completion_tokens
            seen = True
    return total, seen


def derive_server_window_metrics(
    rows: Sequence[RequestLogRow],
    *,
    duration_s: float,
    token_throughput_tokens_s: float | None,
    j_per_token: float | None,
    level_valid: bool,
    slo: SloBounds | None,
) -> ServerWindowMetrics:
    """Derive a window's server metrics from its per-request rows (no re-sampling).

    Every field except ``slo`` is slo-INDEPENDENT: throughput, counts, and latency
    percentiles are pure functions of the rows and the passed-in span-clipped
    denominators. ``slo`` is None when ``slo`` bounds are None; otherwise it is the
    :func:`evaluate_slo` overlay. ``duration_s`` is the measured span duration;
    ``token_throughput_tokens_s`` and ``j_per_token`` are the window's already-clipped
    denominators (never recomputed here).
    """
    in_window = [r for r in rows if r.in_measurement_window]
    completed = [r for r in in_window if r.status == REQUEST_STATUS_OK]
    error_count = sum(1 for r in in_window if r.status == REQUEST_STATUS_ERROR)
    timeout_count = sum(1 for r in in_window if r.status == REQUEST_STATUS_TIMEOUT)
    total = len(in_window)

    request_throughput = (len(completed) / duration_s) if duration_s > 0 else None
    completion_rate = (len(completed) / total) if total > 0 else None
    length_truncated = sum(1 for r in completed if r.finish_reason == FINISH_REASON_LENGTH)

    ttft = _percentiles([r.ttft_ms for r in completed if r.ttft_ms is not None])
    e2e = _percentiles([r.e2e_latency_ms for r in completed if r.e2e_latency_ms is not None])
    tpot = _percentiles([tpot for r in completed if (tpot := _per_request_tpot_ms(r)) is not None])
    itl = _percentiles([d for r in completed for d in _inter_token_latencies_ms(r)])

    # Divergence disclosure over the SAME completed population, so the two totals are
    # comparable (both whole-request receipt-unclipped counts, not the span-clipped
    # energy denominator): server self-reported completion tokens / client-counted.
    client_tokens = sum(r.client_output_tokens for r in completed)
    server_tokens, any_server = _sum_server_completion_tokens(completed)
    token_ratio = (server_tokens / client_tokens) if (any_server and client_tokens > 0) else None

    slo_eval = (
        evaluate_slo(
            rows,
            slo,
            token_throughput_tokens_s=token_throughput_tokens_s,
            level_valid=level_valid,
        )
        if slo is not None
        else None
    )

    return ServerWindowMetrics(
        request_throughput_req_s=request_throughput,
        completed_count=len(completed),
        error_count=error_count,
        timeout_count=timeout_count,
        completion_rate=completion_rate,
        length_truncated_count=length_truncated,
        ttft=ttft,
        tpot=tpot,
        itl=itl,
        e2e=e2e,
        energy_at_operating_point_j_per_token=j_per_token,
        server_reported_client_token_ratio=token_ratio,
        slo=slo_eval,
    )
