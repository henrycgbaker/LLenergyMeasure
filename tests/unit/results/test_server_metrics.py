"""Unit tests for the pure server-metric derivation and the SLO overlay (SM12).

Host-only and backend-free: every test builds :class:`RequestLogRow` fixtures by
hand and asserts hand-computed percentiles, attainment, and goodput. The re-judge
tests pin O5.3 (same rows, different bounds -> different verdict, identical
measurement fields); the edge cases pin empty / all-failed / boundary windows.
"""

from __future__ import annotations

import pytest

from llenergymeasure.results.request_log import (
    REQUEST_STATUS_ERROR,
    REQUEST_STATUS_OK,
    REQUEST_STATUS_TIMEOUT,
    RequestLogRow,
)
from llenergymeasure.results.server_metrics import (
    SloBounds,
    derive_server_window_metrics,
    evaluate_slo,
)


def _row(
    index: int,
    *,
    ttft_ms: float | None = 10.0,
    token_times: list[float] | None = None,
    status: str = REQUEST_STATUS_OK,
    finish_reason: str | None = "stop",
    in_window: bool = True,
    is_ramp: bool = False,
    completed_in_drain: bool = False,
    server_completion_tokens: int | None = None,
    e2e_latency_ms: float | None = 100.0,
) -> RequestLogRow:
    """Build one request-log row with only the SM12-relevant fields set.

    ``token_times`` defaults to a single receipt (so client_output_tokens == 1);
    pass a multi-element list to form a per-request TPOT / ITL.
    """
    times = token_times if token_times is not None else [0.0]
    return RequestLogRow(
        request_index=index,
        issued_at=0.0,
        dispatched_at=0.0,
        first_token_at=(times[0] if times else None),
        completed_at=(times[-1] + 0.01 if times else None),
        ttft_ms=ttft_ms,
        e2e_latency_ms=e2e_latency_ms,
        client_output_tokens=len(times),
        server_prompt_tokens=None,
        server_completion_tokens=server_completion_tokens,
        status=status,
        finish_reason=finish_reason,
        level_index=0,
        window_index=0,
        in_measurement_window=in_window,
        is_ramp=is_ramp,
        completed_in_drain=completed_in_drain,
        output_token_times=times,
    )


def _derive(rows, *, slo=None, span_start=0.0, span_end=10.0, j_per_token=0.5, level_valid=True):
    return derive_server_window_metrics(
        rows,
        span_start=span_start,
        span_end=span_end,
        j_per_token=j_per_token,
        level_valid=level_valid,
        slo=slo,
    )


# ---------------------------------------------------------------------------
# Percentiles + counts (slo-independent measurement)
# ---------------------------------------------------------------------------


class TestPercentilesAndCounts:
    def test_ttft_percentiles_linear_interpolation(self) -> None:
        rows = [_row(i, ttft_ms=v) for i, v in enumerate([10.0, 20.0, 30.0, 40.0])]
        m = _derive(rows)
        # type-7 linear interpolation over [10,20,30,40].
        assert m.ttft.p50_ms == 25.0
        assert m.ttft.p90_ms == pytest.approx(37.0)
        assert m.ttft.p99_ms == pytest.approx(39.7)
        assert m.ttft.samples == 4

    def test_single_sample_percentile_is_the_value(self) -> None:
        m = _derive([_row(0, ttft_ms=12.5)])
        assert m.ttft.p50_ms == 12.5
        assert m.ttft.p99_ms == 12.5
        assert m.ttft.samples == 1

    def test_empty_window_yields_null_percentiles_and_zero_counts(self) -> None:
        m = _derive([])
        assert m.completed_count == 0
        assert m.completion_rate is None
        assert m.request_throughput_req_s == 0.0  # 0 completed / 10 s
        assert m.ttft.p50_ms is None
        assert m.ttft.samples == 0
        assert m.slo is None

    def test_tpot_and_itl_from_token_times(self) -> None:
        # Two requests, receipts 0.0/0.1/0.2 -> tpot 100 ms, itl 100 ms x2 each.
        rows = [_row(i, token_times=[0.0, 0.1, 0.2]) for i in range(2)]
        m = _derive(rows)
        assert m.tpot.p50_ms == 100.0
        assert m.tpot.samples == 2
        assert m.itl.samples == 4  # 2 requests x 2 intervals
        assert m.itl.p50_ms == 100.0

    def test_counts_exclude_ramp_and_split_status(self) -> None:
        rows = [
            _row(0, status=REQUEST_STATUS_OK),
            _row(1, status=REQUEST_STATUS_ERROR),
            _row(2, status=REQUEST_STATUS_TIMEOUT),
            _row(3, status=REQUEST_STATUS_OK, is_ramp=True, in_window=False),  # ramp: excluded
        ]
        m = _derive(rows)
        assert m.completed_count == 1
        assert m.error_count == 1
        assert m.timeout_count == 1
        # completion_rate over the 3 in-window requests (ramp excluded).
        assert m.completion_rate == 1 / 3

    def test_request_throughput_uses_span_duration(self) -> None:
        rows = [_row(i) for i in range(6)]
        assert _derive(rows, span_start=0.0, span_end=3.0).request_throughput_req_s == 2.0

    def test_length_truncation_is_disclosed_and_eligible(self) -> None:
        rows = [
            _row(0, finish_reason="stop"),
            _row(1, finish_reason="length"),
            _row(2, finish_reason="length"),
        ]
        m = _derive(rows, slo=SloBounds(ttft_ms=1000.0, tpot_ms=None, percentile=0.99))
        assert m.length_truncated_count == 2
        # A length stop is a normal completion -> still attainment-eligible (all 3
        # meet the generous ttft bound).
        assert m.slo is not None
        assert m.slo.attainment_fraction == 1.0

    def test_server_client_token_ratio(self) -> None:
        rows = [_row(i, token_times=[0.0, 0.1], server_completion_tokens=3) for i in range(2)]
        # client: 2 tokens each x2 = 4; server: 3 each x2 = 6 -> 1.5.
        m = _derive(rows)
        assert m.server_reported_client_token_ratio == 1.5

    def test_token_ratio_none_when_no_server_usage(self) -> None:
        rows = [_row(i, server_completion_tokens=None) for i in range(2)]
        assert _derive(rows).server_reported_client_token_ratio is None


# ---------------------------------------------------------------------------
# SLO overlay (O5.2 attainment, O5.3 pure re-judgement)
# ---------------------------------------------------------------------------


class TestSloOverlay:
    def test_no_slo_leaves_overlay_none(self) -> None:
        assert _derive([_row(0)], slo=None).slo is None

    def test_attainment_joint_over_all_bounds(self) -> None:
        # ttft 10/20/30/40; each request 3 tokens spaced 0.1 s -> tpot 100 ms.
        rows = [
            _row(i, ttft_ms=t, token_times=[0.0, 0.1, 0.2]) for i, t in enumerate((10, 20, 30, 40))
        ]
        m = _derive(rows, slo=SloBounds(ttft_ms=35.0, tpot_ms=150.0, percentile=0.75))
        # Joint: ttft<=35 passes 3/4 (40 fails); tpot<=150 passes all -> 3/4.
        assert m.slo.attainment_fraction == 0.75
        assert m.slo.slo_pass is True  # 0.75 >= 0.75 (exactly at bound)
        # Direct join: the 3 qualifying requests each delivered 3 in-span tokens (9
        # total) over the 10 s span -> 0.9. The failing (ttft=40) request's 3 tokens
        # are absent from the numerator.
        assert m.slo.goodput_tokens_s == pytest.approx(0.9)
        assert m.slo.energy_at_operating_point_valid is True

    def test_attainment_boundary_exactly_at_bound_passes(self) -> None:
        # A request whose ttft equals the bound MEETS it (<=), not violates.
        rows = [_row(i, ttft_ms=50.0) for i in range(4)]
        m = _derive(rows, slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.99))
        assert m.slo.attainment_fraction == 1.0
        assert m.slo.slo_pass is True

    def test_tpot_only_bound_joint_failure(self) -> None:
        # ttft passes (bound generous); tpot: one request slow (300 ms) fails.
        rows = [
            _row(0, ttft_ms=5.0, token_times=[0.0, 0.1, 0.2]),  # tpot 100
            _row(1, ttft_ms=5.0, token_times=[0.0, 0.3, 0.6]),  # tpot 300
        ]
        m = _derive(rows, slo=SloBounds(ttft_ms=1000.0, tpot_ms=150.0, percentile=0.99))
        assert m.slo.attainment_fraction == 0.5

    def test_single_token_request_passes_tpot_vacuously(self) -> None:
        # One token -> no inter-token interval -> the tpot bound cannot be violated.
        rows = [_row(0, ttft_ms=5.0, token_times=[0.0])]
        m = _derive(rows, slo=SloBounds(ttft_ms=1000.0, tpot_ms=1.0, percentile=0.99))
        assert m.slo.attainment_fraction == 1.0
        assert m.slo.tpot_at_percentile_ms is None  # no TPOT sample formed
        # A vacuous tpot pass is a qualifying request: its 1 in-span token enters the
        # goodput numerator (1 token / 10 s span).
        assert m.slo.goodput_tokens_s == pytest.approx(0.1)

    def test_missing_ttft_under_bound_fails(self) -> None:
        rows = [_row(0, ttft_ms=None)]  # completed but no first token observed
        m = _derive(rows, slo=SloBounds(ttft_ms=100.0, tpot_ms=None, percentile=0.99))
        assert m.slo.attainment_fraction == 0.0
        assert m.slo.slo_pass is False

    def test_all_completed_but_all_violate(self) -> None:
        rows = [_row(i, ttft_ms=500.0) for i in range(4)]
        m = _derive(rows, slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.99))
        assert m.slo.attainment_fraction == 0.0
        assert m.slo.slo_pass is False
        # Non-empty completed population, zero qualifying -> truthful 0.0 (no tokens
        # in the numerator), NOT None.
        assert m.slo.goodput_tokens_s == 0.0
        assert m.slo.energy_at_operating_point_valid is False

    def test_all_failed_window_no_completions(self) -> None:
        rows = [
            _row(0, status=REQUEST_STATUS_ERROR),
            _row(1, status=REQUEST_STATUS_TIMEOUT),
        ]
        m = _derive(rows, slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.99))
        # No completed request: attainment undefined (not 0.0), verdict is a clear
        # fail, and there is no valid operating point.
        assert m.slo.attainment_fraction is None
        assert m.slo.slo_pass is False
        assert m.slo.goodput_tokens_s is None
        assert m.slo.energy_at_operating_point_valid is False
        assert m.completion_rate == 0.0

    def test_energy_operating_point_invalid_when_level_invalid(self) -> None:
        rows = [_row(i, ttft_ms=10.0) for i in range(4)]
        m = _derive(
            rows,
            slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.99),
            level_valid=False,
        )
        assert m.slo.slo_pass is True
        # slo_pass True but the stability gate failed -> not a valid operating point.
        assert m.slo.energy_at_operating_point_valid is False

    def test_goodput_none_when_span_duration_non_positive(self) -> None:
        # A degenerate (zero-length) span has no divisor -> goodput is None even though
        # the requests qualify; attainment is span-independent and still resolves.
        rows = [_row(i, ttft_ms=10.0) for i in range(4)]
        m = _derive(
            rows,
            slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.99),
            span_start=5.0,
            span_end=5.0,
        )
        assert m.slo.attainment_fraction == 1.0
        assert m.slo.goodput_tokens_s is None

    def test_goodput_none_when_completed_population_empty(self) -> None:
        # No completed request at all -> attainment undefined and goodput None (0/0),
        # distinct from the truthful-0.0 zero-qualifying case.
        rows = [_row(0, status=REQUEST_STATUS_ERROR, token_times=[0.0, 0.1, 0.2])]
        m = _derive(rows, slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.99))
        assert m.slo.attainment_fraction is None
        assert m.slo.goodput_tokens_s is None


class TestRejudgeability:
    def test_same_rows_different_bounds_move_only_the_overlay(self) -> None:
        """O5.3: re-judging changes attainment but not any measurement field."""
        rows = [_row(i, ttft_ms=t, token_times=[0.0, 0.1]) for i, t in enumerate((10, 20, 30, 40))]
        lenient = _derive(rows, slo=SloBounds(ttft_ms=100.0, tpot_ms=None, percentile=0.99))
        strict = _derive(rows, slo=SloBounds(ttft_ms=15.0, tpot_ms=None, percentile=0.99))

        # The verdict moves with the bounds.
        assert lenient.slo.attainment_fraction == 1.0
        assert strict.slo.attainment_fraction == 0.25  # only ttft=10 meets <=15

        # Every slo-INDEPENDENT measurement field is identical across the re-judge.
        assert lenient.model_dump(exclude={"slo"}) == strict.model_dump(exclude={"slo"})

    def test_evaluate_slo_matches_derive_overlay(self) -> None:
        rows = [_row(i, ttft_ms=t) for i, t in enumerate((10, 20, 30, 40))]
        bounds = SloBounds(ttft_ms=25.0, tpot_ms=None, percentile=0.5)
        overlay = evaluate_slo(rows, bounds, span_start=0.0, span_end=10.0, level_valid=True)
        derived = _derive(rows, slo=bounds)
        assert derived.slo == overlay
        # ttft<=25 passes 2/4 = 0.5; at percentile 0.5 -> pass.
        assert overlay.attainment_fraction == 0.5
        assert overlay.slo_pass is True
        # Direct join: the 2 qualifying requests each delivered 1 in-span token (2
        # total) over the 10 s span -> 0.2.
        assert overlay.goodput_tokens_s == pytest.approx(0.2)


class TestGoodputDirectJoin:
    """Ground-truth for the literature-exact direct-join goodput (O5.2, section 26).

    Each expectation is chosen so the two superseded formulas FAIL it: the old product
    (attainment x all-in-span-token throughput) and the halfway candidate (attainment x
    completed-only-token throughput). The mutation-bite proof relies on these values.
    """

    def test_failure_knee_excludes_failed_and_violating_tokens(self) -> None:
        # A qualifying completion (6 in-span tokens), a completed-but-SLO-violating
        # request (1 token), and a failed request that still delivered 4 tokens. Only
        # the qualifying request's tokens enter the numerator.
        rows = [
            _row(0, ttft_ms=10.0, token_times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),  # qualifies, 6
            _row(1, ttft_ms=500.0, token_times=[0.0]),  # completed, violates ttft, 1
            _row(2, ttft_ms=10.0, status=REQUEST_STATUS_ERROR, token_times=[0.0, 0.1, 0.2, 0.3]),
        ]
        m = _derive(rows, slo=SloBounds(ttft_ms=50.0, tpot_ms=None, percentile=0.5))
        # completed = {row0, row1}; qualifying = {row0}. attainment 1/2 = 0.5.
        assert m.slo.attainment_fraction == 0.5
        # Direct join = 6 qualifying in-span tokens / 10 s = 0.6.
        assert m.slo.goodput_tokens_s == pytest.approx(0.6)
        # Two-sided bite: the old product would credit the failed + violating tokens
        # through the all-token throughput factor (0.5 x 11/10 = 0.55) and the halfway
        # candidate the violating token through the completed-only factor (0.5 x 7/10 =
        # 0.35); neither equals 0.6.
        assert m.slo.goodput_tokens_s != pytest.approx(0.55)
        assert m.slo.goodput_tokens_s != pytest.approx(0.35)

    def test_drain_straddler_in_span_tokens_only(self) -> None:
        # A drain straddler contributes ONLY its in-span tokens to goodput while its
        # FULL latency judges its SLO compliance (D7 two-policy separation).
        plain = _row(0, ttft_ms=10.0, token_times=[0.0, 0.1])  # 2 in-span tokens, tpot 100 ms
        straddler = _row(
            1,
            ttft_ms=10.0,
            token_times=[1.0, 2.0, 3.0, 4.0, 30.0],  # 4 in-span (<=10), 1 drain-tail (30)
            completed_in_drain=True,
        )
        # Generous tpot bound: the straddler qualifies (full-series tpot 7250 ms <= 8000)
        # and contributes only its 4 in-span tokens (not 5).
        lenient = _derive(
            [plain, straddler], slo=SloBounds(ttft_ms=50.0, tpot_ms=8000.0, percentile=0.5)
        )
        assert lenient.slo.attainment_fraction == 1.0
        # (2 + 4) in-span tokens / 10 s = 0.6; counting the drain-tail token would give 0.7.
        assert lenient.slo.goodput_tokens_s == pytest.approx(0.6)

    def test_drain_straddler_judged_on_full_latency(self) -> None:
        # Same straddler, tpot bound BETWEEN its clipped-series tpot (1000 ms) and its
        # full-series tpot (7250 ms): judged on the FULL series it VIOLATES, so it drops
        # out of both attainment and goodput.
        plain = _row(0, ttft_ms=10.0, token_times=[0.0, 0.1])
        straddler = _row(
            1,
            ttft_ms=10.0,
            token_times=[1.0, 2.0, 3.0, 4.0, 30.0],
            completed_in_drain=True,
        )
        m = _derive([plain, straddler], slo=SloBounds(ttft_ms=50.0, tpot_ms=5000.0, percentile=0.5))
        # Only the plain request qualifies -> attainment 1/2, goodput = 2 tokens / 10 s.
        assert m.slo.attainment_fraction == 0.5
        assert m.slo.goodput_tokens_s == pytest.approx(0.2)


class TestModeInapplicableStableSchema:
    """D8: the server-distinct fields stay None for offline (mode-inapplicable both ways)."""

    def test_offline_result_has_no_server_fields_and_real_tokens(self) -> None:
        from tests.conftest import make_result

        offline = make_result()  # serving_mode defaults to "offline"
        assert offline.serving_mode == "offline"
        # The server-distinct blocks are absent; serving_mode is the discriminator.
        assert offline.server is None
        assert offline.server_metrics is None
        # Offline keeps its real prefill/total token counts (never nulled).
        assert offline.input_tokens == 800
        assert offline.total_tokens == 1000
        # Serialised offline result.json carries the additive fields as null, coherent
        # with the existing server/session null convention (offline byte-stability).
        dumped = offline.model_dump(mode="json")
        assert dumped["server_metrics"] is None
        assert dumped["server"] is None
