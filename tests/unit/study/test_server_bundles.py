"""End-to-end persistence tests for server-mode per-window bundles (SM10).

These drive a REAL :class:`WindowManager` through the REAL ``_drive_levels`` with
a fake engine / transport / energy sink, and assert the on-disk bundles across the
clean, mid-level-abort, and SIGINT variants. The empirical pins the verifier
re-runs live here: the clean close stamps the drain raws into every sibling; SIGINT
leaves the completed windows on disk with the drain null; the persisted window
energies sum to the in-memory session total; and a rate-only group folds into one
launch.

Host-only: NVML is not required (the energy sink and the phase brackets are
injected fakes with known energies), and the fake clock makes the async
orchestration instant and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.domain.bundle_artefacts import RESULT_FILENAME, SYSTEM_FILENAME
from llenergymeasure.domain.environment import (
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    EnvironmentSnapshot,
    GPUEnvironment,
)
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.harness.server_warmup import ServerWarmupResult
from llenergymeasure.harness.traffic import IssuerReport, RequestRecord, RequestShape
from llenergymeasure.harness.window_manager import LevelPlan, WindowSpec, WindowStopEvent
from llenergymeasure.study.manifest import ManifestWriter
from llenergymeasure.study.server_session import (
    ServerCell,
    ServerSession,
    ServerSessionResult,
)

# ---------------------------------------------------------------------------
# Fakes + builders
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


class FakeEngine:
    def __init__(self) -> None:
        self.launched = 0
        self.shutdowns = 0
        self.handle = SimpleNamespace(base_url="http://127.0.0.1:9", engine="vllm")

    def launch(self, config: Any, placement: Any) -> Any:
        self.launched += 1
        return self.handle

    def await_ready(self, handle: Any, probe: Any, *, timeout: float) -> None:
        return None

    def shutdown(self, handle: Any) -> None:
        self.shutdowns += 1


class FakePhaseBracket:
    """A phase bracket returning a core with a KNOWN energy-tracker total."""

    def __init__(self, energy_j: float) -> None:
        self._energy = energy_j

    def __enter__(self) -> FakePhaseBracket:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def finish(self) -> Any:
        return SimpleNamespace(
            energy_measurement=SimpleNamespace(total_j=self._energy),
            timeseries_samples=[],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )


class FakeWarmupHook:
    """Records a ServerWarmupResult (with known energy) per level, as SM8 would."""

    def __init__(self, energy_j: float = 5.0) -> None:
        self.results: list[ServerWarmupResult] = []
        self._energy = energy_j

    async def __call__(self, context: Any) -> None:
        self.results.append(
            ServerWarmupResult(
                level_index=context.level_index,
                mode="composite",
                converged=True,
                timed_out=False,
                elapsed_s=1.0,
                final_observables=None,
                pre_window_protocol="server warmup (test)",
                energy_j=self._energy,
            )
        )


class ProducingEnergySink:
    """Flat-power core spanning each window's [span_start, span_end] (window_energy = P*dt)."""

    def __init__(self, power_w: float = 100.0, samples: int = 41) -> None:
        self._power = power_w
        self._n = samples
        self._open_at: float | None = None

    def open_window(self, event: Any) -> None:
        self._open_at = event.monotonic_at

    def close_window(self, event: WindowStopEvent) -> Any:
        assert self._open_at is not None
        lo, hi = self._open_at, event.monotonic_at
        self._open_at = None
        samples = [
            PowerThermalSample(
                timestamp=lo + (hi - lo) * i / (self._n - 1), power_w=self._power, gpu_index=0
            )
            for i in range(self._n)
        ]
        return SimpleNamespace(
            timeseries_samples=samples, start_time=datetime.now(), end_time=datetime.now()
        )

    def abort_window(self, event: Any) -> None:
        self._open_at = None


class ClosePerWindowSink(ProducingEnergySink):
    """Like ProducingEnergySink but close_window raises on a chosen window index."""

    def __init__(self, raise_on: int, **kw: Any) -> None:
        super().__init__(**kw)
        self._raise_on = raise_on

    def close_window(self, event: WindowStopEvent) -> Any:
        if event.window_index == self._raise_on:
            self._open_at = None
            raise RuntimeError("close teardown failed")
        return super().close_window(event)


class FakeTrafficSource:
    def __init__(self, report: IssuerReport) -> None:
        self._report = report

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        await asyncio.sleep(0)
        return self._report


def _report_with_receipts(times: list[float]) -> tuple[IssuerReport, Any]:
    records = [
        RequestRecord(index=i, issued_at=t, request=RequestShape(index=i), completed_at=t + 0.01)
        for i, t in enumerate(times)
    ]
    receipts = {i: [t] for i, t in enumerate(times)}
    report = IssuerReport(
        records=records,
        issued_count=len(records),
        completed_count=len(records),
        cap_bound_fraction=0.0,
        issuance_duration_s=0.0,
        concurrency_cap=None,
    )
    return report, (lambda r: receipts[r.index])


# Fake-clock spans for 3 windows of duration=10 after a ramp of 30 (start 1000):
# [1030,1040], [1040,1050], [1050,1060]. Four token receipts per window, one in
# each k=4 sub-window quarter, so the level validates AND the within-window CoV
# diagnostic is formable.
_TOKEN_TIMES = [
    1031.0,
    1033.5,
    1036.0,
    1039.0,
    1041.0,
    1043.5,
    1046.0,
    1049.0,
    1051.0,
    1053.5,
    1056.0,
    1059.0,
]


def _server_config(rate: float = 10.0) -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="server",
        server={"traffic": {"rate": rate, "window_seconds": 10}},
        measurement={"baseline": {"enabled": False}},
    )


def _offline_config(model: str) -> ExperimentConfig:
    return ExperimentConfig(task={"model": model}, engine="vllm", serving_mode="offline")


def _srv_result(session_id: str, level_index: int, level_valid: bool, energy: float) -> Any:
    """A minimal server-mode ExperimentResult for cell-granular counting tests."""
    from llenergymeasure.domain.experiment import ExperimentResult, ServerWindowProvenance
    from llenergymeasure.domain.session import SessionBlock

    return ExperimentResult(
        experiment_id="x",
        declared_config_hash="h",
        serving_mode="server",
        input_tokens=0,
        output_tokens=1,
        total_tokens=1,
        total_energy_j=energy,
        total_inference_time_sec=1.0,
        avg_tokens_per_second=1.0,
        avg_energy_per_token_j=1.0,
        total_flops=0.0,
        start_time=datetime.now(),
        end_time=datetime.now(),
        session=SessionBlock(session_id=session_id, window_count=3),
        server=ServerWindowProvenance(
            level_index=level_index,
            window_index=0,
            level_window_count=3,
            level_valid=level_valid,
            pre_window_protocol="p",
            attribution_policy="a",
        ),
    )


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        hardware=EnvironmentMetadata(
            gpu=GPUEnvironment(name="fake-gpu", vram_total_mb=1.0),
            cuda=CUDAEnvironment(driver_supported_version="12.0", driver_version="999"),
            cpu=CPUEnvironment(platform="Linux"),
            collected_at=datetime.now(),
        ),
        python_version="3.12.0",
        tool_version="0.7.0",
    )


def _runner(study_dir: Path, configs: list[ExperimentConfig], event: threading.Event | None) -> Any:
    from llenergymeasure.config.models import StudyConfig

    study = StudyConfig(experiments=configs)
    manifest = ManifestWriter(study, study_dir)
    return SimpleNamespace(
        study_dir=study_dir,
        manifest=manifest,
        study=SimpleNamespace(
            study_execution=SimpleNamespace(experiment_timeout_seconds=30.0, gpu_indices=None)
        ),
        _progress=None,
        _interrupt_event=event,
        _runner_specs=None,
        _get_env_snapshot=_snapshot,
        # Fields StudyRunner._run_one_server_group reads when driven duck-typed.
        _cycle_counters={},
        _skip_set=set(),
    )


def _wire(
    session: ServerSession,
    *,
    energy_sink: Any,
    windows_per_level: int = 3,
    launch_energy: float = 40.0,
    drain_energy: float = 7.0,
    manager_cls: Any = None,
) -> tuple[FakeWarmupHook, FakeClock]:
    """Override the session's heavy seams with fakes; wire a real WindowManager."""
    from llenergymeasure.harness.window_manager import WindowManager

    clock = FakeClock(start=1000.0)
    warmup = FakeWarmupHook()

    def make_manager(sink: Any, wu: Any) -> Any:
        real = WindowManager(
            sink,
            windows_per_level=windows_per_level,
            warmup_hook=wu,
            sleep=clock.sleep,
            clock=clock,
        )
        return manager_cls(real, session._runner._interrupt_event) if manager_cls else real

    def make_plans(shape: Any, transport: Any) -> list[LevelPlan]:
        spec = WindowSpec(rate=10.0, duration_seconds=10.0, ramp_exclusion_seconds=30.0)
        # The fake clock advances continuously across levels, so each level's spans
        # sit one full level-span (ramp + 3 windows = 60 s) later. Offset the token
        # receipts to land in each level's own windows so every level validates.
        plans = []
        for level_index in range(len(session._cells)):
            offset = level_index * (30.0 + windows_per_level * 10.0)
            report, receipt_fn = _report_with_receipts([t + offset for t in _TOKEN_TIMES])
            plans.append(
                LevelPlan(
                    spec=spec,
                    traffic_source=FakeTrafficSource(report),
                    transport=SimpleNamespace(),
                    token_receipt_fn=receipt_fn,
                )
            )
        return plans

    phase_energies = iter([launch_energy, drain_energy])
    session.__dict__["_make_shape_source"] = lambda: lambda i: RequestShape(index=i, payload={})
    session.__dict__["_make_transport"] = lambda base_url: SimpleNamespace()
    session.__dict__["_make_energy_sink"] = lambda: energy_sink
    session.__dict__["_make_warmup"] = lambda shape, transport: warmup
    session.__dict__["_make_manager"] = make_manager
    session.__dict__["_make_level_plans"] = make_plans
    session.__dict__["_make_phase_bracket"] = lambda detail: FakePhaseBracket(next(phase_energies))
    return warmup, clock


def _bundles(study_dir: Path) -> list[Path]:
    return sorted(p.parent for p in study_dir.rglob(RESULT_FILENAME))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Offline degenerate session block (point 2)
# ---------------------------------------------------------------------------


class TestOfflineSessionBlock:
    def test_offline_bundle_stamps_degenerate_session_block(self, tmp_path: Path) -> None:
        from llenergymeasure.study.runner import _save_and_record
        from tests.conftest import make_result

        class _Manifest:
            def mark_completed(self, *a: Any, **k: Any) -> None:
                pass

            def mark_failed(self, *a: Any, **k: Any) -> None:
                pass

        result_files: list[str] = []
        _save_and_record(
            make_result(),
            tmp_path,
            _Manifest(),  # type: ignore[arg-type]
            "hash",
            1,
            result_files,
            model_name="gpt2",
            engine="transformers",
        )
        assert result_files
        block = json.loads(Path(result_files[0]).read_text())["session"]
        # Offline degenerates to one window with all phase raws null (O7.2).
        assert block["window_count"] == 1
        assert block["level_count"] == 1
        assert block["launch_energy_j"] is None
        assert block["drain_energy_j"] is None
        assert block["session_id"]


# ---------------------------------------------------------------------------
# H1: circuit-breaker abort on a server-group failure is group-aware
# ---------------------------------------------------------------------------


class TestCircuitBreakerGroupAware:
    def test_group_probe_failure_aborts_cleanly_without_keyerror(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # A prior offline failure trips the breaker into probe; the probe is a server
        # GROUP whose launch fails -> abort. The skip sweep must not re-process the
        # group's consumed members (which would double-advance their counters and
        # mark a non-existent future-cycle entry -> KeyError that aborts uncleanly).
        from llenergymeasure.config.models import ExecutionConfig, StudyConfig
        from llenergymeasure.study import server_session as ss
        from llenergymeasure.study.runner import StudyRunner

        offline_fail = _offline_config("gpt2-fail")
        r10, r20 = _server_config(10.0), _server_config(20.0)
        offline_c = _offline_config("gpt2-tail")
        study = StudyConfig(
            experiments=[offline_fail, r10, r20, offline_c],
            study_execution=ExecutionConfig(
                max_consecutive_failures=1, circuit_breaker_cooldown_seconds=0
            ),
        )
        manifest = ManifestWriter(study, tmp_path)
        runner = StudyRunner(study, manifest, tmp_path, no_lock=True)

        # The offline dispatch fails (trips the breaker); the server-group launch
        # fails (the probe), driving the real _run_one_server_group except path.
        monkeypatch.setattr(
            runner,
            "_run_one",
            lambda config, mp_ctx, index: {"type": "E", "message": "offline boom"},
        )

        def _boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("launch boom")

        monkeypatch.setattr(ss.ServerSession, "for_group", staticmethod(_boom))

        runner.run()  # must not raise KeyError

        h = compute_declared_config_hash
        # Clean abort: the study is finalized circuit_breaker (never reached without
        # the fix, since the KeyError would escape first).
        assert manifest.manifest.status == "circuit_breaker"
        # The group's cells were marked failed by the group dispatch...
        assert manifest.entry_status(h(r10), 1) == "failed"
        assert manifest.entry_status(h(r20), 1) == "failed"
        # ...the genuinely-remaining offline cell is skipped (marked once, correctly)...
        assert manifest.entry_status(h(offline_c), 1) == "skipped"
        # ...and the consumed members were never re-marked to a phantom cycle 2.
        assert manifest.entry_status(h(r20), 2) is None


# ---------------------------------------------------------------------------
# First-class StudyResult mapping in orchestration (point 6)
# ---------------------------------------------------------------------------


class TestFirstClassOrchestration:
    def test_run_via_runner_maps_server_windows_first_class(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        from llenergymeasure.config.models import StudyConfig
        from llenergymeasure.study import orchestration
        from llenergymeasure.study import runner as runner_module
        from tests.conftest import make_result

        r1, r2 = make_result(), make_result()
        ssr = ServerSessionResult(
            engine="vllm",
            config_hash="h",
            cycle=1,
            index=1,
            serving_mode="server",
            levels=[],
            token_counting="x",
            total_window_energy_j=10.0,
            elapsed_s=1.0,
            aborted=False,
            abort_reason=None,
            experiment_results=[r1, r2],
            result_files=["/a", "/b"],
        )

        class _FakeRunner:
            def __init__(self, *a: Any, **k: Any) -> None:
                self.result_files = ["/offline"]

            def run(self) -> list[Any]:
                return [ssr, {"type": "X", "message": "boom"}]

        monkeypatch.setattr(runner_module, "StudyRunner", _FakeRunner)
        study = StudyConfig(experiments=[_server_config()])
        files, exps, warnings = orchestration._run_via_runner(study, object(), tmp_path)

        # The session's mapped windows enter experiment_results first-class; a failure
        # dict maps to one None; the session's bundle paths join result_files.
        assert exps == [r1, r2, None]
        assert files == ["/offline", "/a", "/b"]
        assert warnings == ["boom"]

    def test_count_outcomes_is_cell_granular(self) -> None:
        from llenergymeasure.study.orchestration import _count_outcomes
        from tests.conftest import make_result

        offline = make_result()
        results = [
            offline,  # one offline cell
            _srv_result("s", 0, True, 10.0),  # server cell A (valid), three windows
            _srv_result("s", 0, True, 10.0),
            _srv_result("s", 0, True, 10.0),
            _srv_result("s", 1, False, 5.0),  # server cell B (gate-failed), two windows
            _srv_result("s", 1, False, 5.0),
            None,  # a failure dict
        ]
        completed, failed, total_energy = _count_outcomes(results)
        # Cells, not windows: offline + valid cell A = 2 completed; invalid cell B +
        # the None = 2 failed (never 6/1 window-granular).
        assert completed == 2
        assert failed == 2
        # total_energy still sums every non-None window (gate-failed levels count).
        assert total_energy == pytest.approx(offline.total_energy_j + 30.0 + 10.0)


# ---------------------------------------------------------------------------
# partition_server_groups (O7.3)
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_rate_only_difference_folds_offline_and_repeats_split(self) -> None:
        from llenergymeasure.study.server_session import partition_server_groups

        a, b = _server_config(10.0), _server_config(20.0)
        offline = ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="offline")
        # [rate10, rate20] fold; an offline cell is its own unit; a repeat rate10
        # (a second cycle) starts a fresh group.
        units = partition_server_groups([a, b, offline, _server_config(10.0)])
        assert units == [[0, 1], [2], [3]]

    def test_non_rate_difference_does_not_fold(self) -> None:
        from llenergymeasure.study.server_session import partition_server_groups

        a = _server_config(10.0)
        b = ExperimentConfig(
            task={"model": "distilgpt2"},  # a non-rate difference
            engine="vllm",
            serving_mode="server",
            server={"traffic": {"rate": 20, "window_seconds": 10}},
        )
        assert partition_server_groups([a, b]) == [[0], [1]]

    def test_group_dispatches_one_launch_and_per_cell_bundles(self, tmp_path: Path) -> None:
        a, b = _server_config(10.0), _server_config(20.0)
        ha, hb = compute_declared_config_hash(a), compute_declared_config_hash(b)
        runner = _runner(tmp_path, [a, b], event=None)
        cells = [ServerCell(a, ha, 1), ServerCell(b, hb, 1)]
        engine = FakeEngine()
        session = ServerSession.for_group(runner, cells, None, index=1, engine=engine)
        _wire(session, energy_sink=ProducingEnergySink(power_w=100.0))

        with session:
            result = session.run()

        assert isinstance(result, ServerSessionResult)
        # ONE launch + readiness for the whole group (O7.3).
        assert engine.launched == 1
        # Two rate levels x three windows = six bundles, split by grid-point hash.
        bundles = _bundles(tmp_path)
        assert len(bundles) == 6
        hashes = {_read(b / RESULT_FILENAME)["declared_config_hash"] for b in bundles}
        assert hashes == {ha, hb}
        # Both grid points complete on their own level (per-cell manifest).
        statuses = {e.config_hash: e.status for e in runner.manifest.manifest.experiments}
        assert statuses == {ha: "completed", hb: "completed"}


# ---------------------------------------------------------------------------
# Clean session -> bundles on disk with the full session block + drain raws
# ---------------------------------------------------------------------------


class TestCleanSession:
    def test_clean_session_persists_bundles_with_session_and_drain(self, tmp_path: Path) -> None:
        config = _server_config(10.0)
        runner = _runner(tmp_path, [config], event=None)
        session = ServerSession(
            runner,
            config,
            None,
            config_hash=compute_declared_config_hash(config),
            cycle=1,
            index=1,
            engine=FakeEngine(),
        )
        _wire(session, energy_sink=ProducingEnergySink(power_w=100.0))

        with session:
            result = session.run()

        assert isinstance(result, ServerSessionResult)
        assert result.valid is True
        bundles = _bundles(tmp_path)
        assert len(bundles) == 3  # one bundle per measured window
        for bundle in bundles:
            payload = _read(bundle / RESULT_FILENAME)
            block = payload["session"]
            # Every sibling carries the SAME session id and the drain raws (clean close).
            assert block["session_id"] == session._session_id
            assert block["window_count"] == 3
            assert block["level_count"] == 1
            assert block["launch_energy_j"] == pytest.approx(40.0)
            assert block["launch_duration_s"] is not None
            assert block["warmup_total_energy_j"] == pytest.approx(5.0)
            assert block["drain_energy_j"] == pytest.approx(7.0)  # drain stamped on clean close
            assert block["drain_duration_s"] is not None
            # Server per-window provenance is present and truthful.
            prov = payload["server"]
            assert prov["level_index"] == 0
            assert prov["level_valid"] is True
            assert prov["pre_window_protocol"] == "server warmup (test)"
            assert prov["warmup"]["converged"] is True
            # The within-window CoV diagnostic is stamped (distinct from the gate).
            assert prov["intra_window_cov"] is not None
            # The session block is ALSO in system.json (dual-serialised).
            sys_block = _read(bundle / SYSTEM_FILENAME)["session"]
            assert sys_block["drain_energy_j"] == pytest.approx(7.0)
        # Timeseries parquet rode the existing writer path (the core carried samples).
        assert all((b / "timeseries.parquet").exists() for b in bundles)
        # config.json is written host-side with the declared + resolved hashes; the
        # observed half and engine_version stay absent (container-boundary, SM12).
        for bundle in bundles:
            cfg = _read(bundle / "config.json")
            assert (
                cfg["declared_config_hash"]
                == _read(bundle / RESULT_FILENAME)["declared_config_hash"]
            )
            assert cfg["resolved_config_hash"]
            assert "declared_config" in cfg
            assert "engine_version" not in cfg
            assert "observed_config_hash" not in cfg

    def test_persisted_window_energy_sum_equals_session_total(self, tmp_path: Path) -> None:
        config = _server_config(10.0)
        runner = _runner(tmp_path, [config], event=None)
        session = ServerSession(
            runner,
            config,
            None,
            config_hash=compute_declared_config_hash(config),
            cycle=1,
            index=1,
            engine=FakeEngine(),
        )
        _wire(session, energy_sink=ProducingEnergySink(power_w=100.0))
        with session:
            result = session.run()
        assert isinstance(result, ServerSessionResult)
        disk_sum = sum(_read(b / RESULT_FILENAME)["total_energy_j"] for b in _bundles(tmp_path))
        assert disk_sum == pytest.approx(result.total_window_energy_j)
        # 3 windows x (100 W over 10 s) = 3000 J.
        assert result.total_window_energy_j == pytest.approx(3000.0)

    def test_first_class_experiment_results_carry_session_block(self, tmp_path: Path) -> None:
        config = _server_config(10.0)
        runner = _runner(tmp_path, [config], event=None)
        session = ServerSession(
            runner,
            config,
            None,
            config_hash=compute_declared_config_hash(config),
            cycle=1,
            index=1,
            engine=FakeEngine(),
        )
        _wire(session, energy_sink=ProducingEnergySink(power_w=100.0))
        with session:
            result = session.run()
        assert isinstance(result, ServerSessionResult)
        # The mapped ExperimentResults are surfaced for StudyResult.experiments (point 6).
        assert len(result.experiment_results) == 3
        assert len(result.result_files) == 3
        assert all(r.serving_mode == "server" for r in result.experiment_results)
        assert all(r.session is not None for r in result.experiment_results)


# ---------------------------------------------------------------------------
# Mid-level abort -> degraded bundles from the preserved cores
# ---------------------------------------------------------------------------


class TestMidLevelAbort:
    def test_abort_preserves_cores_as_degraded_bundles(self, tmp_path: Path) -> None:
        config = _server_config(10.0)
        runner = _runner(tmp_path, [config], event=None)
        session = ServerSession(
            runner,
            config,
            None,
            config_hash=compute_declared_config_hash(config),
            cycle=1,
            index=1,
            engine=FakeEngine(),
        )
        # close_window raises on window 2 -> windows 0,1 preserved as abort cores.
        _wire(session, energy_sink=ClosePerWindowSink(raise_on=2, power_w=100.0))

        with session:
            result = session.run()

        # No valid level -> failure dict, but the two clean cores are on disk.
        assert isinstance(result, dict)
        bundles = _bundles(tmp_path)
        assert len(bundles) == 2
        for bundle in bundles:
            payload = _read(bundle / RESULT_FILENAME)
            assert payload["server"]["level_valid"] is False
            # A degraded abort-core bundle has no within-window diagnostic.
            assert payload["server"]["intra_window_cov"] is None
            # The abort site is disclosed (a close-window failure here).
            assert "close failed" in payload["server"]["invalid_reason"]
            assert payload["total_energy_j"] == pytest.approx(1000.0)  # 100 W over 10 s
        # The single grid point is marked failed (no valid level).
        assert runner.manifest.manifest.experiments[0].status == "failed"


# ---------------------------------------------------------------------------
# SIGINT mid-session -> completed windows persisted, drain null
# ---------------------------------------------------------------------------


class _InterruptAfterFirstLevel:
    """Runs level 0 through the real manager, then simulates the SIGINT cancel."""

    def __init__(self, real: Any, event: threading.Event) -> None:
        self._real = real
        self._event = event

    async def run_level(self, level_index: int, level: Any) -> Any:
        if level_index == 0:
            return await self._real.run_level(0, level)
        self._event.set()
        raise asyncio.CancelledError()


class TestSigint:
    def test_sigint_persists_first_level_with_drain_null(self, tmp_path: Path) -> None:
        event = threading.Event()
        a, b = _server_config(10.0), _server_config(20.0)
        runner = _runner(tmp_path, [a, b], event=event)
        cells = [
            ServerCell(a, compute_declared_config_hash(a), 1),
            ServerCell(b, compute_declared_config_hash(b), 1),
        ]
        session = ServerSession.for_group(runner, cells, None, index=1, engine=FakeEngine())
        _wire(
            session,
            energy_sink=ProducingEnergySink(power_w=100.0),
            manager_cls=_InterruptAfterFirstLevel,
        )

        with session:
            result = session.run()

        assert isinstance(result, ServerSessionResult)
        # Level 0's three windows are on disk despite the interrupt...
        bundles = _bundles(tmp_path)
        assert len(bundles) == 3
        for bundle in bundles:
            block = _read(bundle / RESULT_FILENAME)["session"]
            # ...and their drain fields are NULL (the interrupted path skips the patch).
            assert block["drain_energy_j"] is None
            assert block["drain_duration_s"] is None
            # Launch/warmup raws (measured before the interrupt) are still present.
            assert block["launch_energy_j"] == pytest.approx(40.0)
        # The manifest is left running (the sweep loop's mark_interrupted downgrades it).
        assert session._runner.manifest.manifest.status == "running"


# ---------------------------------------------------------------------------
# Teardown hardening: a fault after a successful run never converts success to
# failure or rewrites completed manifest history (both altitudes).
# ---------------------------------------------------------------------------


class TestTeardownHardening:
    def test_finalize_block_fault_keeps_result_and_completed_cell(
        self, tmp_path: Path, monkeypatch: Any, caplog: Any
    ) -> None:
        # The verifier's proven scenario, through the dispatch site: a teardown fault
        # building the final=True session block after a successful run must degrade to
        # unpatched-but-finalized bundles (FIX A), never escape to flip the cell.
        from llenergymeasure.study import server_session as ss
        from llenergymeasure.study.runner import StudyRunner

        config = _server_config(10.0)
        runner = _runner(tmp_path, [config], event=None)
        real_for_group = ss.ServerSession.for_group

        def wired(rn: Any, cells: Any, spec: Any, *, index: int, engine: Any = None) -> Any:
            session = real_for_group(rn, cells, spec, index=index, engine=FakeEngine())
            _wire(session, energy_sink=ProducingEnergySink(power_w=100.0))
            original = session._build_session_block

            def faulty(*, final: bool) -> Any:
                if final:
                    raise RuntimeError("teardown block build failed")
                return original(final=final)

            session.__dict__["_build_session_block"] = faulty
            return session

        monkeypatch.setattr(ss.ServerSession, "for_group", staticmethod(wired))
        with caplog.at_level("WARNING"):
            result = StudyRunner._run_one_server_group(runner, [config], index=1)

        # The legitimate result survives (not a failure dict) and the cell stays
        # completed with its bundle path...
        assert isinstance(result, ServerSessionResult)
        entry = runner.manifest.manifest.experiments[0]
        assert entry.status == "completed"
        assert entry.result_file
        # ...and the teardown fault was logged loudly (not swallowed).
        assert any("session block" in r.message.lower() for r in caplog.records)

    def test_escaped_teardown_fault_does_not_rewrite_completed_history(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Defense-in-depth (FIX B): even if a future unguarded teardown line escapes
        # the session guard entirely, the dispatch handler must not downgrade a cell
        # already recorded completed.
        from llenergymeasure.study import server_session as ss
        from llenergymeasure.study.runner import StudyRunner

        config = _server_config(10.0)
        runner = _runner(tmp_path, [config], event=None)
        real_for_group = ss.ServerSession.for_group

        def wired(rn: Any, cells: Any, spec: Any, *, index: int, engine: Any = None) -> Any:
            session = real_for_group(rn, cells, spec, index=index, engine=FakeEngine())
            _wire(session, energy_sink=ProducingEnergySink(power_w=100.0))
            return session

        monkeypatch.setattr(ss.ServerSession, "for_group", staticmethod(wired))

        def escaping_exit(self: Any, *exc: Any) -> None:
            raise RuntimeError("escaped teardown")

        monkeypatch.setattr(ss.ServerSession, "__exit__", escaping_exit)

        result = StudyRunner._run_one_server_group(runner, [config], index=1)

        # The dispatch degrades to a failure dict (the guard was bypassed)...
        assert isinstance(result, dict)
        # ...but the completed cell's history is NOT rewritten.
        entry = runner.manifest.manifest.experiments[0]
        assert entry.status == "completed"
        assert entry.result_file
