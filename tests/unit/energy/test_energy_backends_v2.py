"""Unit tests for v2.0 energy samplers and auto-selection.

All tests are fully mocked - no GPU or installed energy packages required.
Tests cover NVMLSampler, ZeusSampler, EnergyMeasurement, and select_energy_sampler().
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.energy import select_energy_sampler
from llenergymeasure.energy.nvml import EnergyMeasurement, NVMLSampler
from llenergymeasure.energy.zeus import ZeusSampler
from llenergymeasure.utils.exceptions import ConfigError

# ---------------------------------------------------------------------------
# NVMLSampler name
# ---------------------------------------------------------------------------


def test_nvml_backend_name() -> None:
    """NVMLSampler.name must return 'nvml'."""
    assert NVMLSampler().name == "nvml"


# ---------------------------------------------------------------------------
# NVMLSampler gpu_indices constructor
# ---------------------------------------------------------------------------


def test_nvml_backend_accepts_gpu_indices() -> None:
    """NVMLSampler(gpu_indices=[0, 1]) stores _gpu_indices correctly."""
    b = NVMLSampler(gpu_indices=[0, 1])
    assert b._gpu_indices == [0, 1]


def test_nvml_backend_defaults_to_gpu_zero() -> None:
    """NVMLSampler() with no args defaults to [0] (backward compatible)."""
    b = NVMLSampler()
    assert b._gpu_indices == [0]


def test_nvml_backend_single_gpu_explicit() -> None:
    """NVMLSampler(gpu_indices=[2]) stores [2]."""
    b = NVMLSampler(gpu_indices=[2])
    assert b._gpu_indices == [2]


# ---------------------------------------------------------------------------
# ZeusSampler name
# ---------------------------------------------------------------------------


def test_zeus_backend_name() -> None:
    """ZeusSampler.name must return 'zeus'."""
    assert ZeusSampler().name == "zeus"


# ---------------------------------------------------------------------------
# NVMLSampler availability
# ---------------------------------------------------------------------------


def test_nvml_is_available_no_gpu() -> None:
    """is_available() returns False when pynvml raises on init."""
    with patch(
        "llenergymeasure.energy.nvml.NVMLSampler.is_available",
        return_value=False,
    ):
        backend = NVMLSampler()
        assert backend.is_available() is False


def test_nvml_is_available_import_error() -> None:
    """is_available() returns False when pynvml is not installed."""
    with patch("importlib.util.find_spec", return_value=None):
        backend = NVMLSampler()
        result = backend.is_available()
    assert result is False


# ---------------------------------------------------------------------------
# ZeusSampler availability
# ---------------------------------------------------------------------------


def test_zeus_is_available_no_package() -> None:
    """is_available() returns False when zeus is not installed."""
    import sys

    # Remove zeus from sys.modules so the deferred import inside is_available()
    # raises ImportError as it would on a machine without zeus installed.
    zeus_modules = {k: v for k, v in sys.modules.items() if k == "zeus" or k.startswith("zeus.")}
    for k in zeus_modules:
        sys.modules.pop(k)

    # Inject a sentinel that raises ImportError on attribute access (simulates absent package)
    sys.modules["zeus.monitor"] = None  # type: ignore[assignment]
    try:
        backend = ZeusSampler()
        result = backend.is_available()
    finally:
        sys.modules.pop("zeus.monitor", None)
        sys.modules.update(zeus_modules)

    assert result is False


# ---------------------------------------------------------------------------
# EnergyMeasurement dataclass
# ---------------------------------------------------------------------------


def test_energy_measurement_dataclass() -> None:
    """EnergyMeasurement can be constructed and fields are accessible."""
    em = EnergyMeasurement(total_j=42.5, duration_sec=10.0)
    assert em.total_j == 42.5
    assert em.duration_sec == 10.0
    assert em.samples == []
    assert em.per_gpu_j is None

    em2 = EnergyMeasurement(
        total_j=100.0,
        duration_sec=5.0,
        samples=[object()],
        per_gpu_j={0: 60.0, 1: 40.0},
    )
    assert em2.per_gpu_j == {0: 60.0, 1: 40.0}


# ---------------------------------------------------------------------------
# select_energy_sampler - null returns None
# ---------------------------------------------------------------------------


def test_select_backend_null_returns_none() -> None:
    """select_energy_sampler(None) must return None without any warnings."""
    result = select_energy_sampler(None)
    assert result is None


# ---------------------------------------------------------------------------
# select_energy_sampler - explicit unavailable raises ConfigError
# ---------------------------------------------------------------------------


def test_select_backend_explicit_unavailable_raises() -> None:
    """Requesting an explicit backend that is not available raises ConfigError."""
    with patch(
        "llenergymeasure.energy.NVMLSampler.is_available",
        return_value=False,
    ):
        with pytest.raises(ConfigError) as exc_info:
            select_energy_sampler("nvml")
        # Error must include install guidance
        assert "pip install" in str(exc_info.value).lower() or "install" in str(exc_info.value)


def test_select_backend_explicit_zeus_unavailable_raises() -> None:
    """Requesting zeus when not installed raises ConfigError with guidance."""
    with patch(
        "llenergymeasure.energy.ZeusSampler.is_available",
        return_value=False,
    ):
        with pytest.raises(ConfigError) as exc_info:
            select_energy_sampler("zeus")
        assert "zeus" in str(exc_info.value).lower()


def test_select_backend_unknown_name_raises() -> None:
    """Requesting an unknown backend name raises ConfigError."""
    with pytest.raises(ConfigError):
        select_energy_sampler("unknown_backend_xyz")


# ---------------------------------------------------------------------------
# select_energy_sampler - auto priority
# ---------------------------------------------------------------------------


def test_select_backend_auto_priority_zeus_first() -> None:
    """Auto-selection returns Zeus when zeus package is available."""
    with (
        patch("importlib.util.find_spec", return_value=MagicMock()) as _mock_find,
        patch(
            "llenergymeasure.energy.ZeusSampler.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_sampler("auto")
    assert isinstance(result, ZeusSampler)


def test_select_backend_auto_priority_nvml_when_no_zeus() -> None:
    """Auto-selection returns NVML when zeus is absent but GPU is available."""

    def fake_find_spec(name: str) -> object:
        if name == "zeus":
            return None  # zeus not installed
        return MagicMock()  # codecarbon present (but NVML checked first)

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch(
            "llenergymeasure.energy.NVMLSampler.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_sampler("auto")
    assert isinstance(result, NVMLSampler)


def test_select_backend_auto_priority_codecarbon_fallback() -> None:
    """Auto-selection returns CodeCarbon when neither zeus nor NVML is available."""
    pytest.importorskip("torch")
    from llenergymeasure.energy.codecarbon import CodeCarbonSampler

    def fake_find_spec(name: str) -> object:
        if name == "zeus":
            return None
        if name == "codecarbon":
            return MagicMock()
        return None

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch(
            "llenergymeasure.energy.NVMLSampler.is_available",
            return_value=False,
        ),
        patch(
            "llenergymeasure.energy.codecarbon.CodeCarbonSampler.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_sampler("auto")
    assert isinstance(result, CodeCarbonSampler)


def test_select_backend_auto_returns_none_when_nothing_available() -> None:
    """Auto-selection returns None when no GPU and no packages are present."""
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch(
            "llenergymeasure.energy.NVMLSampler.is_available",
            return_value=False,
        ),
    ):
        result = select_energy_sampler("auto")
    assert result is None


def test_select_backend_auto_warns_when_nothing_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Auto-selection logs a warning with the per-backend why-chain when nothing is available.

    Regression guard for the silent energy=0.0 failure mode: a disabled energy backend
    must leave a loud trace, not just a zeroed metric.
    """
    with (
        caplog.at_level(logging.WARNING, logger="llenergymeasure.energy"),
        patch("importlib.util.find_spec", return_value=None),
        patch(
            "llenergymeasure.energy.NVMLSampler.is_available",
            return_value=False,
        ),
    ):
        result = select_energy_sampler("auto")

    assert result is None
    assert any("Energy measurement disabled" in rec.message for rec in caplog.records)
    # The why-chain names each probed backend so the failure is diagnosable.
    warning_text = " ".join(rec.message for rec in caplog.records)
    assert "zeus" in warning_text
    assert "nvml" in warning_text
    assert "codecarbon" in warning_text


def test_nvml_sampler_is_available_logs_debug_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NVMLSampler.is_available() leaves a debug trace when the NVML probe raises.

    Fail-soft behaviour (returns False) is unchanged; the change is observability so a
    real driver/permission failure is diagnosable.
    """
    fake_pynvml = MagicMock()
    with (
        caplog.at_level(logging.DEBUG, logger="llenergymeasure.energy.nvml"),
        patch("importlib.util.find_spec", return_value=MagicMock()),
        patch.dict(sys.modules, {"pynvml": fake_pynvml}),
        patch(
            "llenergymeasure.device.gpu_info.nvml_context",
            side_effect=RuntimeError("NVML init failed"),
        ),
    ):
        result = NVMLSampler().is_available()

    assert result is False
    assert any("is_available() probe failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# select_energy_sampler - gpu_indices forwarding
# ---------------------------------------------------------------------------


def test_select_backend_forwards_gpu_indices_nvml() -> None:
    """select_energy_sampler('nvml', gpu_indices=[0, 1]) returns NVMLSampler with those indices."""
    with patch(
        "llenergymeasure.energy.NVMLSampler.is_available",
        return_value=True,
    ):
        result = select_energy_sampler("nvml", gpu_indices=[0, 1])

    assert isinstance(result, NVMLSampler)
    assert result._gpu_indices == [0, 1]


def test_select_backend_forwards_gpu_indices_zeus() -> None:
    """select_energy_sampler('zeus', gpu_indices=[0, 1]) returns ZeusSampler with those indices."""
    with patch(
        "llenergymeasure.energy.ZeusSampler.is_available",
        return_value=True,
    ):
        result = select_energy_sampler("zeus", gpu_indices=[0, 1])

    assert isinstance(result, ZeusSampler)
    assert result._gpu_indices == [0, 1]


def test_select_backend_auto_forwards_gpu_indices() -> None:
    """Auto-selection forwards gpu_indices to the chosen backend."""

    def fake_find_spec(name: str) -> object:
        if name == "zeus":
            return None
        return MagicMock()

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch(
            "llenergymeasure.energy.NVMLSampler.is_available",
            return_value=True,
        ),
    ):
        result = select_energy_sampler("auto", gpu_indices=[2, 3])

    assert isinstance(result, NVMLSampler)
    assert result._gpu_indices == [2, 3]


# ---------------------------------------------------------------------------
# NVMLSampler trapezoidal integration - single GPU
# ---------------------------------------------------------------------------


def test_nvml_trapezoidal_integration() -> None:
    """stop_tracking() correctly integrates power samples via trapezoidal invariant.

    3 samples at t=0.0, 0.1, 0.2 all with power_w=100.0 W.
    Expected energy = 100.0 W * 0.2 s = 20.0 J.
    """
    from llenergymeasure.device.power_thermal import PowerThermalSample

    # Build synthetic samples (single GPU 0)
    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
    ]

    # Mock sampler
    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLSampler(gpu_indices=[0])
    result = backend.stop_tracking(mock_sampler)

    assert isinstance(result, EnergyMeasurement)
    assert result.duration_sec == pytest.approx(0.2, abs=1e-9)
    assert result.total_j == pytest.approx(20.0, abs=1e-6)
    assert result.per_gpu_j == {0: pytest.approx(20.0, abs=1e-6)}  # type: ignore[comparison-overlap]  # Optional dict vs approx mapping


def test_nvml_trapezoidal_skips_none_power() -> None:
    """Trapezoidal integration skips sample pairs where power_w is None."""
    from llenergymeasure.device.power_thermal import PowerThermalSample

    # Middle sample has None power - pairs (0,1) and (1,2) both involve None
    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=None, gpu_index=0),
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
    ]

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLSampler()
    result = backend.stop_tracking(mock_sampler)

    # Both pairs involve a None sample → should integrate to 0.0
    assert result.total_j == pytest.approx(0.0, abs=1e-9)


def test_nvml_trapezoidal_single_sample() -> None:
    """Single sample produces zero energy (no interval to integrate)."""
    from llenergymeasure.device.power_thermal import PowerThermalSample

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = [
        PowerThermalSample(timestamp=0.0, power_w=200.0, gpu_index=0),
    ]

    backend = NVMLSampler()
    result = backend.stop_tracking(mock_sampler)
    assert result.total_j == 0.0
    assert result.duration_sec == 0.0


# ---------------------------------------------------------------------------
# NVMLSampler trapezoidal integration - multi-GPU
# ---------------------------------------------------------------------------


def test_nvml_multi_gpu_trapezoidal_integration() -> None:
    """stop_tracking() sums energy across multiple GPUs, populates per_gpu_j.

    GPU 0: 100W over 0.2s = 20J
    GPU 1: 200W over 0.2s = 40J
    Total: 60J, per_gpu_j = {0: 20J, 1: 40J}

    Samples are interleaved (as produced by _sample_loop).
    """
    from llenergymeasure.device.power_thermal import PowerThermalSample

    # Interleaved samples from GPU 0 and GPU 1
    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.0, power_w=200.0, gpu_index=1),
        PowerThermalSample(timestamp=0.1, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=200.0, gpu_index=1),
        PowerThermalSample(timestamp=0.2, power_w=100.0, gpu_index=0),
        PowerThermalSample(timestamp=0.2, power_w=200.0, gpu_index=1),
    ]

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLSampler(gpu_indices=[0, 1])
    result = backend.stop_tracking(mock_sampler)

    assert isinstance(result, EnergyMeasurement)
    assert result.total_j == pytest.approx(60.0, abs=1e-6)
    assert result.per_gpu_j is not None
    assert result.per_gpu_j[0] == pytest.approx(20.0, abs=1e-6)
    assert result.per_gpu_j[1] == pytest.approx(40.0, abs=1e-6)


def test_nvml_multi_gpu_energy_per_gpu_dict_populated() -> None:
    """per_gpu_j dict contains an entry for each GPU that had samples."""
    from llenergymeasure.device.power_thermal import PowerThermalSample

    sample_data = [
        PowerThermalSample(timestamp=0.0, power_w=50.0, gpu_index=0),
        PowerThermalSample(timestamp=0.0, power_w=75.0, gpu_index=1),
        PowerThermalSample(timestamp=0.0, power_w=100.0, gpu_index=2),
        PowerThermalSample(timestamp=0.1, power_w=50.0, gpu_index=0),
        PowerThermalSample(timestamp=0.1, power_w=75.0, gpu_index=1),
        PowerThermalSample(timestamp=0.1, power_w=100.0, gpu_index=2),
    ]

    mock_sampler = MagicMock()
    mock_sampler.get_samples.return_value = sample_data

    backend = NVMLSampler(gpu_indices=[0, 1, 2])
    result = backend.stop_tracking(mock_sampler)

    assert result.per_gpu_j is not None
    assert set(result.per_gpu_j.keys()) == {0, 1, 2}
    assert result.total_j == pytest.approx(
        result.per_gpu_j[0] + result.per_gpu_j[1] + result.per_gpu_j[2],
        abs=1e-9,
    )


# ---------------------------------------------------------------------------
# CodeCarbonSampler - protocol shape (total_j / per_gpu_j)
# ---------------------------------------------------------------------------


def test_codecarbon_backend_name() -> None:
    """CodeCarbonSampler.name must return 'codecarbon'."""
    from llenergymeasure.energy.codecarbon import CodeCarbonSampler

    assert CodeCarbonSampler().name == "codecarbon"


def test_codecarbon_stop_tracking_returns_energy_measurement() -> None:
    """stop_tracking() returns an EnergyMeasurement exposing total_j and per_gpu_j.

    The harness reads ``.total_j`` and ``.per_gpu_j``; CodeCarbon must mirror the
    NVML/Zeus return shape, not the legacy total_energy_j=... shape.
    """
    from llenergymeasure.energy.codecarbon import CodeCarbonSampler

    # Mock tracker whose _prepare_emissions_data() yields kWh + duration.
    emissions_data = MagicMock()
    emissions_data.energy_consumed = 0.003  # kWh -> 10800 J
    emissions_data.duration = 12.5
    emissions_data.gpu_power = 200.0
    emissions_data.cpu_power = 10.0
    emissions_data.emissions = 0.0001
    tracker = MagicMock()
    tracker._prepare_emissions_data.return_value = emissions_data

    result = CodeCarbonSampler().stop_tracking(tracker)

    assert isinstance(result, EnergyMeasurement)
    # Protocol fields the harness depends on:
    assert result.total_j == pytest.approx(0.003 * 3.6e6, abs=1e-6)
    assert result.duration_sec == pytest.approx(12.5)
    assert result.per_gpu_j is None  # CodeCarbon is process-level, no per-GPU split


def test_codecarbon_stop_tracking_none_tracker_is_empty() -> None:
    """stop_tracking(None) returns a zeroed EnergyMeasurement, not a crash."""
    from llenergymeasure.energy.codecarbon import CodeCarbonSampler

    result = CodeCarbonSampler().stop_tracking(None)
    assert isinstance(result, EnergyMeasurement)
    assert result.total_j == 0.0
    assert result.per_gpu_j is None


# ---------------------------------------------------------------------------
# select_energy_sampler_with_diagnostics - reasons handoff for the harness
# ---------------------------------------------------------------------------


def test_select_with_diagnostics_none_returns_no_reasons() -> None:
    """Intentional disable (None) returns (None, []) - no diagnostics wanted."""
    from llenergymeasure.energy import select_energy_sampler_with_diagnostics

    sampler, reasons = select_energy_sampler_with_diagnostics(None)
    assert sampler is None
    assert reasons == []


def test_select_with_diagnostics_auto_none_carries_reasons() -> None:
    """When auto-selection finds nothing, the per-sampler why-chain is returned.

    This is the structured backup that lets the harness surface the reason
    energy came up empty in measurement_warnings, not only the log.
    """
    from llenergymeasure.energy import select_energy_sampler_with_diagnostics

    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("llenergymeasure.energy.NVMLSampler.is_available", return_value=False),
    ):
        sampler, reasons = select_energy_sampler_with_diagnostics("auto")

    assert sampler is None
    assert any("zeus" in r for r in reasons)
    assert any("nvml" in r for r in reasons)
    assert any("codecarbon" in r for r in reasons)


def test_select_with_diagnostics_auto_success_has_empty_reasons() -> None:
    """A successful auto-selection returns the sampler and no reasons."""
    from llenergymeasure.energy import select_energy_sampler_with_diagnostics

    def fake_find_spec(name: str):
        return None if name == "zeus" else MagicMock()

    with (
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
        patch("llenergymeasure.energy.NVMLSampler.is_available", return_value=True),
    ):
        sampler, reasons = select_energy_sampler_with_diagnostics("auto")

    assert sampler is not None
    assert reasons == []


def test_select_energy_sampler_wrapper_still_returns_bare_sampler() -> None:
    """The bare select_energy_sampler wrapper is unchanged (single return value)."""
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("llenergymeasure.energy.NVMLSampler.is_available", return_value=False),
    ):
        result = select_energy_sampler("auto")
    assert result is None


# ---------------------------------------------------------------------------
# ZeusSampler index space: physical in, CUDA-visible out, physical back
# ---------------------------------------------------------------------------


def test_zeus_monitored_pairs_identity_when_unrestricted(monkeypatch) -> None:
    """No visibility restriction means physical == logical."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ZeusSampler(gpu_indices=[0, 1])._monitored_pairs() == [(0, 0), (1, 1)]


def test_zeus_monitored_pairs_translate_under_a_restriction(monkeypatch) -> None:
    """Physical indices become the visible-set positions ZeusMonitor expects."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert ZeusSampler(gpu_indices=[2, 3])._monitored_pairs() == [(2, 0), (3, 1)]


def test_zeus_window_uses_logical_indices_and_reports_physical(monkeypatch) -> None:
    """The monitor is opened on logical indices; per-GPU energy comes back physical."""
    import sys
    import types
    from unittest.mock import MagicMock

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    opened: dict[str, object] = {}

    class _FakeMonitor:
        def __init__(self, gpu_indices, cpu_indices):
            opened["gpu_indices"] = gpu_indices

        def begin_window(self, name):
            pass

        def end_window(self, name):
            return MagicMock(energy={0: 10.0, 1: 20.0}, time=2.0)

    module = types.ModuleType("zeus.monitor")
    module.ZeusMonitor = _FakeMonitor
    monkeypatch.setitem(sys.modules, "zeus.monitor", module)

    sampler = ZeusSampler(gpu_indices=[2, 3])
    tracker = sampler.start_tracking()
    measurement = sampler.stop_tracking(tracker)

    assert opened["gpu_indices"] == [0, 1]
    assert measurement.per_gpu_j == {2: 10.0, 3: 20.0}
    assert measurement.total_j == 30.0
