"""GPU-free unit tests for the pre-flight validation module.

All tests run without a GPU. torch, pynvml, and huggingface_hub are never
directly imported - access is always via monkeypatch.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import llenergymeasure.harness.preflight
import llenergymeasure.study.preflight
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.harness.preflight import run_preflight
from llenergymeasure.study.preflight import run_study_preflight
from llenergymeasure.utils.exceptions import PreFlightError
from tests.conftest import make_config, make_study

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch all three checks to succeed."""
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )


# ---------------------------------------------------------------------------
# Test: all checks pass → no exception
# ---------------------------------------------------------------------------


def test_preflight_passes_when_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_all_checks_pass(monkeypatch)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_warn_if_persistence_mode_off", lambda: None
    )
    config = make_config()
    # Should not raise
    run_preflight(config)


# ---------------------------------------------------------------------------
# Test: collect-all pattern - multiple failures reported in one error
# ---------------------------------------------------------------------------


def test_preflight_collects_all_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: False)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: False
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight,
        "_check_model_accessible",
        lambda model_id: f"{model_id} not found on HuggingFace Hub",
    )

    config = make_config()
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    msg = str(exc_info.value)
    assert "3 issue(s) found" in msg
    # All three failures present
    assert "CUDA" in msg
    assert "transformers" in msg.lower() or "transformers" in msg.lower()
    assert "not found" in msg


# ---------------------------------------------------------------------------
# Test: CUDA unavailable → failure
# ---------------------------------------------------------------------------


def test_preflight_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: False)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    config = make_config()
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    assert "CUDA not available" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: engine not installed → failure
# ---------------------------------------------------------------------------


def test_preflight_engine_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: False
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    config = make_config(engine="transformers")
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    assert "transformers not available on host" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: gated model → failure with token hint
# ---------------------------------------------------------------------------


def test_preflight_gated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight,
        "_check_model_accessible",
        lambda model_id: f"{model_id} gated model - no HF_TOKEN → export HF_TOKEN=<your_token>",
    )

    config = make_config(model="meta-llama/Llama-2-7b-hf")
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    msg = str(exc_info.value)
    assert "gated model" in msg
    assert "HF_TOKEN" in msg


# ---------------------------------------------------------------------------
# Test: model not found → failure
# ---------------------------------------------------------------------------


def test_preflight_model_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight,
        "_check_model_accessible",
        lambda model_id: f"{model_id} not found on HuggingFace Hub",
    )

    config = make_config(model="nonexistent/fake-model-xyz")
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    assert "not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: local model path exists → pass
# ---------------------------------------------------------------------------


def test_preflight_local_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_all_checks_pass(monkeypatch)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_warn_if_persistence_mode_off", lambda: None
    )

    # Provide a real path - shouldn't raise
    config = make_config(model=str(tmp_path))
    run_preflight(config)  # Should not raise


# ---------------------------------------------------------------------------
# Test: local model path missing → failure
# ---------------------------------------------------------------------------


def test_preflight_local_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )

    # Do NOT monkeypatch _check_model_accessible - use real implementation
    non_existent = "/definitely/does/not/exist/model"
    config = make_config(model=non_existent)
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    assert non_existent in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: persistence mode warning is non-blocking
# ---------------------------------------------------------------------------


def test_preflight_persistence_mode_warning_not_blocking(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Persistence mode off produces a warning but does NOT raise PreFlightError."""
    # Patch all three checks to pass
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    # Simulate pynvml reporting persistence mode off - do NOT reload the module.
    # Instead we mock _warn_if_persistence_mode_off to log the expected warning.
    def fake_warn() -> None:
        logging.getLogger("llenergymeasure.harness.preflight").warning(
            "GPU persistence mode is off. First experiment may have higher "
            "latency. Enable: sudo nvidia-smi -pm 1"
        )

    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_warn_if_persistence_mode_off", fake_warn
    )

    config = make_config()
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.harness.preflight"):
        # Should NOT raise a PreFlightError
        run_preflight(config)

    # Warning should have been logged
    assert any("persistence mode" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Test: error format matches CONTEXT.md spec
# ---------------------------------------------------------------------------


def test_preflight_error_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trigger 2 failures and check the formatted error string."""
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: False)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: False
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    config = make_config()
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    msg = str(exc_info.value)
    # First line: count summary
    first_line = msg.split("\n")[0]
    assert "Pre-flight failed: 2 issue(s) found" in first_line
    # Each failure line starts with ✗
    failure_lines = [line for line in msg.split("\n") if line.strip().startswith("✗")]
    assert len(failure_lines) == 2


# ---------------------------------------------------------------------------
# Unit tests for internal check functions (not going through run_preflight)
# ---------------------------------------------------------------------------


def test_check_cuda_available_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """_check_cuda_available() returns False when torch is not importable."""
    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util, "find_spec", lambda name: None
    )
    result = llenergymeasure.harness.preflight._check_cuda_available()
    assert result is False


def test_check_engine_installed_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """_check_engine_installed() delegates to find_spec for transformers."""
    mock_spec = MagicMock()  # truthy
    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util,
        "find_spec",
        lambda name: mock_spec if name == "transformers" else None,
    )
    assert llenergymeasure.harness.preflight._check_engine_installed("transformers") is True


def test_check_engine_installed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util, "find_spec", lambda name: None
    )
    assert llenergymeasure.harness.preflight._check_engine_installed("vllm") is False


def test_check_model_accessible_local_exists(tmp_path: Path) -> None:
    result = llenergymeasure.harness.preflight._check_model_accessible(str(tmp_path))
    assert result is None


def test_check_model_accessible_local_missing() -> None:
    result = llenergymeasure.harness.preflight._check_model_accessible(
        "/absolutely/nonexistent/path"
    )
    assert result is not None
    assert "/absolutely/nonexistent/path" in result


def test_check_model_accessible_no_hf_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None (skip) when huggingface_hub is not installed."""
    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util, "find_spec", lambda name: None
    )
    result = llenergymeasure.harness.preflight._check_model_accessible("some/hub-model")
    assert result is None


def test_check_model_accessible_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns gated error string when HfApi raises a 403."""
    mock_hf_hub = MagicMock()
    mock_api = MagicMock()
    mock_api.model_info.side_effect = Exception("403 Client Error: Forbidden")
    mock_hf_hub.HfApi.return_value = mock_api

    monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf_hub)

    # Patch find_spec to pretend huggingface_hub is installed
    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str):
        if name == "huggingface_hub":
            return MagicMock()  # truthy
        return original_find_spec(name)

    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util, "find_spec", patched_find_spec
    )

    result = llenergymeasure.harness.preflight._check_model_accessible("meta-llama/Llama-2-7b-hf")
    assert result is not None
    assert "gated model" in result
    assert "HF_TOKEN" in result


def test_check_model_accessible_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns not-found error string when HfApi raises a 404."""
    mock_hf_hub = MagicMock()
    mock_api = MagicMock()
    mock_api.model_info.side_effect = Exception("404 Client Error: Not Found")
    mock_hf_hub.HfApi.return_value = mock_api

    monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf_hub)

    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str):
        if name == "huggingface_hub":
            return MagicMock()
        return original_find_spec(name)

    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util, "find_spec", patched_find_spec
    )

    result = llenergymeasure.harness.preflight._check_model_accessible("some/nonexistent-model")
    assert result is not None
    assert "not found" in result


def test_check_model_accessible_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None (non-blocking) when HfApi raises a generic network error."""
    mock_hf_hub = MagicMock()
    mock_api = MagicMock()
    mock_api.model_info.side_effect = Exception("Connection timeout")
    mock_hf_hub.HfApi.return_value = mock_api

    monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hf_hub)

    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name: str):
        if name == "huggingface_hub":
            return MagicMock()
        return original_find_spec(name)

    monkeypatch.setattr(
        llenergymeasure.harness.preflight.importlib.util, "find_spec", patched_find_spec
    )

    result = llenergymeasure.harness.preflight._check_model_accessible("some/model")
    assert result is None  # Network errors are non-blocking


# ---------------------------------------------------------------------------
# run_study_preflight: auto-elevation and multi-engine guard (DOCK-05)
# ---------------------------------------------------------------------------


def test_run_study_preflight_single_engine_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-engine study passes study preflight unconditionally."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = make_study(["transformers"])
    # Should not raise
    run_study_preflight(study)


def test_run_study_preflight_multi_engine_docker_available_auto_elevates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """All-auto multi-engine study elevates every engine (no error) when Docker is available."""
    monkeypatch.setattr("llenergymeasure.infra.runner_resolution.is_docker_available", lambda: True)
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", lambda skip=False: None
    )
    study = make_study(["transformers", "vllm"])
    with caplog.at_level(logging.INFO, logger="llenergymeasure.study.preflight"):
        # Should not raise
        run_study_preflight(study)

    # Elevation log line must be present, name the elevated engines, and report
    # which engines kept an explicit pin (none, here).
    info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
    elevation_lines = [m for m in info_messages if "elevated" in m.lower()]
    assert elevation_lines, f"Expected an elevation log message, got: {info_messages}"
    line = elevation_lines[0]
    assert "transformers" in line and "vllm" in line
    assert "explicit" in line.lower()


def test_run_study_preflight_multi_engine_no_docker_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-engine study raises PreFlightError when Docker is not available."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = make_study(["transformers", "vllm"])
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study)

    msg = str(exc_info.value)
    assert "Docker" in msg
    assert "transformers" in msg or "vllm" in msg
    assert "NVIDIA Container Toolkit" in msg or "single engine" in msg


def test_run_study_preflight_error_message_contains_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PreFlightError message lists the conflicting engines."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = make_study(["transformers", "tensorrt"])
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study)

    msg = str(exc_info.value)
    # Both engine names must appear in the error message
    assert "transformers" in msg
    assert "tensorrt" in msg


# ---------------------------------------------------------------------------
# Hardware-compat integration (Check 4 in run_preflight, via build_config_probe)
# ---------------------------------------------------------------------------


def test_run_preflight_collects_hardware_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_preflight collects check_hardware errors into PreFlightError."""
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    class _FakeBackend:
        name = "tensorrt"

        def check_hardware(self, config):
            return ["SM too low for TRT-LLM", "FP8 not supported on this GPU"]

    monkeypatch.setattr("llenergymeasure.engines.get_engine", lambda name: _FakeBackend())

    config = ExperimentConfig(task={"model": "test-model"}, engine="tensorrt")
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    msg = str(exc_info.value)
    assert "SM too low" in msg
    assert "FP8 not supported" in msg


def test_run_preflight_passes_when_hardware_check_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_preflight succeeds when check_hardware returns no errors."""
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_warn_if_persistence_mode_off", lambda: None
    )

    class _FakeBackend:
        name = "transformers"

        def check_hardware(self, config):
            return []

    monkeypatch.setattr("llenergymeasure.engines.get_engine", lambda name: _FakeBackend())

    config = ExperimentConfig(task={"model": "test-model"}, engine="transformers")
    run_preflight(config)


def test_run_preflight_handles_engine_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_preflight does not crash if get_engine raises during check_hardware."""
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_warn_if_persistence_mode_off", lambda: None
    )

    def raise_error(name):
        raise ImportError("tensorrt_llm not installed")

    monkeypatch.setattr("llenergymeasure.engines.get_engine", raise_error)

    config = ExperimentConfig(task={"model": "test-model"}, engine="tensorrt")
    # Should not raise - the try/except in preflight catches ImportError from get_engine
    run_preflight(config)


def test_run_preflight_hardware_errors_counted_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_hardware errors contribute to the total issue count in PreFlightError."""
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    class _FakeBackend:
        name = "tensorrt"

        def check_hardware(self, config):
            return ["FP8 requires SM >= 8.9 but this GPU has SM 8.0"]

    monkeypatch.setattr("llenergymeasure.engines.get_engine", lambda name: _FakeBackend())

    config = ExperimentConfig(task={"model": "test-model"}, engine="tensorrt")
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    msg = str(exc_info.value)
    assert "1 issue(s) found" in msg


# ---------------------------------------------------------------------------
# get_compute_capability - basic contract test
# ---------------------------------------------------------------------------


def test_get_compute_capability_returns_tuple_or_none() -> None:
    """get_compute_capability returns (major, minor) tuple or None - never raises."""
    from llenergymeasure.device.gpu_info import get_compute_capability

    result = get_compute_capability()
    assert result is None or (isinstance(result, tuple) and len(result) == 2)


# ---------------------------------------------------------------------------
# Test: TRT-LLM checkpoint-compat now routes through the check_hardware hook
# (the standalone Check 5 was folded into the tensorrt plugin; the detailed
# checkpoint-gate unit tests live in tests/unit/engines/test_check_hardware.py).
# ---------------------------------------------------------------------------


def test_run_preflight_routes_checkpoint_compat_through_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_preflight surfaces the TRT checkpoint-compat error via check_hardware.

    Proves the routing: the AWQ/GPTQ gate is no longer a standalone preflight
    check - it flows through the real tensorrt plugin's check_hardware (Check 4,
    build_config_probe), the same uniform hook every engine uses. Stubbing the
    plugin-owned quant reader keeps the test hermetic (no HF Hub call).
    """
    import llenergymeasure.engines.tensorrt.plugin as trt_plugin

    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )
    # Plugin-owned quant reader (moved out of preflight into the tensorrt plugin).
    monkeypatch.setattr(trt_plugin, "_read_model_quant_method", lambda _: "awq")

    config = make_config(
        engine="tensorrt",
        tensorrt={"engine_params": {"tensor_parallel_size": 1}},
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    )
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    msg = str(exc_info.value)
    assert "AWQ" in msg
    assert "engine_path" in msg


def test_run_preflight_checkpoint_compat_mock_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint-compat error returned by a plugin's check_hardware surfaces.

    Uses a mock plugin to assert the harness collects whatever check_hardware
    returns - checkpoint-compat is no longer special-cased in preflight.
    """
    monkeypatch.setattr(llenergymeasure.harness.preflight, "_check_cuda_available", lambda: True)
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_engine_installed", lambda engine: True
    )
    monkeypatch.setattr(
        llenergymeasure.harness.preflight, "_check_model_accessible", lambda model_id: None
    )

    class _FakePlugin:
        name = "tensorrt"

        def check_hardware(self, config):
            return ["some-model is an HF AWQ checkpoint; set tensorrt.engine_params.engine_path"]

    monkeypatch.setattr("llenergymeasure.engines.get_engine", lambda name: _FakePlugin())

    config = ExperimentConfig(task={"model": "test-model"}, engine="tensorrt")
    with pytest.raises(PreFlightError) as exc_info:
        run_preflight(config)

    assert "AWQ checkpoint" in str(exc_info.value)
