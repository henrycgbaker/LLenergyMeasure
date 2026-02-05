"""Runtime parameter tests using pytest with SSOT introspection.

These tests verify that configuration parameters are actually applied at inference time.
They require CUDA GPU access and run real inference with a small model.

**SSOT Architecture**: Test parameters are dynamically discovered from Pydantic models
via `llenergymeasure.config.introspection`. This eliminates hardcoded param paths that
can drift from the actual configuration schema.

Run with:
    pytest tests/runtime/ -v                          # All tests
    pytest tests/runtime/ -v -k pytorch               # PyTorch only
    pytest tests/runtime/ -v --backend vllm           # vLLM only
    pytest tests/runtime/ -v --quick                  # Quick subset
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from .test_all_params import (
    TEST_MAX_OUTPUT,
    TEST_MODEL,
    TEST_SAMPLE_SIZE,
    TEST_TIMEOUT_SECONDS,
    should_skip_param,
    verify_inference_results,
)

# =============================================================================
# SSOT Parameter Discovery
# =============================================================================


def get_ssot_backend_params(backend: str) -> dict[str, list[Any]]:
    """Get parameters for a backend from SSOT introspection.

    Uses llenergymeasure.config.introspection as the Single Source of Truth.
    Falls back to test_all_params manual definitions only if import fails.
    """
    try:
        from llenergymeasure.config.introspection import get_backend_params

        introspected = get_backend_params(backend)
        params: dict[str, list[Any]] = {}
        for param_path, meta in introspected.items():
            test_values = meta.get("test_values", [])
            if test_values:
                params[param_path] = test_values
        return params
    except ImportError:
        # Fallback to manual definitions
        from .test_all_params import PYTORCH_PARAMS, TENSORRT_PARAMS, VLLM_PARAMS

        return {
            "pytorch": PYTORCH_PARAMS,
            "vllm": VLLM_PARAMS,
            "tensorrt": TENSORRT_PARAMS,
        }.get(backend, {})


def get_ssot_shared_params() -> dict[str, list[Any]]:
    """Get shared/universal parameters from SSOT introspection."""
    try:
        from llenergymeasure.config.introspection import get_shared_params

        introspected = get_shared_params()
        params: dict[str, list[Any]] = {}
        for param_path, meta in introspected.items():
            test_values = meta.get("test_values", [])
            if test_values:
                params[param_path] = test_values
        return params
    except ImportError:
        from .test_all_params import SHARED_PARAMS

        return SHARED_PARAMS


def get_quick_params(backend: str) -> dict[str, list[Any]]:
    """Get reduced param set for quick mode testing.

    Takes first 2 values from each param for faster execution.
    """
    all_params = {**get_ssot_shared_params(), **get_ssot_backend_params(backend)}
    return {param: values[:2] for param, values in all_params.items() if values}


def get_all_test_params(backend: str, quick: bool = False) -> dict[str, list[Any]]:
    """Get all test parameters for a backend.

    Args:
        backend: One of "pytorch", "vllm", "tensorrt"
        quick: If True, return reduced param set

    Returns:
        Dict mapping param paths to test values
    """
    if quick:
        return get_quick_params(backend)
    return {**get_ssot_shared_params(), **get_ssot_backend_params(backend)}


# =============================================================================
# Test Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent


def create_test_config(
    backend: str,
    param: str,
    value: Any,
    output_dir: Path,
) -> Path:
    """Create a test configuration file.

    Uses backend-native architecture where batching/quantization/parallelism
    are in backend-specific sections.
    """
    config = {
        "config_name": f"{backend}-test-{param.replace('.', '_')}-{value}".replace("/", "_"),
        "model_name": TEST_MODEL,
        "backend": backend,
        "gpus": [0],
        "max_input_tokens": 64,
        "max_output_tokens": TEST_MAX_OUTPUT,
        "num_input_prompts": TEST_SAMPLE_SIZE,
        "fp_precision": "float16",
        "decoder": {"preset": "deterministic"},
        "dataset": {"name": "ai_energy_score", "sample_size": TEST_SAMPLE_SIZE},
    }

    # Add backend defaults (including batching)
    if backend == "pytorch":
        config["pytorch"] = {
            "batch_size": 1,
            "batching_strategy": "static",
            "attn_implementation": "sdpa",
        }
    elif backend == "vllm":
        config["vllm"] = {
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.7,
            "max_model_len": 512,
        }
    elif backend == "tensorrt":
        config["tensorrt"] = {
            "max_batch_size": 4,
            "builder_opt_level": 3,
            "force_rebuild": True,
        }

    # Apply the parameter variation
    parts = param.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_value = str(value).replace(".", "_").replace("/", "_")
    config_path = output_dir / f"{backend}_{parts[-1]}_{safe_value}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


def run_experiment_subprocess(config_path: Path) -> tuple[int, str, str]:
    """Run experiment via subprocess and return (exit_code, stdout, stderr)."""
    cmd = ["lem", "experiment", str(config_path), "--yes"]

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src") + ":" + os.environ.get("PYTHONPATH", ""),
    }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            cwd=PROJECT_ROOT,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Timeout after {TEST_TIMEOUT_SECONDS}s"


# =============================================================================
# Dynamic Test Generation via pytest_generate_tests
# =============================================================================


def generate_param_test_cases(
    backend: str, quick: bool = False
) -> list[tuple[str, Any, str | None]]:
    """Generate (param_path, value, skip_reason) tuples for a backend.

    Returns:
        List of (param_path, test_value, skip_reason_or_none) tuples
    """
    params = get_all_test_params(backend, quick=quick)
    test_cases: list[tuple[str, Any, str | None]] = []

    for param_path, values in params.items():
        for value in values:
            skip, reason = should_skip_param(param_path, value)
            test_cases.append((param_path, value, reason if skip else None))

    return test_cases


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically generate test cases from SSOT introspection.

    This hook is called by pytest to generate parametrized test cases.
    It reads params from Pydantic models via introspection.
    """
    # Only parametrize tests that have the right fixtures
    if "param_path" not in metafunc.fixturenames:
        return
    if "param_value" not in metafunc.fixturenames:
        return
    if "backend" not in metafunc.fixturenames:
        return

    # Get backend from markers or config
    backend_marker = metafunc.definition.get_closest_marker("backend")
    if backend_marker:
        backends = [backend_marker.args[0]]
    else:
        # Check command line option
        backend_opt = metafunc.config.getoption("--backend", default=None)
        if backend_opt:
            backends = [backend_opt]
        else:
            # Infer from test class name
            cls_name = metafunc.cls.__name__ if metafunc.cls else ""
            if "PyTorch" in cls_name:
                backends = ["pytorch"]
            elif "VLLM" in cls_name:
                backends = ["vllm"]
            elif "TensorRT" in cls_name:
                backends = ["tensorrt"]
            else:
                backends = ["pytorch", "vllm", "tensorrt"]

    quick = metafunc.config.getoption("--quick", default=False)

    # Generate test cases
    all_cases: list[tuple[str, str, Any]] = []
    ids: list[str] = []

    for backend in backends:
        test_cases = generate_param_test_cases(backend, quick=quick)
        for param_path, value, _skip_reason in test_cases:
            all_cases.append((backend, param_path, value))
            ids.append(f"{backend}-{param_path}={value}")

    metafunc.parametrize(
        "backend,param_path,param_value",
        all_cases,
        ids=ids,
    )


# =============================================================================
# Generic Parameter Test
# =============================================================================


class TestRuntimeParams:
    """Dynamic runtime tests for all backends using SSOT introspection.

    Test parameters are discovered from Pydantic models, eliminating
    hardcoded param paths that can drift from the actual schema.
    """

    @pytest.mark.requires_gpu
    @pytest.mark.slow
    def test_param_application(
        self,
        backend: str,
        param_path: str,
        param_value: Any,
        clean_results: None,
        temp_config_dir: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test that a parameter is correctly applied at inference time.

        This is the core SSOT test - it dynamically tests any param discovered
        from introspection.

        Args:
            backend: Backend to test (pytorch, vllm, tensorrt)
            param_path: Full parameter path (e.g., "pytorch.batch_size")
            param_value: Value to test
            clean_results: Fixture to clean results directory
            temp_config_dir: Fixture providing temp directory for configs
            request: pytest request fixture for markers
        """
        # Check skip conditions
        skip, reason = should_skip_param(param_path, param_value)
        if skip:
            pytest.skip(reason)

        # Check backend requirements
        if backend == "vllm":
            pytest.importorskip("vllm", reason="vLLM not installed")
        elif backend == "tensorrt":
            pytest.importorskip("tensorrt_llm", reason="TensorRT-LLM not installed")

        # Create config with the parameter variation
        config_path = create_test_config(
            backend=backend,
            param=param_path,
            value=param_value,
            output_dir=temp_config_dir,
        )

        # Run experiment
        exit_code, stdout, stderr = run_experiment_subprocess(config_path)

        # Assert success
        assert exit_code == 0, f"{param_path}={param_value} failed: {stderr}"

        # Verify actual inference happened
        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {param_path}={param_value}: {error}"


# =============================================================================
# Backend-Specific Test Classes (for better pytest organisation)
# =============================================================================


class TestPyTorchRuntime:
    """Runtime tests for PyTorch backend.

    Uses SSOT introspection for parameter discovery.
    """

    @pytest.fixture(autouse=True)
    def _setup_backend(self) -> None:
        """Set backend for this test class."""
        self.backend = "pytorch"

    @pytest.mark.requires_gpu
    @pytest.mark.slow
    def test_pytorch_baseline(
        self,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test PyTorch baseline configuration runs successfully."""
        config_path = create_test_config(
            backend="pytorch",
            param="fp_precision",
            value="float16",
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Baseline failed: {stderr}"

        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"
        assert metrics.get("throughput_tokens_per_second", 0) > 0


def _make_pytorch_param_test(param_path: str, test_values: list[Any]) -> type:
    """Factory to create a parametrized test method for a PyTorch param."""

    @pytest.mark.requires_gpu
    @pytest.mark.slow
    @pytest.mark.parametrize("value", test_values, ids=[str(v) for v in test_values])
    def test_method(
        self: TestPyTorchRuntime,
        value: Any,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        skip, reason = should_skip_param(param_path, value)
        if skip:
            pytest.skip(reason)

        config_path = create_test_config(
            backend="pytorch",
            param=param_path,
            value=value,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"{param_path}={value} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {param_path}={value}: {error}"

    return test_method


# Dynamically add PyTorch param tests to TestPyTorchRuntime
_pytorch_params = get_ssot_backend_params("pytorch")
_shared_params = get_ssot_shared_params()
_all_pytorch_params = {**_shared_params, **_pytorch_params}

for _param_path, _values in _all_pytorch_params.items():
    if _values:
        _method_name = f"test_pytorch_{_param_path.replace('.', '_')}"
        setattr(TestPyTorchRuntime, _method_name, _make_pytorch_param_test(_param_path, _values))


class TestVLLMRuntime:
    """Runtime tests for vLLM backend.

    Uses SSOT introspection for parameter discovery.
    """

    @pytest.fixture(autouse=True)
    def _setup_backend(self) -> None:
        """Set backend for this test class."""
        self.backend = "vllm"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    def test_vllm_baseline(
        self,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test vLLM baseline configuration runs successfully."""
        config_path = create_test_config(
            backend="vllm",
            param="fp_precision",
            value="float16",
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Baseline failed: {stderr}"

        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"


def _make_vllm_param_test(param_path: str, test_values: list[Any]) -> type:
    """Factory to create a parametrized test method for a vLLM param."""

    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    @pytest.mark.parametrize("value", test_values, ids=[str(v) for v in test_values])
    def test_method(
        self: TestVLLMRuntime,
        value: Any,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        skip, reason = should_skip_param(param_path, value)
        if skip:
            pytest.skip(reason)

        config_path = create_test_config(
            backend="vllm",
            param=param_path,
            value=value,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"{param_path}={value} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {param_path}={value}: {error}"

    return test_method


# Dynamically add vLLM param tests to TestVLLMRuntime
_vllm_params = get_ssot_backend_params("vllm")
_all_vllm_params = {**_shared_params, **_vllm_params}

for _param_path, _values in _all_vllm_params.items():
    if _values:
        _method_name = f"test_vllm_{_param_path.replace('.', '_')}"
        setattr(TestVLLMRuntime, _method_name, _make_vllm_param_test(_param_path, _values))


class TestTensorRTRuntime:
    """Runtime tests for TensorRT-LLM backend.

    Uses SSOT introspection for parameter discovery.
    """

    @pytest.fixture(autouse=True)
    def _setup_backend(self) -> None:
        """Set backend for this test class."""
        self.backend = "tensorrt"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_tensorrt
    @pytest.mark.slow
    def test_tensorrt_baseline(
        self,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        """Test TensorRT baseline configuration runs successfully."""
        config_path = create_test_config(
            backend="tensorrt",
            param="fp_precision",
            value="float16",
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"Baseline failed: {stderr}"

        config_name = config_path.stem
        success, error, metrics = verify_inference_results(config_name)
        assert success, f"Inference validation failed: {error}"


def _make_tensorrt_param_test(param_path: str, test_values: list[Any]) -> type:
    """Factory to create a parametrized test method for a TensorRT param."""

    @pytest.mark.requires_gpu
    @pytest.mark.requires_tensorrt
    @pytest.mark.slow
    @pytest.mark.parametrize("value", test_values, ids=[str(v) for v in test_values])
    def test_method(
        self: TestTensorRTRuntime,
        value: Any,
        clean_results: None,
        temp_config_dir: Path,
    ) -> None:
        skip, reason = should_skip_param(param_path, value)
        if skip:
            pytest.skip(reason)

        config_path = create_test_config(
            backend="tensorrt",
            param=param_path,
            value=value,
            output_dir=temp_config_dir,
        )

        exit_code, stdout, stderr = run_experiment_subprocess(config_path)
        assert exit_code == 0, f"{param_path}={value} failed: {stderr}"

        config_name = config_path.stem
        success, error, _ = verify_inference_results(config_name)
        assert success, f"Inference validation failed for {param_path}={value}: {error}"

    return test_method


# Dynamically add TensorRT param tests to TestTensorRTRuntime
_tensorrt_params = get_ssot_backend_params("tensorrt")
_all_tensorrt_params = {**_shared_params, **_tensorrt_params}

for _param_path, _values in _all_tensorrt_params.items():
    if _values:
        _method_name = f"test_tensorrt_{_param_path.replace('.', '_')}"
        setattr(TestTensorRTRuntime, _method_name, _make_tensorrt_param_test(_param_path, _values))


# =============================================================================
# Comprehensive Parameter Sweep (via test_all_params.py)
# =============================================================================


class TestFullParameterSweep:
    """Run comprehensive parameter sweep using test_all_params.py script.

    This is a convenience test that runs the full parameter sweep script.
    For more granular testing, use the individual test classes above.
    """

    @pytest.mark.requires_gpu
    @pytest.mark.slow
    def test_pytorch_full_sweep(self, request: pytest.FixtureRequest) -> None:
        """Run full PyTorch parameter sweep."""
        import sys

        quick = request.config.getoption("--quick")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "test_all_params.py"),
            "--backend",
            "pytorch",
            "--output",
            str(PROJECT_ROOT / "results" / "test_results_pytorch.json"),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour for full sweep
            cwd=PROJECT_ROOT,
        )

        # Check results file
        results_path = PROJECT_ROOT / "results" / "test_results_pytorch.json"
        if results_path.exists():
            with open(results_path) as f:
                report = json.load(f)
            failed = report.get("summary", {}).get("failed", 0)
            total = report.get("summary", {}).get("total", 0)
            assert failed == 0, f"{failed}/{total} tests failed. See {results_path}"
        else:
            assert result.returncode == 0, f"Sweep failed: {result.stderr}"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_vllm
    @pytest.mark.slow
    def test_vllm_full_sweep(self, request: pytest.FixtureRequest) -> None:
        """Run full vLLM parameter sweep."""
        import sys

        quick = request.config.getoption("--quick")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "test_all_params.py"),
            "--backend",
            "vllm",
            "--output",
            str(PROJECT_ROOT / "results" / "test_results_vllm.json"),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=PROJECT_ROOT,
        )

        results_path = PROJECT_ROOT / "results" / "test_results_vllm.json"
        if results_path.exists():
            with open(results_path) as f:
                report = json.load(f)
            failed = report.get("summary", {}).get("failed", 0)
            total = report.get("summary", {}).get("total", 0)
            assert failed == 0, f"{failed}/{total} tests failed. See {results_path}"
        else:
            assert result.returncode == 0, f"Sweep failed: {result.stderr}"

    @pytest.mark.requires_gpu
    @pytest.mark.requires_tensorrt
    @pytest.mark.slow
    def test_tensorrt_full_sweep(self, request: pytest.FixtureRequest) -> None:
        """Run full TensorRT parameter sweep."""
        import sys

        quick = request.config.getoption("--quick")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "test_all_params.py"),
            "--backend",
            "tensorrt",
            "--output",
            str(PROJECT_ROOT / "results" / "test_results_tensorrt.json"),
        ]
        if quick:
            cmd.append("--quick")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=PROJECT_ROOT,
        )

        results_path = PROJECT_ROOT / "results" / "test_results_tensorrt.json"
        if results_path.exists():
            with open(results_path) as f:
                report = json.load(f)
            failed = report.get("summary", {}).get("failed", 0)
            total = report.get("summary", {}).get("total", 0)
            assert failed == 0, f"{failed}/{total} tests failed. See {results_path}"
        else:
            assert result.returncode == 0, f"Sweep failed: {result.stderr}"
