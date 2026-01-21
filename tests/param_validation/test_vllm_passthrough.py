"""GPU-based passthrough tests for vLLM parameters.

These tests verify that vLLM parameters are correctly passed through
to the backend configuration. Requires GPU and vLLM installation.

Run with: pytest tests/param_validation/test_vllm_passthrough.py -v
"""

from __future__ import annotations

import gc

import pytest

from .backends.vllm import VLLM_PARAM_SPECS, register_vllm_params
from .conftest import requires_gpu, requires_vllm
from .registry import ParamSpec, VerificationType
from .verifiers import PassthroughVerifier


def cleanup_gpu() -> None:
    """Clean up GPU memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def get_scheduler_config(llm: object) -> object:
    """Get scheduler config from vLLM LLM object, handling v0 vs v1 API."""
    # vLLM v1 API
    if hasattr(llm.llm_engine, "scheduler") and hasattr(
        llm.llm_engine.scheduler, "scheduler_config"
    ):
        return llm.llm_engine.scheduler.scheduler_config
    # Try direct access on engine (older API or alternative v1 path)
    if hasattr(llm.llm_engine, "scheduler_config"):
        return llm.llm_engine.scheduler_config
    # Try getting from vllm_config
    if (
        hasattr(llm, "llm_engine")
        and hasattr(llm.llm_engine, "vllm_config")
        and hasattr(llm.llm_engine.vllm_config, "scheduler_config")
    ):
        return llm.llm_engine.vllm_config.scheduler_config
    raise AttributeError("Cannot find scheduler_config in vLLM LLM object")


def setup_module() -> None:
    """Ensure vLLM params are registered before tests.

    Uses idempotent registration - safe to call multiple times.
    """
    register_vllm_params()


def get_passthrough_params() -> list[tuple[str, ParamSpec, object]]:
    """Get vLLM passthrough parameters for pytest."""
    params = []
    for spec in VLLM_PARAM_SPECS:
        if spec.verification_type != VerificationType.PASSTHROUGH:
            continue
        if spec.skip_reason:
            continue
        # Only include specs that have a passthrough path or checker
        if spec.passthrough_path is None and spec.passthrough_checker is None:
            continue
        for test_value in spec.test_values:
            params.append((f"{spec.name}={test_value}", spec, test_value))
    return params


@requires_gpu
@requires_vllm
class TestVLLMPassthrough:
    """Test vLLM parameter passthrough verification."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    @pytest.mark.parametrize(
        "test_id,spec,test_value",
        get_passthrough_params(),
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_passthrough(self, test_id: str, spec: ParamSpec, test_value: object) -> None:
        """Test that parameter is correctly passed through to vLLM."""
        from vllm import LLM

        # Build vLLM kwargs
        kwargs = {
            "model": "facebook/opt-125m",
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
            "max_model_len": 512,  # Keep small for fast tests
        }

        # Map param to vLLM constructor arg
        self._add_param_to_kwargs(spec, test_value, kwargs)

        try:
            llm = LLM(**kwargs)

            verifier = PassthroughVerifier()
            result = verifier.verify(spec, llm, test_value)

            del llm
            cleanup_gpu()

            if result.failed:
                pytest.fail(f"{spec.full_name}={test_value}: {result.message}")

        except RuntimeError as e:
            cleanup_gpu()
            if "Engine core initialization failed" in str(e):
                pytest.skip(f"{spec.full_name}={test_value}: vLLM v1 engine core init failed")
            pytest.fail(f"{spec.full_name}={test_value}: Failed to create LLM: {e}")
        except ValueError as e:
            cleanup_gpu()
            # Skip tests for unsupported configurations in vLLM v1
            if "not supported" in str(e).lower():
                pytest.skip(f"{spec.full_name}={test_value}: {e}")
            pytest.fail(f"{spec.full_name}={test_value}: Failed to create LLM: {e}")
        except Exception as e:
            cleanup_gpu()
            pytest.fail(f"{spec.full_name}={test_value}: Failed to create LLM: {e}")

    def _add_param_to_kwargs(self, spec: ParamSpec, test_value: object, kwargs: dict) -> None:
        """Add parameter to vLLM constructor kwargs."""
        param_name = spec.name

        # Handle nested params (e.g., lora.enabled -> enable_lora)
        if param_name.startswith("lora."):
            # LoRA requires a compatible model - OPT doesn't support LoRA in vLLM v1
            kwargs["model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            sub_param = param_name.split(".")[-1]
            if sub_param == "enabled":
                kwargs["enable_lora"] = test_value
            elif sub_param == "max_loras":
                kwargs["enable_lora"] = True
                kwargs["max_loras"] = test_value
            elif sub_param == "max_rank":
                kwargs["enable_lora"] = True
                kwargs["max_lora_rank"] = test_value
        else:
            # Direct mapping
            kwargs[param_name] = test_value


@requires_gpu
@requires_vllm
class TestVLLMMemoryBatching:
    """Test vLLM memory and batching parameters specifically."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_max_num_seqs(self) -> None:
        """Verify max_num_seqs is applied to scheduler config."""
        from vllm import LLM

        for max_seqs in [32, 64, 128]:
            llm = LLM(
                model="facebook/opt-125m",
                gpu_memory_utilization=0.3,
                enforce_eager=True,
                max_num_seqs=max_seqs,
            )

            try:
                scheduler_config = get_scheduler_config(llm)
                actual = scheduler_config.max_num_seqs
                assert actual == max_seqs, f"Expected {max_seqs}, got {actual}"
            except AttributeError:
                pytest.skip("Cannot access scheduler_config in this vLLM version")

            del llm
            cleanup_gpu()

    def test_gpu_memory_utilization(self) -> None:
        """Verify gpu_memory_utilization is applied to cache config."""
        from vllm import LLM

        mem_util = 0.5
        llm = LLM(
            model="facebook/opt-125m",
            gpu_memory_utilization=mem_util,
            enforce_eager=True,
        )

        actual = llm.llm_engine.cache_config.gpu_memory_utilization
        assert abs(actual - mem_util) < 0.01, f"Expected {mem_util}, got {actual}"

        del llm
        cleanup_gpu()

    def test_swap_space(self) -> None:
        """Verify swap_space is applied to cache config."""
        from vllm import LLM

        swap_gb = 2.0
        llm = LLM(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            swap_space=swap_gb,
        )

        actual_bytes = llm.llm_engine.cache_config.swap_space_bytes
        actual_gb = actual_bytes / (1024**3)
        assert abs(actual_gb - swap_gb) < 0.1, f"Expected {swap_gb}GB, got {actual_gb}GB"

        del llm
        cleanup_gpu()


@requires_gpu
@requires_vllm
class TestVLLMKVCache:
    """Test vLLM KV cache parameters specifically."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_enable_prefix_caching(self) -> None:
        """Verify enable_prefix_caching is applied."""
        from vllm import LLM

        for enabled in [True, False]:
            llm = LLM(
                model="facebook/opt-125m",
                gpu_memory_utilization=0.3,
                enforce_eager=True,
                enable_prefix_caching=enabled,
            )

            actual = llm.llm_engine.cache_config.enable_prefix_caching
            assert actual == enabled, f"Expected {enabled}, got {actual}"

            del llm
            cleanup_gpu()

    def test_enable_chunked_prefill(self) -> None:
        """Verify enable_chunked_prefill is applied."""
        from vllm import LLM

        llm = LLM(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            enable_chunked_prefill=True,
        )

        try:
            scheduler_config = get_scheduler_config(llm)
            actual = scheduler_config.chunked_prefill_enabled
            assert actual is True, f"Expected True, got {actual}"
        except AttributeError:
            pytest.skip("Cannot access scheduler_config in this vLLM version")

        del llm
        cleanup_gpu()

    def test_block_size(self) -> None:
        """Verify block_size is applied."""
        from vllm import LLM

        # Note: block_size=8 may fail in vLLM v1 due to internal constraints
        for block_size in [16, 32]:
            try:
                llm = LLM(
                    model="facebook/opt-125m",
                    gpu_memory_utilization=0.3,
                    enforce_eager=True,
                    block_size=block_size,
                )

                actual = llm.llm_engine.cache_config.block_size
                assert actual == block_size, f"Expected {block_size}, got {actual}"

                del llm
            except RuntimeError as e:
                if "Engine core initialization failed" in str(e):
                    pytest.skip(f"block_size={block_size} not supported in this vLLM version")
                raise
            finally:
                cleanup_gpu()


@requires_gpu
@requires_vllm
class TestVLLMExecution:
    """Test vLLM execution mode parameters."""

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def test_enforce_eager(self) -> None:
        """Verify enforce_eager controls CUDA graph usage."""
        from vllm import LLM

        for enforce_eager in [True, False]:
            try:
                llm = LLM(
                    model="facebook/opt-125m",
                    gpu_memory_utilization=0.3,
                    enforce_eager=enforce_eager,
                )

                actual = llm.llm_engine.model_config.enforce_eager
                assert actual == enforce_eager, f"Expected {enforce_eager}, got {actual}"

                del llm
            except RuntimeError as e:
                if "Engine core initialization failed" in str(e):
                    pytest.skip(f"enforce_eager={enforce_eager} not supported in this vLLM version")
                raise
            finally:
                cleanup_gpu()

    def test_max_model_len(self) -> None:
        """Verify max_model_len limits context length."""
        from vllm import LLM

        max_len = 256
        llm = LLM(
            model="facebook/opt-125m",
            gpu_memory_utilization=0.3,
            enforce_eager=True,
            max_model_len=max_len,
        )

        actual = llm.llm_engine.model_config.max_model_len
        assert actual == max_len, f"Expected {max_len}, got {actual}"

        del llm
        cleanup_gpu()


@requires_gpu
@requires_vllm
class TestVLLMLoRA:
    """Test vLLM LoRA parameters.

    Uses TinyLlama model which supports LoRA (OPT doesn't support LoRA in vLLM v1).
    vLLM v1 paths: lora_config is at llm_engine.vllm_config.lora_config
    """

    # LoRA-compatible model (OPT doesn't support LoRA in vLLM v1)
    LORA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    @pytest.fixture(autouse=True)
    def cleanup(self) -> None:
        """Clean up GPU between tests."""
        yield
        cleanup_gpu()

    def _get_lora_config(self, llm: object) -> object:
        """Get lora_config from vLLM LLM, handling v0 vs v1 API."""
        # vLLM v1 path
        if hasattr(llm.llm_engine, "vllm_config") and hasattr(
            llm.llm_engine.vllm_config, "lora_config"
        ):
            return llm.llm_engine.vllm_config.lora_config
        # vLLM v0 path (fallback)
        if hasattr(llm.llm_engine, "lora_config"):
            return llm.llm_engine.lora_config
        raise AttributeError("Cannot find lora_config in vLLM LLM object")

    def test_enable_lora(self) -> None:
        """Verify enable_lora creates lora_config."""
        from vllm import LLM

        try:
            llm = LLM(
                model=self.LORA_MODEL,
                gpu_memory_utilization=0.3,
                enforce_eager=True,
                enable_lora=True,
                max_loras=2,
            )

            lora_config = self._get_lora_config(llm)
            assert lora_config is not None, "LoRA config should not be None"

            del llm
        except RuntimeError as e:
            if "Engine core initialization failed" in str(e):
                pytest.skip(f"vLLM v1 engine init failed: {e}")
            raise
        finally:
            cleanup_gpu()

    def test_max_loras(self) -> None:
        """Verify max_loras is applied."""
        from vllm import LLM

        max_loras = 4
        try:
            llm = LLM(
                model=self.LORA_MODEL,
                gpu_memory_utilization=0.3,
                enforce_eager=True,
                enable_lora=True,
                max_loras=max_loras,
            )

            lora_config = self._get_lora_config(llm)
            actual = lora_config.max_loras
            assert actual == max_loras, f"Expected {max_loras}, got {actual}"

            del llm
        except RuntimeError as e:
            if "Engine core initialization failed" in str(e):
                pytest.skip(f"vLLM v1 engine init failed: {e}")
            raise
        finally:
            cleanup_gpu()

    def test_max_lora_rank(self) -> None:
        """Verify max_lora_rank is applied."""
        from vllm import LLM

        max_rank = 32
        try:
            llm = LLM(
                model=self.LORA_MODEL,
                gpu_memory_utilization=0.3,
                enforce_eager=True,
                enable_lora=True,
                max_lora_rank=max_rank,
            )

            lora_config = self._get_lora_config(llm)
            actual = lora_config.max_lora_rank
            assert actual == max_rank, f"Expected {max_rank}, got {actual}"

            del llm
        except RuntimeError as e:
            if "Engine core initialization failed" in str(e):
                pytest.skip(f"vLLM v1 engine init failed: {e}")
            raise
        finally:
            cleanup_gpu()
