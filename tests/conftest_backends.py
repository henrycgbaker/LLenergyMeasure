"""Mock backend for testing without GPU.

Provides a MockBackend that conforms to the InferenceBackend protocol
for contract testing and unit tests.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from llm_energy_measure.core.inference_backends.protocols import (
    BackendResult,
    BackendRuntime,
    ConfigWarning,
)

if TYPE_CHECKING:
    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.domain.model_info import ModelInfo


class MockBackend:
    """Test backend that simulates inference without GPU.

    Used for contract testing and unit tests where GPU is not available.
    Configurable latency and token counts for realistic simulation.
    """

    def __init__(
        self,
        latency_per_prompt_ms: float = 10.0,
        output_tokens_per_prompt: int = 50,
        fail_on_inference: bool = False,
    ) -> None:
        """Initialize mock backend.

        Args:
            latency_per_prompt_ms: Simulated latency per prompt in milliseconds.
            output_tokens_per_prompt: Number of output tokens to generate per prompt.
            fail_on_inference: If True, raise exception during inference (for error testing).
        """
        self._latency_ms = latency_per_prompt_ms
        self._output_tokens = output_tokens_per_prompt
        self._fail_on_inference = fail_on_inference
        self._initialized = False
        self._config: ExperimentConfig | None = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "mock"

    @property
    def version(self) -> str:
        """Backend version string."""
        return "mock-1.0.0"

    def is_available(self) -> bool:
        """Mock backend is always available."""
        return True

    def initialize(self, config: ExperimentConfig, runtime: BackendRuntime) -> None:
        """Simulate model loading.

        Args:
            config: Experiment configuration.
            runtime: Runtime context (ignored in mock).
        """
        self._config = config
        self._initialized = True

    def run_inference(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Simulate inference with configurable latency.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with simulated token counts and timing.

        Raises:
            RuntimeError: If fail_on_inference is True.
        """
        if self._fail_on_inference:
            raise RuntimeError("Mock inference failure (configured)")

        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Simulate processing time
        num_prompts = len(prompts)
        total_latency_sec = (self._latency_ms * num_prompts) / 1000.0
        time.sleep(total_latency_sec * 0.1)  # Sleep 10% of simulated time

        # Calculate token counts
        # Assume ~10 tokens per prompt for input (simplified)
        input_tokens = num_prompts * 10
        output_tokens = num_prompts * self._output_tokens
        total_tokens = input_tokens + output_tokens

        return BackendResult(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            inference_time_sec=total_latency_sec,
            batch_latencies_ms=[self._latency_ms] * num_prompts,
            output_texts=[f"Mock response for prompt {i}" for i in range(num_prompts)],
            backend_metadata={
                "backend": self.name,
                "version": self.version,
                "simulated": True,
            },
        )

    def cleanup(self) -> None:
        """Clean up mock backend (no-op)."""
        self._initialized = False
        self._config = None

    def get_model_info(self) -> ModelInfo:
        """Return mock model metadata.

        Returns:
            ModelInfo with placeholder values.
        """
        from llm_energy_measure.domain.model_info import ModelInfo, QuantizationSpec

        return ModelInfo(
            name=self._config.model_name if self._config else "mock-model",
            revision=None,
            num_parameters=1_000_000,  # 1M params
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            vocab_size=50000,
            model_type="mock",
            torch_dtype="float32",
            quantization=QuantizationSpec(
                enabled=False, bits=None, method="none", compute_dtype="float32"
            ),
        )

    def get_supported_params(self) -> set[str]:
        """Return all parameters as supported (mock accepts everything)."""
        return {
            "model_name",
            "fp_precision",
            "max_input_tokens",
            "max_output_tokens",
            "min_output_tokens",
            "random_seed",
            "batch_size",
            "batching.batch_size",
            "batching.strategy",
            "decoder.temperature",
            "decoder.top_p",
            "decoder.top_k",
        }

    def validate_config(self, config: ExperimentConfig) -> list[ConfigWarning]:
        """Mock backend has no config warnings."""
        return []


def register_mock_backend() -> None:
    """Register MockBackend in the backend registry for testing."""
    from llm_energy_measure.core.inference_backends import register_backend

    register_backend("mock", MockBackend)
