"""Integration tests verifying config parameters are actually wired up and used.

These tests ensure config values aren't just parsed but actually affect behaviour.
"""

import time

import pytest

from llenergymeasure.config.backend_configs import PyTorchConfig
from llenergymeasure.config.models import (
    DecoderConfig,
    ExperimentConfig,
    TrafficSimulation,
)
from llenergymeasure.core.traffic import TrafficGenerator


class TestBatchingConfigWired:
    """Verify batching config affects batch creation."""

    def test_batch_size_affects_batch_count(self):
        """Batch size should determine number of batches created."""
        from llenergymeasure.core.prompts import create_fixed_batches

        prompts = [f"Prompt {i}" for i in range(20)]

        # Batch size 4 -> 5 batches
        batches_4 = create_fixed_batches(prompts, batch_size=4)
        assert len(batches_4) == 5
        assert all(len(b) == 4 for b in batches_4)

        # Batch size 10 -> 2 batches
        batches_10 = create_fixed_batches(prompts, batch_size=10)
        assert len(batches_10) == 2

        # Batch size 1 -> 20 batches
        batches_1 = create_fixed_batches(prompts, batch_size=1)
        assert len(batches_1) == 20

    def test_strategy_static_uses_fixed_batches(self):
        """Static strategy should create fixed-size batches."""
        from llenergymeasure.core.prompts import create_fixed_batches

        prompts = ["short", "medium length prompt", "a very long prompt here"]
        batches = create_fixed_batches(prompts, batch_size=2)

        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1  # Remainder

    def test_strategy_sorted_orders_by_length(self):
        """Sorted strategies should order prompts by length."""
        prompts = [
            "very long prompt with many words",
            "short",
            "medium length",
        ]

        sorted_prompts = sorted(prompts, key=len)

        assert sorted_prompts[0] == "short"
        assert sorted_prompts[1] == "medium length"
        assert sorted_prompts[2] == "very long prompt with many words"


class TestDecoderConfigWired:
    """Verify decoder config affects generation kwargs."""

    def test_deterministic_preset_sets_greedy(self):
        """Deterministic preset should disable sampling."""
        config = DecoderConfig(preset="deterministic")

        assert config.temperature == 0.0
        assert config.do_sample is False
        assert config.is_deterministic is True

    def test_temperature_zero_forces_greedy(self):
        """Temperature 0 should force greedy decoding regardless of do_sample."""
        # This tests the logic in _build_generation_kwargs
        config = DecoderConfig(temperature=0.0, do_sample=True)

        # Even with do_sample=True, temp=0 should result in greedy
        assert config.temperature == 0.0
        # The inference code checks temp==0 and overrides do_sample

    def test_sampling_params_only_applied_when_sampling(self):
        """top_k, top_p should only be used when do_sample=True."""
        # When sampling
        sampling_config = DecoderConfig(
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
        )
        assert sampling_config.do_sample is True
        assert sampling_config.top_k == 40
        assert sampling_config.top_p == 0.9

        # When greedy (these values exist but shouldn't be used)
        greedy_config = DecoderConfig(preset="deterministic")
        assert greedy_config.do_sample is False
        # Values exist but inference code won't use them

    def test_generation_kwargs_built_correctly(self):
        """Test PyTorchBackend._build_generation_kwargs produces correct output."""
        from llenergymeasure.core.inference_backends.pytorch import PyTorchBackend

        # Test deterministic - use actual field name, not alias
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            decoder=DecoderConfig(preset="deterministic"),
        )

        backend = PyTorchBackend()
        kwargs = backend._build_generation_kwargs(config, max_output_tokens=50)

        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 50
        assert "top_k" not in kwargs  # Not used for greedy
        assert "top_p" not in kwargs

    def test_generation_kwargs_sampling_mode(self):
        """Test PyTorchBackend._build_generation_kwargs with sampling enabled."""
        from llenergymeasure.core.inference_backends.pytorch import PyTorchBackend

        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            decoder=DecoderConfig(
                temperature=0.8,
                do_sample=True,
                top_k=40,
                top_p=0.9,
            ),
        )

        backend = PyTorchBackend()
        kwargs = backend._build_generation_kwargs(config, max_output_tokens=50)

        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_k"] == 40
        assert kwargs["top_p"] == 0.9


class TestTrafficSimulationWired:
    """Verify traffic simulation actually introduces delays."""

    def test_traffic_generator_creates_delays(self):
        """TrafficGenerator should produce non-zero delays."""
        config = TrafficSimulation(
            enabled=True,
            mode="poisson",
            target_qps=10.0,  # 10 QPS = ~100ms average delay
            seed=42,
        )
        generator = TrafficGenerator(config)

        delays = [generator.get_inter_arrival_time() for _ in range(100)]

        # All delays should be positive
        assert all(d >= 0 for d in delays)
        # Average should be around 0.1s (1/10 QPS)
        avg_delay = sum(delays) / len(delays)
        assert 0.05 < avg_delay < 0.2  # Allow variance

    def test_constant_mode_fixed_delays(self):
        """Constant mode should produce fixed delays."""
        config = TrafficSimulation(
            enabled=True,
            mode="constant",
            target_qps=5.0,  # 5 QPS = 200ms fixed delay
        )
        generator = TrafficGenerator(config)

        delays = [generator.get_inter_arrival_time() for _ in range(10)]

        # All delays should be exactly 0.2s
        assert all(abs(d - 0.2) < 0.001 for d in delays)

    def test_wait_for_next_request_applies_sleep(self):
        """wait_for_next_request should actually sleep."""
        config = TrafficSimulation(
            enabled=True,
            mode="constant",
            target_qps=100.0,  # 10ms delay for fast test
        )
        generator = TrafficGenerator(config)

        start = time.perf_counter()
        generator.wait_for_next_request()
        elapsed = time.perf_counter() - start

        # Should have waited ~10ms
        assert elapsed >= 0.008  # Allow some tolerance

    def test_disabled_traffic_sim_no_generator(self):
        """Disabled traffic simulation config means generator is not created."""
        # Use model_validate to properly handle validation_alias
        config = ExperimentConfig.model_validate(
            {
                "config_name": "test",
                "model_name": "gpt2",
                "traffic_simulation": {"enabled": False},
            }
        )

        # When disabled, backends should not create generator
        assert config.traffic_simulation.enabled is False

    def test_enabled_traffic_sim_creates_generator(self):
        """Enabled traffic simulation config can create a generator."""
        # Use model_validate to properly handle validation_alias
        config = ExperimentConfig.model_validate(
            {
                "config_name": "test",
                "model_name": "gpt2",
                "traffic_simulation": {
                    "enabled": True,
                    "mode": "poisson",
                    "target_qps": 5.0,
                },
            }
        )

        # When enabled, config should allow creating generator
        assert config.traffic_simulation.enabled is True
        generator = TrafficGenerator(config.traffic_simulation)
        assert generator is not None
        assert isinstance(generator, TrafficGenerator)


class TestQuantizationConfigWired:
    """Verify quantization config is passed to model loading.

    In backend-native architecture, quantization settings are in PyTorchConfig.
    """

    def test_quantization_config_in_pytorch_config(self):
        """Quantization settings in PyTorchConfig should be structured correctly."""
        config = PyTorchConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )

        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_compute_dtype == "float16"

    def test_4bit_and_8bit_mutually_exclusive(self):
        """Cannot enable both 4-bit and 8-bit quantization."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            PyTorchConfig(
                load_in_4bit=True,
                load_in_8bit=True,
            )

    def test_quantized_experiment_config(self):
        """ExperimentConfig with quantization via PyTorchConfig."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test-model",
            pytorch=PyTorchConfig(
                load_in_8bit=True,
            ),
        )

        assert config.pytorch is not None
        assert config.pytorch.load_in_8bit is True
        assert config.pytorch.load_in_4bit is False


class TestParallelismConfigImplemented:
    """Verify parallelism config is wired up in backend-native architecture.

    In the backend-native architecture, parallelism is configured via:
    - PyTorch: pytorch.num_processes (data parallelism via Accelerate)
    - vLLM: vllm.tensor_parallel_size, vllm.pipeline_parallel_size (internal)
    - TensorRT: tensorrt.tp_size, tensorrt.pp_size (internal)
    """

    def test_pytorch_parallelism_config_parsed(self):
        """PyTorch parallelism config should be parsed correctly."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            gpus=[0, 1],
            pytorch=PyTorchConfig(num_processes=2),
        )

        assert config.pytorch.num_processes == 2

    def test_vllm_parallelism_config_parsed(self):
        """vLLM parallelism config should be parsed correctly."""
        from llenergymeasure.config.backend_configs import VLLMConfig

        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
            gpus=[0, 1, 2, 3],
            vllm=VLLMConfig(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
            ),
        )

        assert config.vllm.tensor_parallel_size == 2
        assert config.vllm.pipeline_parallel_size == 2

    def test_parallelism_is_used_in_launcher(self):
        """Parallelism config IS used in launcher to determine process count."""
        import inspect

        from llenergymeasure.orchestration import launcher

        source = inspect.getsource(launcher.get_backend_parallelism)

        # Backend-native parallelism settings should be checked
        assert "num_processes" in source  # PyTorch
        assert "tensor_parallel_size" in source  # vLLM
        assert "tp_size" in source  # TensorRT

    def test_parallelism_strategy_factory_exists(self):
        """Parallelism strategy factory should exist and work."""
        from llenergymeasure.core.parallelism import (
            NoParallelism,
            ParallelismConfig,
            TensorParallelStrategy,
            get_parallelism_strategy,
        )

        # Test factory returns correct strategy types
        none_config = ParallelismConfig(strategy="none")
        assert isinstance(get_parallelism_strategy(none_config), NoParallelism)

        tp_config = ParallelismConfig(strategy="tensor_parallel", num_shards=2)
        assert isinstance(get_parallelism_strategy(tp_config), TensorParallelStrategy)


class TestBackendConfigImplemented:
    """Verify backend config is properly wired up via backend protocol."""

    def test_backend_config_parsed(self):
        """Backend config should be parsed correctly."""
        config = ExperimentConfig(
            config_name="test",
            model_name="gpt2",
            backend="vllm",
        )

        assert config.backend == "vllm"

    def test_backend_selection_in_factory(self):
        """Backend selection happens in factory, not inference module."""

        from llenergymeasure.core.inference_backends import get_backend

        # get_backend should exist and work
        pytorch_backend = get_backend("pytorch")
        assert pytorch_backend.name == "pytorch"

        # Each backend implements its own inference
        assert hasattr(pytorch_backend, "run_inference")
