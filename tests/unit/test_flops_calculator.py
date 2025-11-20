"""Unit tests for FLOPs calculator."""

import pytest
import torch

from llm_efficiency.metrics import FLOPsCalculator


class TestFLOPsCalculator:
    """Tests for FLOPs calculator."""

    def test_initialization(self, temp_cache_dir):
        """Test calculator initialization."""
        calc = FLOPsCalculator(cache_dir=temp_cache_dir)
        assert calc.cache_dir.exists()
        assert calc.cache == {}

    def test_cache_key_generation(self, flops_calculator):
        """Test cache key generation."""
        key = flops_calculator._get_cache_key("test-model", 128)
        assert key == "test-model::128"
        assert "::" in key

    def test_compute_flops_unquantized(self, flops_calculator, tiny_model, mock_device):
        """Test FLOPs computation for unquantized model."""
        tiny_model = tiny_model.to(mock_device)
        flops = flops_calculator.get_flops(
            model=tiny_model,
            model_name="tiny-gpt2",
            sequence_length=32,
            device=mock_device,
            is_quantized=False,
        )
        assert flops > 0
        assert isinstance(flops, int)

    def test_flops_caching(self, flops_calculator, tiny_model, mock_device):
        """Test that FLOPs are properly cached."""
        tiny_model = tiny_model.to(mock_device)

        # First call - computed
        flops1 = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False
        )

        # Verify cached
        cache_key = flops_calculator._get_cache_key("tiny-gpt2", 32)
        assert cache_key in flops_calculator.cache

        # Second call - from cache
        flops2 = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False
        )

        assert flops1 == flops2

    def test_flops_different_sequence_lengths(self, flops_calculator, tiny_model, mock_device):
        """Test FLOPs for different sequence lengths."""
        tiny_model = tiny_model.to(mock_device)

        flops_32 = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False
        )
        flops_64 = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", 64, mock_device, is_quantized=False
        )

        # Longer sequence should require more FLOPs
        assert flops_64 > flops_32

    def test_architectural_estimation(self, flops_calculator, tiny_model):
        """Test architectural FLOPs estimation."""
        flops = flops_calculator._estimate_flops_from_architecture(
            model=tiny_model, sequence_length=32, num_output_tokens=1
        )
        assert flops > 0
        assert isinstance(flops, int)

    def test_flops_batch_uniform_length(self, flops_calculator, tiny_model, mock_device):
        """Test batch FLOPs with uniform sequence lengths."""
        tiny_model = tiny_model.to(mock_device)
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=mock_device)

        total_flops = flops_calculator.get_flops_batch(
            model=tiny_model,
            model_name="tiny-gpt2",
            input_ids=input_ids,
            device=mock_device,
            is_quantized=False,
        )

        # Should be batch_size * single_flops
        single_flops = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", seq_len, mock_device, is_quantized=False
        )
        assert total_flops == single_flops * batch_size

    def test_flops_batch_variable_length(self, flops_calculator, tiny_model, mock_device):
        """Test batch FLOPs with variable sequence lengths."""
        tiny_model = tiny_model.to(mock_device)
        # Create variable length batch (padded)
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 0, 0, 0],  # length 5
                [1, 2, 3, 4, 5, 6, 7, 8],  # length 8
            ],
            device=mock_device,
        )

        total_flops = flops_calculator.get_flops_batch(
            model=tiny_model,
            model_name="tiny-gpt2",
            input_ids=input_ids,
            device=mock_device,
            is_quantized=False,
        )

        assert total_flops > 0

    def test_clear_cache(self, flops_calculator, tiny_model, mock_device):
        """Test cache clearing."""
        tiny_model = tiny_model.to(mock_device)

        # Add some cached values
        flops_calculator.get_flops(tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False)
        assert len(flops_calculator.cache) > 0

        # Clear cache
        flops_calculator.clear_cache()
        assert len(flops_calculator.cache) == 0
        assert not flops_calculator.cache_file.exists()

    def test_cache_persistence(self, temp_cache_dir, tiny_model, mock_device):
        """Test that cache persists across calculator instances."""
        tiny_model = tiny_model.to(mock_device)

        # First calculator
        calc1 = FLOPsCalculator(cache_dir=temp_cache_dir)
        flops1 = calc1.get_flops(tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False)

        # Second calculator (new instance)
        calc2 = FLOPsCalculator(cache_dir=temp_cache_dir)
        flops2 = calc2.get_flops(
            tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False, force_recompute=False
        )

        # Should load from cache
        assert flops1 == flops2

    @pytest.mark.slow
    def test_force_recompute(self, flops_calculator, tiny_model, mock_device):
        """Test force recompute flag."""
        tiny_model = tiny_model.to(mock_device)

        # First computation
        flops1 = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False
        )

        # Force recompute
        flops2 = flops_calculator.get_flops(
            tiny_model, "tiny-gpt2", 32, mock_device, is_quantized=False, force_recompute=True
        )

        # Should still be equal (same model)
        assert flops1 == flops2
