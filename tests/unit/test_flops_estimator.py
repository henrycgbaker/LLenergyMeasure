"""Tests for FlopsEstimator and FlopsResult."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.core.flops import FlopsEstimator, estimate_flops, get_flops_estimator
from llenergymeasure.domain.metrics import FlopsResult


class TestFlopsResult:
    """Tests for FlopsResult model."""

    def test_creation(self):
        result = FlopsResult(
            value=1e12,
            method="calflops",
            confidence="high",
            precision="fp16",
        )

        assert result.value == 1e12
        assert result.method == "calflops"
        assert result.confidence == "high"
        assert result.precision == "fp16"
        assert result.notes is None

    def test_creation_with_notes(self):
        result = FlopsResult(
            value=5e11,
            method="architecture",
            confidence="medium",
            precision="fp32",
            notes="Based on model config: llama",
        )

        assert result.notes == "Based on model config: llama"

    def test_is_valid_true(self):
        result = FlopsResult(
            value=1e12,
            method="calflops",
            confidence="high",
            precision="fp16",
        )
        assert result.is_valid is True

    def test_is_valid_false_zero(self):
        result = FlopsResult(
            value=0.0,
            method="parameter_estimate",
            confidence="low",
            precision="fp16",
        )
        assert result.is_valid is False

    def test_is_valid_false_negative(self):
        result = FlopsResult(
            value=-1.0,
            method="parameter_estimate",
            confidence="low",
            precision="fp16",
        )
        assert result.is_valid is False

    def test_method_literal_validation(self):
        """Test that only valid methods are accepted."""
        # Valid methods
        for method in ["calflops", "architecture", "parameter_estimate"]:
            result = FlopsResult(value=1e10, method=method, confidence="high", precision="fp16")
            assert result.method == method

    def test_confidence_literal_validation(self):
        """Test that only valid confidence levels are accepted."""
        for confidence in ["high", "medium", "low"]:
            result = FlopsResult(
                value=1e10, method="calflops", confidence=confidence, precision="fp16"
            )
            assert result.confidence == confidence


class TestFlopsEstimator:
    """Tests for FlopsEstimator class."""

    @pytest.fixture
    def estimator(self):
        return FlopsEstimator()

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with config and parameters."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_size = 4096
        model.config.num_hidden_layers = 32
        model.config.num_attention_heads = 32
        model.config.intermediate_size = 11008
        model.config.model_type = "llama"
        model.config.hidden_act = "silu"

        # Mock parameters for parameter-based estimation
        param1 = MagicMock()
        param1.numel.return_value = 1_000_000
        param2 = MagicMock()
        param2.numel.return_value = 500_000
        model.parameters.return_value = [param1, param2]

        return model

    @pytest.fixture
    def mock_input_ids(self):
        return torch.zeros((1, 512), dtype=torch.long)

    def test_get_compute_precision_no_config(self, estimator):
        """Test precision detection with no config."""
        assert estimator._get_compute_precision(None) == "fp16"

    def test_get_compute_precision_quantized_4bit(self, estimator):
        """Test precision detection for 4-bit quantized models."""
        config = MagicMock()
        # In backend-native arch, quantization is in pytorch config
        config.pytorch = MagicMock()
        config.pytorch.load_in_4bit = True
        config.pytorch.load_in_8bit = False
        config.pytorch.bnb_4bit_compute_dtype = "float16"

        assert estimator._get_compute_precision(config) == "float16"

    def test_get_compute_precision_quantized_8bit(self, estimator):
        """Test precision detection for 8-bit quantized models (always FP16)."""
        config = MagicMock()
        config.pytorch = MagicMock()
        config.pytorch.load_in_4bit = False
        config.pytorch.load_in_8bit = True

        assert estimator._get_compute_precision(config) == "fp16"

    def test_get_compute_precision_from_fp_precision(self, estimator):
        """Test precision detection from fp_precision config."""
        config = MagicMock()
        # No quantization
        config.pytorch = MagicMock()
        config.pytorch.load_in_4bit = False
        config.pytorch.load_in_8bit = False
        config.fp_precision = "float32"

        assert estimator._get_compute_precision(config) == "fp32"

    def test_calflops_fallback_on_import_error(self, estimator, mock_model, mock_input_ids):
        """Test that calflops import error falls back to architecture."""
        with (
            patch.dict("sys.modules", {"calflops": None}),
            patch.object(estimator, "_try_calflops", return_value=None),
        ):
            result = estimator.estimate(mock_model, mock_input_ids)

        # Should fall back to architecture or parameter_estimate
        assert result.method in ["architecture", "parameter_estimate"]

    def test_architecture_estimation(self, estimator, mock_model):
        """Test architecture-based FLOPs estimation."""
        result = estimator._try_architecture(mock_model, seq_len=512, precision="fp16")

        assert result is not None
        assert result.method == "architecture"
        assert result.confidence == "medium"
        assert result.value > 0
        assert "llama" in (result.notes or "")

    def test_architecture_estimation_missing_config(self, estimator):
        """Test architecture estimation with missing config attributes."""
        model = MagicMock()
        model.config = MagicMock(spec=[])  # Empty spec, no attributes

        result = estimator._try_architecture(model, seq_len=512, precision="fp16")

        assert result is None

    def test_parameter_estimate(self, estimator, mock_model):
        """Test parameter-based FLOPs estimation."""
        result = estimator._parameter_estimate(mock_model, seq_len=512, precision="fp16")

        # 2 * (1_000_000 + 500_000) params * 512 tokens = 1,536,000,000
        expected = 2 * 1_500_000 * 512
        assert result.value == expected
        assert result.method == "parameter_estimate"
        assert result.confidence == "low"

    def test_parameter_estimate_error_handling(self, estimator):
        """Test parameter estimation with failing model.parameters()."""
        model = MagicMock()
        model.parameters.side_effect = RuntimeError("No parameters")

        result = estimator._parameter_estimate(model, seq_len=512, precision="fp16")

        assert result.value == 0.0
        assert result.method == "parameter_estimate"
        assert result.confidence == "low"
        assert "failed" in (result.notes or "").lower()

    def test_parse_flops_string_gflops(self, estimator):
        """Test parsing GFLOPS string."""
        assert estimator._parse_flops_string("1.5 GFLOPS") == 1.5e9
        assert estimator._parse_flops_string("2.0G") == 2.0e9

    def test_parse_flops_string_tflops(self, estimator):
        """Test parsing TFLOPS string."""
        assert estimator._parse_flops_string("1.0 TFLOPS") == 1.0e12
        assert estimator._parse_flops_string("0.5T") == 0.5e12

    def test_parse_flops_string_mflops(self, estimator):
        """Test parsing MFLOPS string."""
        assert estimator._parse_flops_string("100 MFLOPS") == 100e6
        assert estimator._parse_flops_string("50M") == 50e6

    def test_parse_flops_string_invalid(self, estimator):
        """Test parsing invalid string."""
        assert estimator._parse_flops_string("invalid") is None
        assert estimator._parse_flops_string("") is None

    def test_estimate_uses_fallback_chain(self, estimator, mock_model, mock_input_ids):
        """Test that estimate uses the fallback chain correctly."""
        # Make calflops fail
        with patch.object(estimator, "_try_calflops", return_value=None):
            result = estimator.estimate(mock_model, mock_input_ids)

        # Should use architecture (has all required attributes)
        assert result.method == "architecture"
        assert result.value > 0

    def test_estimate_1d_input(self, estimator, mock_model):
        """Test estimation with 1D input tensor."""
        input_ids = torch.zeros(512, dtype=torch.long)

        with patch.object(estimator, "_try_calflops", return_value=None):
            result = estimator.estimate(mock_model, input_ids)

        assert result.value > 0


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_flops_estimator_singleton(self):
        """Test that get_flops_estimator returns same instance."""
        est1 = get_flops_estimator()
        est2 = get_flops_estimator()

        assert est1 is est2

    def test_estimate_flops_convenience(self):
        """Test the estimate_flops convenience function."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_size = 768
        model.config.num_hidden_layers = 12
        model.config.num_attention_heads = 12
        model.config.intermediate_size = 3072
        model.config.model_type = "bert"
        model.config.hidden_act = "gelu"

        param = MagicMock()
        param.numel.return_value = 110_000_000
        model.parameters.return_value = [param]

        input_ids = torch.zeros((1, 128), dtype=torch.long)

        result = estimate_flops(model, input_ids)

        assert isinstance(result, FlopsResult)
        assert result.value > 0


class TestCalflopsIntegration:
    """Tests for calflops integration (when available)."""

    def test_try_calflops_success(self):
        """Test successful calflops estimation."""
        estimator = FlopsEstimator()
        model = MagicMock()

        mock_result = (1.5e12, 7.5e11, 7e9)  # (flops, macs, params)

        # Create a mock calflops module
        mock_calflops = MagicMock()
        mock_calflops.calculate_flops = MagicMock(return_value=mock_result)

        with patch.dict("sys.modules", {"calflops": mock_calflops}):
            result = estimator._try_calflops(model, seq_len=512, precision="fp16")

        assert result is not None
        assert result.value == 1.5e12
        assert result.method == "calflops"
        assert result.confidence == "high"

    def test_try_calflops_import_error(self):
        """Test calflops import error handling."""
        estimator = FlopsEstimator()
        model = MagicMock()

        # Simulate import error by removing calflops from modules
        with patch.dict("sys.modules", {"calflops": None}):
            # This should catch ImportError and return None
            result = estimator._try_calflops(model, seq_len=512, precision="fp16")

        assert result is None

    def test_try_calflops_runtime_error(self):
        """Test calflops runtime error handling."""
        estimator = FlopsEstimator()
        model = MagicMock()

        # Create a mock calflops module that raises an error
        mock_calflops = MagicMock()
        mock_calflops.calculate_flops = MagicMock(side_effect=RuntimeError("Model not supported"))

        with patch.dict("sys.modules", {"calflops": mock_calflops}):
            result = estimator._try_calflops(model, seq_len=512, precision="fp16")

        assert result is None


class TestBNBQuantizationHandling:
    """Tests for BitsAndBytes quantization handling in backend-native architecture."""

    def test_bnb_4bit_uses_compute_dtype(self):
        """Test that 4-bit BNB models use the bnb_4bit_compute_dtype."""
        from llenergymeasure.config.backend_configs import PyTorchConfig

        estimator = FlopsEstimator()

        config = ExperimentConfig(
            config_name="test",
            model_name="test/model",
            pytorch=PyTorchConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
            ),
        )

        precision = estimator._get_compute_precision(config)

        assert precision == "float16"

    def test_bnb_8bit_uses_fp16_precision(self):
        """Test that 8-bit BNB models use FP16 precision."""
        from llenergymeasure.config.backend_configs import PyTorchConfig

        estimator = FlopsEstimator()

        config = ExperimentConfig(
            config_name="test",
            model_name="test/model",
            pytorch=PyTorchConfig(
                load_in_8bit=True,
            ),
        )

        precision = estimator._get_compute_precision(config)

        assert precision == "fp16"

    def test_non_quantized_uses_fp_precision(self):
        """Test that non-quantized models use fp_precision config."""
        estimator = FlopsEstimator()

        config = ExperimentConfig(
            config_name="test",
            model_name="test/model",
            fp_precision="bfloat16",
        )

        precision = estimator._get_compute_precision(config)

        assert precision == "bf16"
