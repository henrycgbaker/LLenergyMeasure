"""Tests for model loader utilities."""

from unittest.mock import MagicMock, patch

import torch

from llm_bench.core.model_loader import (
    ModelWrapper,
    QuantizationSupport,
    detect_quantization_support,
    get_torch_dtype,
)


class TestGetTorchDtype:
    """Tests for get_torch_dtype."""

    def test_float16(self):
        assert get_torch_dtype("float16") == torch.float16

    def test_bfloat16(self):
        assert get_torch_dtype("bfloat16") == torch.bfloat16

    def test_float32(self):
        assert get_torch_dtype("float32") == torch.float32

    def test_unknown_defaults_to_float16(self):
        assert get_torch_dtype("unknown") == torch.float16

    def test_empty_defaults_to_float16(self):
        assert get_torch_dtype("") == torch.float16


class TestModelWrapper:
    """Tests for ModelWrapper."""

    def test_wraps_model(self):
        inner_model = MagicMock()
        wrapper = ModelWrapper(inner_model)

        assert wrapper.model is inner_model

    def test_forward_calls_model(self):
        inner_model = MagicMock()
        inner_model.return_value = "output"
        wrapper = ModelWrapper(inner_model)

        input_ids = torch.tensor([[1, 2, 3]])
        wrapper.forward(input_ids)

        inner_model.assert_called_once()
        call_kwargs = inner_model.call_args[1]
        assert "input_ids" in call_kwargs


class TestQuantizationSupport:
    """Tests for QuantizationSupport dataclass."""

    def test_creation(self):
        qs = QuantizationSupport(
            supports_4bit=True,
            supports_8bit=True,
            default_4bit_quant_type="nf4",
            default_8bit_quant_type="fp8",
        )

        assert qs.supports_4bit is True
        assert qs.supports_8bit is True
        assert qs.default_4bit_quant_type == "nf4"
        assert qs.default_8bit_quant_type == "fp8"


class TestDetectQuantizationSupport:
    """Tests for detect_quantization_support."""

    def test_no_bitsandbytes(self):
        with (
            patch.dict("sys.modules", {"bitsandbytes": None}),
            patch("importlib.import_module", side_effect=ImportError),
        ):
            result = detect_quantization_support()

        assert result.supports_4bit is False
        assert result.supports_8bit is False
        assert result.default_4bit_quant_type is None
        assert result.default_8bit_quant_type is None

    def test_old_bitsandbytes(self):
        mock_bnb = MagicMock()
        mock_bnb.__version__ = "0.37.0"

        with patch("importlib.import_module", return_value=mock_bnb):
            result = detect_quantization_support()

        assert result.supports_4bit is False
        assert result.supports_8bit is False

    def test_supports_8bit_only(self):
        mock_bnb = MagicMock()
        mock_bnb.__version__ = "0.38.0"

        with patch("importlib.import_module", return_value=mock_bnb):
            result = detect_quantization_support()

        assert result.supports_4bit is False
        assert result.supports_8bit is True

    def test_supports_both(self):
        mock_bnb = MagicMock()
        mock_bnb.__version__ = "0.40.0"

        with patch("importlib.import_module", return_value=mock_bnb):
            result = detect_quantization_support()

        assert result.supports_4bit is True
        assert result.supports_8bit is True
        assert result.default_4bit_quant_type == "nf4"
