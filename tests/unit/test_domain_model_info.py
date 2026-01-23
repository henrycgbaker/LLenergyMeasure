"""Tests for domain model_info models."""

from unittest.mock import MagicMock

from llenergymeasure.domain.model_info import ModelInfo, QuantizationSpec


class TestQuantizationSpec:
    """Tests for QuantizationSpec model."""

    def test_defaults(self):
        spec = QuantizationSpec()
        assert spec.enabled is False
        assert spec.bits is None
        assert spec.method == "none"
        assert spec.compute_dtype == "float16"

    def test_bnb_4bit(self):
        spec = QuantizationSpec(
            enabled=True,
            bits=4,
            method="bitsandbytes",
            compute_dtype="float16",
        )
        assert spec.is_bnb is True
        assert spec.bits == 4

    def test_bnb_8bit(self):
        spec = QuantizationSpec(
            enabled=True,
            bits=8,
            method="bitsandbytes",
        )
        assert spec.is_bnb is True
        assert spec.bits == 8

    def test_non_bnb_method(self):
        spec = QuantizationSpec(
            enabled=True,
            bits=4,
            method="gptq",
        )
        assert spec.is_bnb is False

    def test_flops_reduction_factor_bnb(self):
        """BNB computes at FP16 after dequantization, so no reduction."""
        spec = QuantizationSpec(
            enabled=True,
            bits=4,
            method="bitsandbytes",
        )
        assert spec.flops_reduction_factor == 1.0

    def test_flops_reduction_factor_none(self):
        spec = QuantizationSpec()
        assert spec.flops_reduction_factor == 1.0


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_minimal_creation(self):
        info = ModelInfo(
            name="test-model",
            num_parameters=7_000_000_000,
        )
        assert info.name == "test-model"
        assert info.num_parameters == 7_000_000_000
        assert info.parameters_billions == 7.0

    def test_full_creation(self):
        info = ModelInfo(
            name="meta-llama/Llama-2-7b-hf",
            revision="abc123",
            num_parameters=6_738_415_616,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=32000,
            model_type="llama",
            torch_dtype="float16",
        )
        assert info.num_layers == 32
        assert info.hidden_size == 4096
        assert info.model_type == "llama"
        assert info.is_quantized is False

    def test_with_quantization(self):
        info = ModelInfo(
            name="test-model",
            num_parameters=7_000_000_000,
            quantization=QuantizationSpec(
                enabled=True,
                bits=4,
                method="bitsandbytes",
            ),
        )
        assert info.is_quantized is True
        assert info.quantization.bits == 4

    def test_parameters_billions_property(self):
        info = ModelInfo(
            name="test-model",
            num_parameters=70_000_000_000,  # 70B
        )
        assert info.parameters_billions == 70.0

    def test_from_hf_config(self):
        # Mock a HuggingFace config object
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.vocab_size = 32000
        mock_config.model_type = "llama"

        info = ModelInfo.from_hf_config(
            model_name="meta-llama/Llama-2-7b-hf",
            config=mock_config,
            num_parameters=6_738_415_616,
            torch_dtype="float16",
        )

        assert info.name == "meta-llama/Llama-2-7b-hf"
        assert info.num_layers == 32
        assert info.hidden_size == 4096
        assert info.model_type == "llama"

    def test_from_hf_config_with_quantization(self):
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 32
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_config.vocab_size = 32000
        mock_config.model_type = "llama"

        quant = QuantizationSpec(enabled=True, bits=4, method="bitsandbytes")

        info = ModelInfo.from_hf_config(
            model_name="test-model",
            config=mock_config,
            num_parameters=7_000_000_000,
            quantization=quant,
        )

        assert info.is_quantized is True
        assert info.quantization.bits == 4
