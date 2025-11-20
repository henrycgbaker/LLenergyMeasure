"""
Unit tests for model loading utilities.

Tests model and tokenizer loading with various configurations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from llm_efficiency.core.model_loader import (
    load_model_and_tokenizer,
    detect_quantization_support,
)
from llm_efficiency.config import ExperimentConfig, QuantizationConfig


class TestDetectQuantizationSupport:
    """Tests for quantization support detection."""

    @patch('llm_efficiency.core.model_loader.torch.cuda.is_available')
    def test_cuda_available(self, mock_cuda):
        """Test quantization support when CUDA is available."""
        mock_cuda.return_value = True
        
        assert detect_quantization_support() is True

    @patch('llm_efficiency.core.model_loader.torch.cuda.is_available')
    def test_cuda_not_available(self, mock_cuda):
        """Test quantization support when CUDA is not available."""
        mock_cuda.return_value = False
        
        assert detect_quantization_support() is False


class TestLoadModelAndTokenizer:
    """Tests for model and tokenizer loading."""

    @patch('llm_efficiency.core.model_loader.AutoTokenizer')
    @patch('llm_efficiency.core.model_loader.AutoModelForCausalLM')
    def test_load_basic_model(self, mock_model_class, mock_tokenizer_class):
        """Test loading basic model without quantization."""
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
            quantization=QuantizationConfig(enabled=False),
        )
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch('llm_efficiency.core.model_loader.AutoTokenizer')
    @patch('llm_efficiency.core.model_loader.AutoModelForCausalLM')
    @patch('llm_efficiency.core.model_loader.detect_quantization_support')
    def test_load_quantized_model(self, mock_detect, mock_model_class, mock_tokenizer_class):
        """Test loading quantized model."""
        mock_detect.return_value = True
        
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
            quantization=QuantizationConfig(enabled=True, bits=4),
        )
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        
        # Check quantization config was passed
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert "quantization_config" in call_kwargs

    @patch('llm_efficiency.core.model_loader.AutoTokenizer')
    @patch('llm_efficiency.core.model_loader.AutoModelForCausalLM')
    def test_tokenizer_padding(self, mock_model_class, mock_tokenizer_class):
        """Test tokenizer padding token is set."""
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
        )
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Pad token should be set to eos_token
        assert mock_tokenizer.pad_token == "<eos>"

    @patch('llm_efficiency.core.model_loader.AutoTokenizer')
    @patch('llm_efficiency.core.model_loader.AutoModelForCausalLM')
    def test_precision_float32(self, mock_model_class, mock_tokenizer_class):
        """Test model loading with float32 precision."""
        config = ExperimentConfig(
            model_name="test-model",
            precision="float32",
        )
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        load_model_and_tokenizer(config)
        
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float32

    @patch('llm_efficiency.core.model_loader.AutoTokenizer')
    @patch('llm_efficiency.core.model_loader.AutoModelForCausalLM')
    def test_precision_float16(self, mock_model_class, mock_tokenizer_class):
        """Test model loading with float16 precision."""
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
        )
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        load_model_and_tokenizer(config)
        
        call_kwargs = mock_model_class.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float16
