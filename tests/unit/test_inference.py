"""
Unit tests for inference engine.

Tests InferenceEngine and inference experiment runner.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from llm_efficiency.core.inference import InferenceEngine, run_inference_experiment
from llm_efficiency.config import ExperimentConfig, BatchingConfig, DecoderConfig


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
        )
        
        engine = InferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=mock_config,
        )
        
        assert engine.model == mock_model
        assert engine.tokenizer == mock_tokenizer
        assert engine.config == mock_config

    @patch('llm_efficiency.core.inference.torch.no_grad')
    def test_generate_batch(self, mock_no_grad):
        """Test batch generation."""
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["generated text"]
        
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
            max_output_tokens=10,
            decoder=DecoderConfig(
                max_new_tokens=10,
                temperature=1.0,
                do_sample=False,
            ),
        )
        
        engine = InferenceEngine(mock_model, mock_tokenizer, config)
        
        prompts = ["test prompt"]
        outputs = engine._generate_batch(prompts)
        
        assert len(outputs) == 1
        assert outputs[0] == "generated text"
        mock_model.generate.assert_called_once()

    def test_run_inference_single_batch(self):
        """Test running inference with single batch."""
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["output"]
        
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
            batching=BatchingConfig(batch_size=2),
        )
        
        engine = InferenceEngine(mock_model, mock_tokenizer, config)
        
        prompts = ["prompt1", "prompt2"]
        outputs, metrics = engine.run_inference(prompts, warmup=False)
        
        assert len(outputs) == 2
        assert "total_time_seconds" in metrics
        assert "tokens_per_second" in metrics
        assert metrics["num_prompts"] == 2

    def test_run_inference_multiple_batches(self):
        """Test running inference with multiple batches."""
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["output"]
        
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
            batching=BatchingConfig(batch_size=2),
        )
        
        engine = InferenceEngine(mock_model, mock_tokenizer, config)
        
        prompts = ["p1", "p2", "p3", "p4", "p5"]
        outputs, metrics = engine.run_inference(prompts, batch_size=2, warmup=False)
        
        # Should create 3 batches (2, 2, 1)
        assert len(outputs) == 5
        assert mock_model.generate.call_count >= 3

    def test_warmup_run(self):
        """Test warmup inference."""
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["warmup"]
        
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
        )
        
        engine = InferenceEngine(mock_model, mock_tokenizer, config)
        
        prompts = ["test"]
        outputs, metrics = engine.run_inference(prompts, warmup=True)
        
        # Warmup should run generate at least twice (warmup + actual)
        assert mock_model.generate.call_count >= 2

    def test_metrics_calculation(self):
        """Test metrics are calculated correctly."""
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),  # 2 input tokens
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.batch_decode.return_value = ["output"]
        
        config = ExperimentConfig(
            model_name="test-model",
            precision="float16",
        )
        
        engine = InferenceEngine(mock_model, mock_tokenizer, config)
        
        prompts = ["test"]
        outputs, metrics = engine.run_inference(prompts, warmup=False)
        
        # Check all required metrics present
        assert "total_time_seconds" in metrics
        assert "total_input_tokens" in metrics
        assert "total_output_tokens" in metrics
        assert "total_tokens" in metrics
        assert "num_prompts" in metrics
        assert "tokens_per_second" in metrics
        assert "queries_per_second" in metrics
        assert "avg_latency_per_query" in metrics
        assert "avg_tokens_per_prompt" in metrics
        
        assert metrics["num_prompts"] == 1


class TestRunInferenceExperiment:
    """Tests for run_inference_experiment function."""

    @patch('llm_efficiency.core.inference.InferenceEngine')
    def test_basic_experiment(self, mock_engine_class):
        """Test basic experiment run."""
        mock_engine = Mock()
        mock_engine.run_inference.return_value = (
            ["output1", "output2"],
            {
                "total_time_seconds": 1.0,
                "total_input_tokens": 10,
                "total_output_tokens": 10,
                "total_tokens": 20,
                "num_prompts": 2,
                "tokens_per_second": 20.0,
                "queries_per_second": 2.0,
                "avg_latency_per_query": 0.5,
                "avg_tokens_per_prompt": 10.0,
            },
        )
        mock_engine_class.return_value = mock_engine
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        prompts = ["test1", "test2"]
        config = ExperimentConfig(model_name="test", precision="float16")
        
        outputs, metrics = run_inference_experiment(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompts=prompts,
            config=config,
            warmup=False,
        )
        
        assert len(outputs) == 2
        assert metrics["num_prompts"] == 2
        mock_engine.run_inference.assert_called_once()

    @patch('llm_efficiency.core.inference.InferenceEngine')
    def test_experiment_with_accelerator(self, mock_engine_class):
        """Test experiment with accelerator."""
        mock_engine = Mock()
        mock_engine.run_inference.return_value = (
            ["output"],
            {
                "total_time_seconds": 1.0,
                "total_input_tokens": 5,
                "total_output_tokens": 5,
                "total_tokens": 10,
                "num_prompts": 1,
                "tokens_per_second": 10.0,
                "queries_per_second": 1.0,
                "avg_latency_per_query": 1.0,
                "avg_tokens_per_prompt": 10.0,
            },
        )
        mock_engine_class.return_value = mock_engine
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_accelerator = Mock()
        prompts = ["test"]
        config = ExperimentConfig(model_name="test", precision="float16")
        
        outputs, metrics = run_inference_experiment(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompts=prompts,
            config=config,
            accelerator=mock_accelerator,
            warmup=False,
        )
        
        assert len(outputs) == 1
        # Should synchronize processes
        mock_accelerator.wait_for_everyone.assert_called()

    @patch('llm_efficiency.core.inference.InferenceEngine')
    def test_experiment_with_warmup(self, mock_engine_class):
        """Test experiment with warmup."""
        mock_engine = Mock()
        mock_engine.run_inference.return_value = (
            ["output"],
            {"num_prompts": 1, "tokens_per_second": 10.0},
        )
        mock_engine_class.return_value = mock_engine
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        prompts = ["test"]
        config = ExperimentConfig(model_name="test", precision="float16")
        
        run_inference_experiment(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompts=prompts,
            config=config,
            warmup=True,
        )
        
        # Check warmup was enabled
        call_kwargs = mock_engine.run_inference.call_args[1]
        assert call_kwargs["warmup"] is True
