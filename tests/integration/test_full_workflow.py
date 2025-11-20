"""
Integration test for complete end-to-end workflow.

Tests the full pipeline from configuration to results.
"""

import pytest
import torch
from pathlib import Path

from llm_efficiency.config import ExperimentConfig, BatchingConfig
from llm_efficiency.core import (
    setup_accelerator,
    generate_experiment_id,
    load_model_and_tokenizer,
    run_inference_experiment,
)
from llm_efficiency.data import load_prompts_from_dataset, filter_prompts_by_length
from llm_efficiency.metrics import FLOPsCalculator, EnergyTracker, get_gpu_memory_stats
from llm_efficiency.storage import ResultsManager, create_results, ModelInfo
from llm_efficiency.utils.logging import setup_logging


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflow."""

    def test_minimal_workflow(self, tmp_path):
        """Test minimal end-to-end workflow."""
        # Setup logging
        setup_logging(level="INFO")

        # Configuration
        config = ExperimentConfig(
            model_name="hf-internal-testing/tiny-random-gpt2",
            precision="float16",
            num_input_prompts=5,
            max_input_tokens=32,
            max_output_tokens=16,
            batching=BatchingConfig(batch_size=2),
            results_dir=str(tmp_path / "results"),
        )

        # Load model
        model, tokenizer = load_model_and_tokenizer(config)
        device = next(model.parameters()).device

        # Prepare test prompts
        test_prompts = [
            "Hello, how are you?",
            "What is AI?",
            "Tell me a story.",
            "How does Python work?",
            "Explain machine learning.",
        ]

        # Filter by length
        filtered_prompts = filter_prompts_by_length(
            test_prompts, tokenizer, max_tokens=config.max_input_tokens
        )

        assert len(filtered_prompts) > 0

        # Run inference
        outputs, inference_metrics = run_inference_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=filtered_prompts,
            config=config,
            warmup=True,
        )

        # Validate inference results
        assert len(outputs) == len(filtered_prompts)
        assert inference_metrics["tokens_per_second"] > 0
        assert inference_metrics["num_prompts"] == len(filtered_prompts)

        # Calculate FLOPs
        calculator = FLOPsCalculator(cache_dir=tmp_path / "flops_cache")
        flops = calculator.get_flops(
            model=model,
            model_name=config.model_name,
            sequence_length=config.max_input_tokens,
            device=device,
            is_quantized=config.quantization.enabled,
        )

        assert flops > 0

        # Get memory stats
        if torch.cuda.is_available():
            memory_stats = get_gpu_memory_stats(device)
            assert "gpu_current_memory_allocated_bytes" in memory_stats

        # Create results
        results = create_results(
            experiment_id="test_001",
            config=config.to_dict(),
            model_info={
                "model_name": config.model_name,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "precision": config.precision,
            },
            inference_metrics=inference_metrics,
            compute_metrics={
                "flops": flops,
                "gpu_memory_allocated_mb": 0.0,
                "gpu_memory_peak_mb": 0.0,
            },
            outputs=outputs if config.save_outputs else None,
        )

        # Save results
        manager = ResultsManager(results_dir=tmp_path / "results")
        saved_path = manager.save_experiment(results)

        assert saved_path.exists()

        # Load and verify
        loaded = manager.load_experiment("test_001")
        assert loaded is not None
        assert loaded.experiment_id == "test_001"
        assert loaded.inference.tokens_per_second > 0

    @pytest.mark.slow
    def test_with_energy_tracking(self, tmp_path):
        """Test workflow with energy tracking."""
        config = ExperimentConfig(
            model_name="hf-internal-testing/tiny-random-gpt2",
            precision="float16",
            num_input_prompts=3,
            max_input_tokens=16,
            max_output_tokens=8,
            batching=BatchingConfig(batch_size=1),
        )

        # Load model
        model, tokenizer = load_model_and_tokenizer(config)

        # Test prompts
        test_prompts = ["Hello", "World", "Test"]

        # Energy tracking
        with EnergyTracker(
            experiment_id="test_energy_001",
            output_dir=tmp_path / "energy",
        ) as tracker:
            # Run inference
            outputs, metrics = run_inference_experiment(
                model, tokenizer, test_prompts, config, warmup=False
            )

        # Get energy results
        energy_results = tracker.get_results()

        assert energy_results["duration_seconds"] > 0
        assert energy_results["energy_consumed_kwh"] >= 0

    @pytest.mark.integration
    def test_full_pipeline_with_accelerate(self, tmp_path):
        """Test complete pipeline with Accelerate."""
        # Setup Accelerate
        accelerator = setup_accelerator(mixed_precision="no")

        # Generate experiment ID
        exp_id = generate_experiment_id(accelerator, id_file=tmp_path / "exp_id.txt")
        assert len(exp_id) == 4  # Zero-padded 4 digits

        # Configuration
        config = ExperimentConfig(
            model_name="hf-internal-testing/tiny-random-gpt2",
            precision="float16",
            num_input_prompts=4,
            batching=BatchingConfig(batch_size=2),
        )

        # Load and prepare model
        model, tokenizer = load_model_and_tokenizer(config)
        model = accelerator.prepare(model)

        # Prompts
        test_prompts = ["A", "B", "C", "D"]

        # Inference
        outputs, metrics = run_inference_experiment(
            model, tokenizer, test_prompts, config, accelerator, warmup=False
        )

        assert len(outputs) == 4
        assert metrics["tokens_per_second"] > 0

    def test_results_aggregation(self, tmp_path):
        """Test results aggregation across multiple experiments."""
        manager = ResultsManager(results_dir=tmp_path / "results")

        # Create multiple experiments
        for i in range(3):
            results = create_results(
                experiment_id=f"test_{i:03d}",
                config={"model_name": "test-model"},
                inference_metrics={
                    "total_time_seconds": 1.0 + i,
                    "total_input_tokens": 100,
                    "total_output_tokens": 100,
                    "total_tokens": 200,
                    "num_prompts": 10,
                    "tokens_per_second": 200.0 / (1.0 + i),
                    "queries_per_second": 10.0 / (1.0 + i),
                    "avg_latency_per_query": (1.0 + i) / 10,
                    "avg_tokens_per_prompt": 20.0,
                },
            )
            manager.save_experiment(results)

        # List experiments
        exp_ids = manager.list_experiments()
        assert len(exp_ids) == 3

        # Aggregate
        aggregated = manager.aggregate_experiments()
        assert len(aggregated) == 3

        # Export to CSV
        csv_path = tmp_path / "results.csv"
        manager.export_to_csv(csv_path)
        assert csv_path.exists()

        # Generate summary
        summary = manager.generate_summary()
        assert summary["total_experiments"] == 3
        assert "throughput" in summary
