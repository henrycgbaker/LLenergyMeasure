"""
Advanced Workflow Example
==========================

This example demonstrates advanced usage patterns combining multiple
features for production LLM efficiency benchmarking.

Features demonstrated:
- Automated multi-configuration benchmarking
- Performance profiling integration
- Caching for efficiency
- Custom metrics and reporting
- Production-ready workflow
"""

import time
from pathlib import Path
from typing import List, Dict, Any
import json

from llm_efficiency.config import ExperimentConfig, QuantizationConfig
from llm_efficiency.core.experiment import run_experiment
from llm_efficiency.storage.results import ExperimentResult, ResultsManager
from llm_efficiency.utils import (
    PerformanceProfiler,
    DiskCache,
    retry_with_exponential_backoff,
)


class BenchmarkSuite:
    """
    Advanced benchmark suite for comprehensive LLM efficiency evaluation.

    Combines profiling, caching, and batch experimentation.
    """

    def __init__(self, output_dir: Path, cache_dir: Path):
        """
        Initialize benchmark suite.

        Args:
            output_dir: Directory for experiment results
            cache_dir: Directory for caching
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.profiler = PerformanceProfiler()
        self.cache = DiskCache(cache_dir=cache_dir, ttl=86400)  # 24h cache
        self.results: List[ExperimentResult] = []

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @retry_with_exponential_backoff(max_retries=3, initial_delay=2.0)
    def run_single_experiment(
        self,
        config: ExperimentConfig,
        use_cache: bool = True
    ) -> ExperimentResult:
        """
        Run a single experiment with caching and profiling.

        Args:
            config: Experiment configuration
            use_cache: Whether to use cached results

        Returns:
            Experiment result
        """

        # Create cache key from config
        cache_key = self._create_cache_key(config)

        # Check cache first
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                print(f"  ✓ Using cached result for {config.model_name}")
                return cached_result

        # Run experiment with profiling
        with self.profiler.profile(
            f"experiment_{config.model_name}",
            metadata={
                "model": config.model_name,
                "precision": config.precision,
                "quantization": config.quantization.enabled,
            }
        ):
            result = run_experiment(config)

        # Cache the result
        if use_cache:
            self.cache.set(cache_key, result)

        self.results.append(result)
        return result

    def run_benchmark_suite(
        self,
        models: List[str],
        precisions: List[str],
        quantization_configs: List[Dict[str, Any]],
    ) -> List[ExperimentResult]:
        """
        Run comprehensive benchmark across multiple configurations.

        Args:
            models: List of model names
            precisions: List of precision types
            quantization_configs: List of quantization configurations

        Returns:
            List of all experiment results
        """

        total_experiments = len(models) * len(precisions) * len(quantization_configs)

        print(f"\n{'=' * 70}")
        print(f"BENCHMARK SUITE: {total_experiments} experiments")
        print(f"{'=' * 70}")
        print(f"Models: {len(models)}")
        print(f"Precisions: {len(precisions)}")
        print(f"Quantization configs: {len(quantization_configs)}")

        experiment_count = 0

        for model in models:
            for precision in precisions:
                for quant_config in quantization_configs:
                    experiment_count += 1

                    print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                    print(f"Model: {model}")
                    print(f"Precision: {precision}")
                    print(f"Quantization: {quant_config}")

                    # Create configuration
                    config = ExperimentConfig(
                        model_name=model,
                        precision=precision,
                        quantization=QuantizationConfig(**quant_config),
                        batch_size=4,
                        num_batches=20,
                        max_length=128,
                        output_dir=self.output_dir / f"{model}_{precision}".replace("/", "_"),
                    )

                    try:
                        # Run experiment
                        start_time = time.time()
                        result = self.run_single_experiment(config)
                        elapsed = time.time() - start_time

                        print(f"✓ Completed in {elapsed:.1f}s")
                        print(f"  Throughput: {result.metrics.tokens_per_second:.2f} tokens/sec")
                        print(f"  Energy: {result.metrics.total_energy_kwh:.6f} kWh")

                    except Exception as e:
                        print(f"✗ Failed: {e}")
                        continue

        return self.results

    def generate_comprehensive_report(self, output_file: Path):
        """
        Generate comprehensive benchmark report.

        Args:
            output_file: Path to save report
        """

        if not self.results:
            print("No results to report")
            return

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nTotal experiments: {len(self.results)}")

        # Performance summary
        report_lines.append("\n--- PERFORMANCE SUMMARY ---")
        report_lines.append(self.profiler.get_summary())

        # Top performers
        report_lines.append("\n--- TOP PERFORMERS ---")

        fastest = max(self.results, key=lambda r: r.metrics.tokens_per_second)
        report_lines.append(f"\nFastest:")
        report_lines.append(f"  {fastest.config.model_name} ({fastest.config.precision})")
        report_lines.append(f"  {fastest.metrics.tokens_per_second:.2f} tokens/sec")

        most_efficient = min(self.results, key=lambda r: r.metrics.energy_per_token)
        report_lines.append(f"\nMost Energy Efficient:")
        report_lines.append(f"  {most_efficient.config.model_name} ({most_efficient.config.precision})")
        report_lines.append(f"  {most_efficient.metrics.energy_per_token:.8f} kWh/token")

        # Group by model
        report_lines.append("\n--- BY MODEL ---")

        by_model: Dict[str, List[ExperimentResult]] = {}
        for result in self.results:
            model = result.config.model_name
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)

        for model, model_results in by_model.items():
            report_lines.append(f"\n{model}:")
            for r in model_results:
                config_str = f"{r.config.precision}"
                if r.config.quantization.enabled:
                    if r.config.quantization.load_in_4bit:
                        config_str += "+4bit"
                    elif r.config.quantization.load_in_8bit:
                        config_str += "+8bit"

                report_lines.append(
                    f"  {config_str:20} "
                    f"{r.metrics.tokens_per_second:8.2f} tok/s  "
                    f"{r.metrics.energy_per_token:12.8f} kWh/tok"
                )

        # Cache statistics
        report_lines.append("\n--- CACHE STATISTICS ---")
        report_lines.append(f"Cache hits: {self.cache.hits}")
        report_lines.append(f"Cache misses: {self.cache.misses}")
        report_lines.append(f"Hit rate: {self.cache.hit_rate():.1%}")

        report_lines.append("\n" + "=" * 80)

        # Save report
        report_text = "\n".join(report_lines)
        print(report_text)

        output_file.write_text(report_text)
        print(f"\nReport saved to: {output_file}")

        # Save profiling data
        profile_file = output_file.with_suffix('.profile.json')
        self.profiler.save(profile_file)
        print(f"Profiling data saved to: {profile_file}")

    def _create_cache_key(self, config: ExperimentConfig) -> str:
        """Create cache key from configuration."""
        key_parts = [
            config.model_name,
            config.precision,
            str(config.quantization.enabled),
            str(config.quantization.load_in_4bit),
            str(config.quantization.load_in_8bit),
            str(config.batch_size),
            str(config.num_batches),
            str(config.max_length),
        ]
        return ":".join(key_parts)


def example_automated_benchmark():
    """Run automated multi-configuration benchmark."""

    print("=" * 70)
    print("ADVANCED WORKFLOW: Automated Benchmarking")
    print("=" * 70)

    # Initialize benchmark suite
    suite = BenchmarkSuite(
        output_dir=Path("./results/advanced_benchmark"),
        cache_dir=Path("./cache/benchmark_cache"),
    )

    # Define configurations to test
    models = [
        "gpt2",
        # "gpt2-medium",  # Add more models as needed
    ]

    precisions = [
        "float16",
        # "float32",  # Add if testing precision impact
    ]

    quantization_configs = [
        {"enabled": False},  # Baseline
        # {"enabled": True, "load_in_8bit": True},  # Uncomment if GPU available
        # {"enabled": True, "load_in_4bit": True, "quant_type": "nf4"},
    ]

    # Run benchmark suite
    results = suite.run_benchmark_suite(
        models=models,
        precisions=precisions,
        quantization_configs=quantization_configs,
    )

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    report_file = Path("./results/advanced_benchmark/comprehensive_report.txt")
    suite.generate_comprehensive_report(report_file)

    return suite


def example_custom_metrics():
    """Demonstrate custom metrics calculation."""

    print("\n" + "=" * 70)
    print("CUSTOM METRICS CALCULATION")
    print("=" * 70)

    # Load existing results
    manager = ResultsManager(results_dir=Path("./results"))
    experiments = manager.list_experiments()

    if not experiments:
        print("\nNo experiments found. Run benchmarks first.")
        return

    # Calculate custom metrics
    for exp_id in experiments[:3]:  # First 3 experiments
        result = manager.load_result(exp_id)

        print(f"\n{result.config.model_name}:")

        # Custom metric 1: Cost efficiency (tokens per dollar)
        # Assuming $0.12 per kWh
        energy_cost = result.metrics.total_energy_kwh * 0.12
        tokens_per_dollar = result.metrics.total_tokens / energy_cost if energy_cost > 0 else 0
        print(f"  Tokens per dollar: {tokens_per_dollar:,.0f}")

        # Custom metric 2: Carbon efficiency (tokens per kg CO2)
        tokens_per_kg_co2 = (
            result.metrics.total_tokens / result.metrics.co2_emissions
            if result.metrics.co2_emissions > 0
            else 0
        )
        print(f"  Tokens per kg CO2: {tokens_per_kg_co2:,.0f}")

        # Custom metric 3: Computational efficiency (tokens per TFLOP)
        tflops = result.metrics.total_flops / 1e12
        tokens_per_tflop = result.metrics.total_tokens / tflops if tflops > 0 else 0
        print(f"  Tokens per TFLOP: {tokens_per_tflop:.2f}")

        # Custom metric 4: Normalized efficiency score
        # Combine multiple metrics into single score
        score = (
            result.metrics.tokens_per_second * 0.4 +  # Throughput weight
            (1.0 / result.metrics.energy_per_token) * 0.4 +  # Energy weight
            (1.0 / result.metrics.latency_per_token) * 0.2  # Latency weight
        )
        print(f"  Efficiency score: {score:.2f}")


def main():
    """Run advanced workflow examples."""

    print("\n" + "=" * 70)
    print("ADVANCED WORKFLOW EXAMPLES")
    print("=" * 70)

    # Run automated benchmark
    suite = example_automated_benchmark()

    # Calculate custom metrics
    example_custom_metrics()

    print("\n" + "=" * 70)
    print("ADVANCED WORKFLOW COMPLETE!")
    print("=" * 70)
    print("\nKey features demonstrated:")
    print("  1. Automated multi-configuration benchmarking")
    print("  2. Performance profiling integration")
    print("  3. Intelligent caching for efficiency")
    print("  4. Comprehensive reporting")
    print("  5. Custom metrics calculation")
    print("\nProduction tips:")
    print("  - Use BenchmarkSuite for systematic testing")
    print("  - Enable caching to avoid redundant experiments")
    print("  - Profile experiments to identify bottlenecks")
    print("  - Calculate custom metrics for your use case")
    print("  - Automate report generation")
    print("  - Set up CI/CD for regular benchmarking")
    print("=" * 70)


if __name__ == "__main__":
    main()
