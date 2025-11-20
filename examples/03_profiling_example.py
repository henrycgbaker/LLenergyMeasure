"""
Performance Profiling Example
==============================

This example demonstrates how to use the built-in performance profiling
utilities to track execution time, memory usage, and CPU utilization.

Features demonstrated:
- Using the PerformanceProfiler class
- Context manager profiling
- Function decorator profiling
- Generating performance reports
- Exporting profiling data
"""

import time
import numpy as np
from pathlib import Path

from llm_efficiency.utils import (
    PerformanceProfiler,
    profile_function,
    timer,
    get_memory_usage,
    get_cpu_usage,
)


# Example 1: Using PerformanceProfiler class
def example_profiler_class():
    """Demonstrate PerformanceProfiler for tracking multiple operations."""

    print("=" * 70)
    print("Example 1: PerformanceProfiler Class")
    print("=" * 70)

    profiler = PerformanceProfiler()

    # Profile multiple operations
    with profiler.profile("data_loading", metadata={"size": 1000}):
        # Simulate data loading
        data = np.random.randn(1000, 1000)
        time.sleep(0.1)

    with profiler.profile("preprocessing", metadata={"operations": ["normalize", "scale"]}):
        # Simulate preprocessing
        normalized = (data - data.mean()) / data.std()
        scaled = normalized * 100
        time.sleep(0.05)

    with profiler.profile("computation", metadata={"algorithm": "matrix_multiply"}):
        # Simulate heavy computation
        result = np.matmul(data, data.T)
        time.sleep(0.2)

    # Print summary
    print("\n" + profiler.get_summary())

    # Get specific statistics
    total_time = sum(r.duration_seconds for r in profiler.results)
    total_memory = sum(r.memory_delta_mb for r in profiler.results)

    print(f"\nTotal execution time: {total_time:.3f} seconds")
    print(f"Total memory delta: {total_memory:.2f} MB")

    # Export to JSON
    output_file = Path("./profiling_results.json")
    profiler.save(output_file)
    print(f"\nResults saved to: {output_file}")

    return profiler


# Example 2: Using the @profile_function decorator
@profile_function(name="expensive_operation")
def expensive_operation(size: int):
    """Example function with automatic profiling."""
    # Simulate expensive computation
    data = np.random.randn(size, size)
    result = np.linalg.svd(data)
    return result


def example_decorator():
    """Demonstrate function decorator profiling."""

    print("\n" + "=" * 70)
    print("Example 2: Function Decorator Profiling")
    print("=" * 70)

    # These calls are automatically profiled
    print("\nRunning expensive_operation(100)...")
    expensive_operation(100)

    print("\nRunning expensive_operation(500)...")
    expensive_operation(500)

    print("\n✓ Function execution profiled automatically!")


# Example 3: Using timer context manager
def example_timer():
    """Demonstrate timer context manager for simple timing."""

    print("\n" + "=" * 70)
    print("Example 3: Timer Context Manager")
    print("=" * 70)

    # Time a specific block of code
    with timer("database_query"):
        # Simulate database query
        time.sleep(0.5)
        results = list(range(10000))

    with timer("data_processing"):
        # Simulate data processing
        processed = [x * 2 for x in results]
        time.sleep(0.3)

    print("\n✓ Operations timed successfully!")


# Example 4: Memory and CPU monitoring
def example_resource_monitoring():
    """Demonstrate real-time resource monitoring."""

    print("\n" + "=" * 70)
    print("Example 4: Resource Monitoring")
    print("=" * 70)

    print("\nInitial resource usage:")
    print(f"  Memory: {get_memory_usage():.2f} MB")
    print(f"  CPU: {get_cpu_usage():.1f}%")

    # Allocate memory
    print("\nAllocating 100MB array...")
    large_array = np.random.randn(100 * 1024 * 1024 // 8)  # ~100MB

    print(f"  Memory after allocation: {get_memory_usage():.2f} MB")

    # CPU-intensive task
    print("\nRunning CPU-intensive task...")
    for _ in range(3):
        _ = np.linalg.svd(np.random.randn(1000, 1000))

    print(f"  CPU during computation: {get_cpu_usage():.1f}%")

    # Cleanup
    del large_array
    print("\n✓ Resource monitoring complete!")


# Example 5: Profiling LLM inference
def example_llm_profiling():
    """Demonstrate profiling an LLM experiment workflow."""

    print("\n" + "=" * 70)
    print("Example 5: LLM Experiment Profiling")
    print("=" * 70)

    profiler = PerformanceProfiler()

    # Simulate different stages of LLM inference
    with profiler.profile("model_loading", metadata={"model": "gpt2"}):
        # Simulate model loading (normally takes seconds)
        time.sleep(0.5)

    with profiler.profile("tokenization", metadata={"num_samples": 100}):
        # Simulate tokenization
        time.sleep(0.1)

    with profiler.profile("inference", metadata={"batch_size": 8, "num_batches": 10}):
        # Simulate inference (normally the slowest part)
        time.sleep(1.0)

    with profiler.profile("metrics_calculation"):
        # Simulate metrics calculation
        time.sleep(0.2)

    with profiler.profile("results_saving"):
        # Simulate saving results
        time.sleep(0.1)

    # Print detailed summary
    print("\n" + profiler.get_summary())

    # Analyze bottlenecks
    print("\n--- Performance Analysis ---")
    results_sorted = sorted(
        profiler.results,
        key=lambda r: r.duration_seconds,
        reverse=True
    )

    print("\nTop 3 bottlenecks:")
    for i, result in enumerate(results_sorted[:3], 1):
        print(f"  {i}. {result.name}: {result.duration_seconds:.3f}s")

    # Calculate stage percentages
    total_time = sum(r.duration_seconds for r in profiler.results)
    print("\nTime breakdown:")
    for result in profiler.results:
        percentage = (result.duration_seconds / total_time) * 100
        print(f"  {result.name}: {percentage:.1f}%")

    return profiler


def main():
    """Run all profiling examples."""

    print("\n" + "=" * 70)
    print("PERFORMANCE PROFILING EXAMPLES")
    print("=" * 70)

    # Run examples
    example_profiler_class()
    example_decorator()
    example_timer()
    example_resource_monitoring()
    example_llm_profiling()

    print("\n" + "=" * 70)
    print("PROFILING EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Use PerformanceProfiler for comprehensive profiling")
    print("  2. Use @profile_function for automatic function profiling")
    print("  3. Use timer() for simple timing of code blocks")
    print("  4. Use get_memory_usage() and get_cpu_usage() for monitoring")
    print("  5. Export results to JSON for further analysis")
    print("\nFor production use:")
    print("  - Profile model loading, inference, and metrics calculation")
    print("  - Identify bottlenecks and optimize hot paths")
    print("  - Track memory growth to prevent OOM errors")
    print("  - Monitor CPU usage for parallel processing opportunities")
    print("=" * 70)


if __name__ == "__main__":
    main()
