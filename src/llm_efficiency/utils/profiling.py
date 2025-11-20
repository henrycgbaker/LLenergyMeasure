"""
Performance profiling utilities.

Provides decorators and context managers for profiling code execution time,
memory usage, and other performance metrics.
"""

import functools
import logging
import time
import psutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Results from profiling a code block."""
    
    name: str
    duration_seconds: float
    memory_delta_mb: float
    peak_memory_mb: float
    cpu_percent: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "memory_delta_mb": self.memory_delta_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_percent": self.cpu_percent,
            "metadata": self.metadata,
        }


class PerformanceProfiler:
    """
    Performance profiler for tracking execution time and resource usage.
    
    Example:
        profiler = PerformanceProfiler()
        
        with profiler.profile("model_loading"):
            model = load_model(...)
        
        with profiler.profile("inference"):
            outputs = model.generate(...)
        
        profiler.print_summary()
        profiler.save_results("profile_results.json")
    """
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.process = psutil.Process()
    
    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict] = None):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the profiled section
            metadata: Optional metadata to attach to the result
        """
        metadata = metadata or {}
        
        # Capture initial state
        start_time = time.perf_counter()
        mem_info_start = self.process.memory_info()
        start_memory = mem_info_start.rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # Capture final state
            end_time = time.perf_counter()
            mem_info_end = self.process.memory_info()
            end_memory = mem_info_end.rss / 1024 / 1024  # MB
            
            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            peak_memory = max(start_memory, end_memory)
            cpu_percent = self.process.cpu_percent()
            
            # Store result
            result = ProfileResult(
                name=name,
                duration_seconds=duration,
                memory_delta_mb=memory_delta,
                peak_memory_mb=peak_memory,
                cpu_percent=cpu_percent,
                metadata=metadata,
            )
            self.results.append(result)
            
            logger.debug(
                f"Profile '{name}': {duration:.2f}s, "
                f"mem_delta={memory_delta:.1f}MB, "
                f"peak_mem={peak_memory:.1f}MB"
            )
    
    def get_results(self) -> List[ProfileResult]:
        """Get all profiling results."""
        return self.results
    
    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get a specific result by name."""
        for result in self.results:
            if result.name == name:
                return result
        return None
    
    def print_summary(self):
        """Print a summary of all profiling results."""
        if not self.results:
            print("No profiling results available.")
            return
        
        print("\n" + "=" * 80)
        print("Performance Profile Summary")
        print("=" * 80)
        
        total_time = sum(r.duration_seconds for r in self.results)
        
        for result in self.results:
            pct = (result.duration_seconds / total_time * 100) if total_time > 0 else 0
            print(f"\n{result.name}:")
            print(f"  Duration:     {result.duration_seconds:.4f}s ({pct:.1f}%)")
            print(f"  Memory Delta: {result.memory_delta_mb:+.2f} MB")
            print(f"  Peak Memory:  {result.peak_memory_mb:.2f} MB")
            print(f"  CPU:          {result.cpu_percent:.1f}%")
            if result.metadata:
                print(f"  Metadata:     {result.metadata}")
        
        print(f"\nTotal Time: {total_time:.4f}s")
        print("=" * 80 + "\n")
    
    def save_results(self, output_file: Path):
        """Save results to JSON file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total_time": sum(r.duration_seconds for r in self.results),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved profiling results to {output_file}")
    
    def reset(self):
        """Clear all results."""
        self.results = []


def profile_function(name: Optional[str] = None):
    """
    Decorator for profiling function execution.
    
    Example:
        @profile_function("my_function")
        def expensive_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            with profiler.profile(func_name):
                result = func(*args, **kwargs)
            
            # Log the result
            prof_result = profiler.get_result(func_name)
            if prof_result:
                logger.info(
                    f"{func_name}: {prof_result.duration_seconds:.4f}s, "
                    f"mem={prof_result.memory_delta_mb:+.1f}MB"
                )
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def timer(name: str = "Operation"):
    """
    Simple timer context manager.
    
    Example:
        with timer("Model loading"):
            model = load_model(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"{name} took {duration:.4f}s")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in MB
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / 1024 / 1024,
        "vms_mb": mem_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
    }


def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    process = psutil.Process()
    return process.cpu_percent(interval=0.1)
