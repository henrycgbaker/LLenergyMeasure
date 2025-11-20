"""
Unit tests for profiling utilities.

Tests performance profiling, timers, and resource monitoring.
"""

import pytest
import time
from pathlib import Path

from llm_efficiency.utils.profiling import (
    PerformanceProfiler,
    ProfileResult,
    profile_function,
    timer,
)


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_creation(self):
        """Test creating a profile result."""
        result = ProfileResult(
            name="test_operation",
            duration_seconds=1.5,
            memory_delta_mb=50.0,
            peak_memory_mb=500.0,
            cpu_percent=75.0,
            metadata={"iterations": 100},
        )
        
        assert result.name == "test_operation"
        assert result.duration_seconds == 1.5
        assert result.memory_delta_mb == 50.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ProfileResult(
            name="test",
            duration_seconds=1.0,
            memory_delta_mb=10.0,
            peak_memory_mb=100.0,
            cpu_percent=50.0,
        )
        
        data = result.to_dict()
        
        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["duration_seconds"] == 1.0


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        
        assert len(profiler.results) == 0
        assert profiler.process is not None

    def test_profile_context(self):
        """Test profiling a code block."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("sleep_test"):
            time.sleep(0.1)
        
        assert len(profiler.results) == 1
        result = profiler.results[0]
        assert result.name == "sleep_test"
        assert result.duration_seconds >= 0.1
        assert result.duration_seconds < 0.2  # Allow some overhead

    def test_profile_with_metadata(self):
        """Test profiling with metadata."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test", metadata={"key": "value"}):
            pass
        
        result = profiler.results[0]
        assert result.metadata == {"key": "value"}

    def test_multiple_profiles(self):
        """Test multiple profiling sessions."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("op1"):
            time.sleep(0.05)
        
        with profiler.profile("op2"):
            time.sleep(0.05)
        
        assert len(profiler.results) == 2
        assert profiler.results[0].name == "op1"
        assert profiler.results[1].name == "op2"

    def test_get_result(self):
        """Test getting a specific result."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("target"):
            pass
        
        result = profiler.get_result("target")
        assert result is not None
        assert result.name == "target"
        
        assert profiler.get_result("nonexistent") is None

    def test_save_results(self, tmp_path):
        """Test saving results to file."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test"):
            pass
        
        output_file = tmp_path / "profile.json"
        profiler.save_results(output_file)
        
        assert output_file.exists()

    def test_reset(self):
        """Test resetting profiler."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test"):
            pass
        
        assert len(profiler.results) > 0
        
        profiler.reset()
        
        assert len(profiler.results) == 0


class TestProfileFunction:
    """Tests for profile_function decorator."""

    def test_decorator_basic(self):
        """Test basic function profiling."""
        @profile_function("test_func")
        def sample_function():
            time.sleep(0.05)
            return "result"
        
        result = sample_function()
        
        assert result == "result"

    def test_decorator_with_args(self):
        """Test profiling function with arguments."""
        @profile_function()
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5


class TestTimer:
    """Tests for timer context manager."""

    def test_timer_basic(self):
        """Test basic timer usage."""
        # Just verify it doesn't crash
        with timer("test operation"):
            time.sleep(0.05)
        
        # Test completes without error
        assert True
