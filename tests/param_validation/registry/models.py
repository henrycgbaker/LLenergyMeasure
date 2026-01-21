"""Core data models for the parameter validation framework.

Defines ParamSpec, VerificationResult, and related types used throughout
the framework for declarative parameter testing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class VerificationType(Enum):
    """Types of verification that can be performed on a parameter."""

    PASSTHROUGH = auto()  # Verify param reaches backend config
    BEHAVIOUR = auto()  # Verify observable output/perf change
    INTROSPECTION = auto()  # Inspect model/engine state
    MOCK = auto()  # CI-safe via patching (no GPU)


class HardwareRequirement(Enum):
    """Hardware/software requirements for running a test."""

    GPU = auto()  # Any CUDA GPU
    HOPPER = auto()  # Hopper (SM 9.0+) for FP8
    AMPERE = auto()  # Ampere (SM 8.0+) for BF16
    VLLM = auto()  # vLLM installed
    TENSORRT = auto()  # TensorRT-LLM installed
    FLASH_ATTN = auto()  # Flash Attention installed
    BF16 = auto()  # BF16 support
    MULTI_GPU = auto()  # Multiple GPUs for parallelism tests


class VerificationStatus(Enum):
    """Status of a verification result."""

    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()
    ERROR = auto()


@dataclass
class VerificationResult:
    """Result of a parameter verification."""

    status: VerificationStatus
    message: str
    param_name: str
    test_value: Any
    actual_value: Any | None = None
    expected_value: Any | None = None
    error: Exception | None = None
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == VerificationStatus.FAILED


class PassthroughChecker(Protocol):
    """Protocol for passthrough checking functions."""

    def __call__(self, instance: Any, expected: Any) -> tuple[bool, str]: ...


class BehaviourAssertion(Protocol):
    """Protocol for behaviour assertion functions."""

    def __call__(
        self, baseline_output: Any, test_output: Any, test_value: Any
    ) -> tuple[bool, str]: ...


@dataclass
class ParamSpec:
    """Declarative specification for a parameter to be tested.

    Defines all metadata needed to generate and run parameter validation
    tests across different backends.

    Attributes:
        name: Full parameter path (e.g., "vllm.max_num_seqs").
        backend: Backend this param belongs to ("vllm", "pytorch", "tensorrt", "shared").
        config_path: Path to set in experiment config (e.g., "vllm.max_num_seqs").
        test_values: List of values to test (derived from Literal/constraints).
        verification_type: Type of verification to perform.
        hardware_requirements: Set of hardware requirements for this test.
        passthrough_path: Dot-separated path to check in backend instance
            (e.g., "llm_engine.scheduler_config.max_num_seqs").
        passthrough_checker: Custom function to check passthrough (overrides path).
        behaviour_assertion: Function to assert behaviour change.
        description: Human-readable description of what this param does.
        category: Grouping category (e.g., "memory", "batching", "kv_cache").
        energy_impact: Whether this param has significant energy impact.
        default_value: The default value from the Pydantic model.
        pydantic_field_info: Original Pydantic FieldInfo for reference.
        skip_reason: If set, skip this param with this reason.
        expected_transform: Function to transform expected value before comparison.
    """

    name: str
    backend: str
    config_path: str
    test_values: list[Any]
    verification_type: VerificationType = VerificationType.PASSTHROUGH
    hardware_requirements: set[HardwareRequirement] = field(
        default_factory=lambda: {HardwareRequirement.GPU}
    )
    passthrough_path: str | None = None
    passthrough_checker: PassthroughChecker | None = None
    behaviour_assertion: BehaviourAssertion | None = None
    description: str = ""
    category: str = "general"
    energy_impact: bool = False
    default_value: Any = None
    pydantic_field_info: Any = None
    skip_reason: str | None = None
    expected_transform: Callable[[Any], Any] | None = None

    @property
    def full_name(self) -> str:
        """Full qualified name including backend."""
        return f"{self.backend}.{self.name}"

    @property
    def is_nested(self) -> bool:
        """Whether this param is nested (contains dots in name)."""
        return "." in self.name

    @property
    def parent_path(self) -> str | None:
        """Get parent path for nested params."""
        if not self.is_nested:
            return None
        return self.name.rsplit(".", 1)[0]

    def requires_gpu(self) -> bool:
        """Check if this param requires a GPU to test."""
        return HardwareRequirement.GPU in self.hardware_requirements

    def can_mock(self) -> bool:
        """Check if this param can be tested via mocking (no GPU)."""
        return self.verification_type == VerificationType.MOCK or (
            self.verification_type == VerificationType.PASSTHROUGH
            and self.passthrough_path is not None
        )


@dataclass
class BackendCapabilities:
    """Capabilities detected for a specific backend."""

    backend: str
    available: bool
    version: tuple[int, int, int] | None = None
    supported_params: set[str] = field(default_factory=set)
    unsupported_params: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParamCoverage:
    """Coverage report for parameter testing."""

    total_params: int
    tested_params: int
    skipped_params: int
    coverage_by_backend: dict[str, float] = field(default_factory=dict)
    coverage_by_category: dict[str, float] = field(default_factory=dict)
    untested_params: list[str] = field(default_factory=list)

    @property
    def coverage_percent(self) -> float:
        if self.total_params == 0:
            return 100.0
        return (self.tested_params / self.total_params) * 100


@dataclass
class RunSummary:
    """Summary of a parameter validation test run."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    results: list[VerificationResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        executed = self.passed + self.failed
        if executed == 0:
            return 100.0
        return (self.passed / executed) * 100

    def add_result(self, result: VerificationResult) -> None:
        self.results.append(result)
        self.total_tests += 1
        self.duration_ms += result.duration_ms

        match result.status:
            case VerificationStatus.PASSED:
                self.passed += 1
            case VerificationStatus.FAILED:
                self.failed += 1
            case VerificationStatus.SKIPPED:
                self.skipped += 1
            case VerificationStatus.ERROR:
                self.errors += 1
