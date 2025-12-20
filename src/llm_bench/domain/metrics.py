"""Metrics domain models for LLM Bench."""

from typing import Literal

from pydantic import BaseModel, Field


class FlopsResult(BaseModel):
    """FLOPs estimation result with provenance tracking.

    Tracks both the estimated value and the method used to obtain it,
    allowing downstream consumers to understand confidence levels.

    Note: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
    computation happens at FP16 after dequantization.
    """

    value: float = Field(..., description="Estimated FLOPs count")
    method: Literal["calflops", "architecture", "parameter_estimate"] = Field(
        ..., description="Estimation method used"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence level of the estimate"
    )
    precision: str = Field(..., description="Compute precision (e.g., fp16, fp32)")
    notes: str | None = Field(default=None, description="Additional context or warnings")

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid (non-zero) estimate."""
        return self.value > 0


class InferenceMetrics(BaseModel):
    """Metrics from model inference."""

    total_tokens: int = Field(..., description="Total tokens generated")
    input_tokens: int = Field(..., description="Number of input/prompt tokens")
    output_tokens: int = Field(..., description="Number of output/generated tokens")
    inference_time_sec: float = Field(..., description="Total inference time in seconds")
    tokens_per_second: float = Field(..., description="Throughput in tokens/second")
    latency_per_token_ms: float = Field(..., description="Average latency per token in ms")

    @property
    def throughput(self) -> float:
        """Alias for tokens_per_second."""
        return self.tokens_per_second


class EnergyMetrics(BaseModel):
    """Energy consumption metrics."""

    total_energy_j: float = Field(..., description="Total energy consumed in Joules")
    gpu_energy_j: float = Field(0.0, description="GPU energy in Joules")
    cpu_energy_j: float = Field(0.0, description="CPU energy in Joules")
    ram_energy_j: float = Field(0.0, description="RAM energy in Joules")
    gpu_power_w: float = Field(0.0, description="Average GPU power in Watts")
    cpu_power_w: float = Field(0.0, description="Average CPU power in Watts")
    duration_sec: float = Field(..., description="Measurement duration in seconds")
    emissions_kg_co2: float = Field(0.0, description="Carbon emissions in kg CO2")
    energy_per_token_j: float = Field(0.0, description="Energy per token in Joules")

    @property
    def total_power_w(self) -> float:
        """Total average power consumption."""
        return self.gpu_power_w + self.cpu_power_w


class ComputeMetrics(BaseModel):
    """Computational metrics (FLOPs, memory)."""

    flops_total: float = Field(..., description="Total FLOPs for the inference")
    flops_per_token: float = Field(0.0, description="FLOPs per token")
    flops_per_second: float = Field(0.0, description="FLOPs throughput")
    peak_memory_mb: float = Field(0.0, description="Peak GPU memory usage in MB")
    model_memory_mb: float = Field(0.0, description="Model memory footprint in MB")

    flops_method: str = Field(
        "unknown", description="Method used to estimate FLOPs (calflops, architecture, parameter)"
    )
    flops_confidence: str = Field("unknown", description="Confidence level (high, medium, low)")
    compute_precision: str = Field("fp16", description="Compute precision used")


class CombinedMetrics(BaseModel):
    """All metrics combined for a single measurement."""

    inference: InferenceMetrics
    energy: EnergyMetrics
    compute: ComputeMetrics

    @property
    def efficiency_tokens_per_joule(self) -> float:
        """Tokens generated per Joule of energy."""
        if self.energy.total_energy_j > 0:
            return self.inference.total_tokens / self.energy.total_energy_j
        return 0.0

    @property
    def efficiency_flops_per_watt(self) -> float:
        """FLOPs per Watt (computational efficiency)."""
        if self.energy.total_power_w > 0:
            return self.compute.flops_per_second / self.energy.total_power_w
        return 0.0
