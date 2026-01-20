"""Benchmark-related Pydantic schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SortField(str, Enum):
    """Valid sort fields for leaderboard."""

    TOKENS_PER_JOULE = "tokens_per_joule"
    THROUGHPUT = "throughput_tokens_per_sec"
    ENERGY = "total_energy_joules"
    MEMORY = "peak_memory_mb"
    CREATED_AT = "created_at"


class SortOrder(str, Enum):
    """Sort order."""

    ASC = "asc"
    DESC = "desc"


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")


class BenchmarkBase(BaseModel):
    """Base benchmark fields."""

    experiment_id: str
    model_name: str
    model_family: str | None = None
    backend: str
    hardware: str
    gpu_name: str | None = None

    # Key metrics
    tokens_per_joule: float
    throughput_tokens_per_sec: float
    total_energy_joules: float
    avg_energy_per_token_joules: float
    peak_memory_mb: float

    # Latency metrics
    ttft_ms: float | None = None
    itl_ms: float | None = None

    # Token counts
    total_tokens: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class BenchmarkResponse(BenchmarkBase):
    """Single benchmark response (full detail)."""

    id: int
    user_id: int | None = None
    raw_result: dict[str, Any] = Field(..., description="Full AggregatedResult JSON")
    config: dict[str, Any] = Field(..., description="Experiment configuration")
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BenchmarkSummary(BenchmarkBase):
    """Benchmark summary for list views (without full raw_result)."""

    id: int
    user_id: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class BenchmarkListResponse(BaseModel):
    """Paginated list of benchmarks."""

    items: list[BenchmarkSummary]
    total: int = Field(..., description="Total number of matching benchmarks")
    page: int
    per_page: int
    pages: int = Field(..., description="Total number of pages")


class BenchmarkCreate(BaseModel):
    """Schema for uploading a new benchmark result.

    Accepts the raw AggregatedResult JSON from the CLI tool.
    """

    raw_result: dict[str, Any] = Field(..., description="AggregatedResult JSON from CLI output")


class CompareResponse(BaseModel):
    """Comparison response for multiple benchmarks."""

    benchmarks: list[BenchmarkResponse]
    comparison: dict[str, Any] = Field(
        ...,
        description="Comparison metrics (best values highlighted)",
    )


class ModelStats(BaseModel):
    """Aggregated stats for a model across all benchmarks."""

    model_name: str
    model_family: str | None = None
    benchmark_count: int
    best_tokens_per_joule: float
    best_throughput: float
    avg_tokens_per_joule: float
    avg_throughput: float


class ModelsListResponse(BaseModel):
    """List of models with aggregated stats."""

    items: list[ModelStats]
    total: int
