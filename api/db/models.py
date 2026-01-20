"""SQLAlchemy ORM models for LLM Bench API."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class User(Base):
    """User model for GitHub OAuth authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    github_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Benchmark(Base):
    """Benchmark result model.

    Stores LLM efficiency benchmark results with denormalised key metrics
    for fast leaderboard queries, plus full result blob for detail views.
    """

    __tablename__ = "benchmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    # User who uploaded (nullable for anonymous uploads)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    # Model identification
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_family: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Hardware/backend info (indexed for filtering)
    backend: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    hardware: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    gpu_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Key metrics (denormalised for fast leaderboard queries)
    tokens_per_joule: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    throughput_tokens_per_sec: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    total_energy_joules: Mapped[float] = mapped_column(Float, nullable=False)
    avg_energy_per_token_joules: Mapped[float] = mapped_column(Float, nullable=False)
    peak_memory_mb: Mapped[float] = mapped_column(Float, nullable=False)

    # Streaming latency metrics (optional)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Total tokens for context
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Full result blob (complete AggregatedResult JSON)
    raw_result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Configuration used
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Composite indexes for common query patterns
    __table_args__ = (
        Index("ix_benchmarks_leaderboard", "tokens_per_joule", "throughput_tokens_per_sec"),
        Index("ix_benchmarks_filter", "backend", "hardware", "model_family"),
    )
