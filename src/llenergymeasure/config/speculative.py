"""Unified speculative decoding configuration for cross-backend compatibility.

Speculative decoding uses a small draft model to propose multiple tokens,
then the main model verifies them in a single forward pass. This can provide
2-3x latency improvement for compatible model pairs.

Backend Support:
    | Feature      | PyTorch | vLLM | TensorRT |
    |--------------|---------|------|----------|
    | Draft model  | Yes     | Yes  | Yes      |
    | N-gram       | No      | Yes  | Planned  |
    | Medusa       | No      | Yes  | Yes      |
    | EAGLE        | No      | Yes  | Planned  |
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class UnifiedSpeculativeConfig(BaseModel):
    """Unified speculative decoding configuration.

    Works across all backends that support speculative decoding.
    Backends map this to their native speculative decoding APIs.

    Methods:
        draft_model: Use a separate smaller model as the draft
        ngram: Use n-gram matching from the prompt (vLLM/TensorRT)
        medusa: Use Medusa heads for speculation (vLLM/TensorRT)
        eagle: Use EAGLE architecture (vLLM)

    Examples:
        # Disable speculative decoding (default)
        speculative:
          enabled: false

        # Use a draft model
        speculative:
          enabled: true
          method: draft_model
          draft_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
          num_speculative_tokens: 5

        # Use n-gram speculation (vLLM only)
        speculative:
          enabled: true
          method: ngram
          num_speculative_tokens: 5
    """

    enabled: bool = Field(
        default=False,
        description="Enable speculative decoding",
    )
    method: Literal["draft_model", "ngram", "medusa", "eagle"] = Field(
        default="draft_model",
        description="Speculation method",
    )
    draft_model: str | None = Field(
        default=None,
        description="Draft model name/path for speculative decoding",
    )
    num_speculative_tokens: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of tokens to speculate per step (1-20)",
    )

    # Draft model configuration
    draft_tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Tensor parallel size for draft model",
    )

    # N-gram specific options
    ngram_min: int = Field(
        default=1,
        ge=1,
        description="Minimum n-gram window for prompt lookup",
    )
    ngram_max: int | None = Field(
        default=None,
        ge=1,
        description="Maximum n-gram window for prompt lookup",
    )

    @model_validator(mode="after")
    def validate_draft_model_requirement(self) -> UnifiedSpeculativeConfig:
        """Validate draft_model is set when using draft_model method."""
        if self.enabled and self.method == "draft_model" and not self.draft_model:
            raise ValueError(
                "speculative.draft_model is required when method='draft_model'. "
                "Specify a smaller model (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)."
            )
        return self

    @property
    def uses_draft_model(self) -> bool:
        """True if this configuration uses a separate draft model."""
        return self.enabled and self.method == "draft_model" and self.draft_model is not None
