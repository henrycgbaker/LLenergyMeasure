"""Multi-strategy FLOPs estimation with graceful degradation.

This module provides FLOPs estimation using a fallback chain:
1. calflops library - direct measurement (high confidence)
2. Architecture-based - uses model config (medium confidence)
3. Parameter-based - simple 2*P approximation (low confidence)

Key insight: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
computation happens at FP16 after dequantization. We do NOT apply
reduction factors for BNB quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from llenergymeasure.domain.metrics import FlopsResult

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig


class FlopsEstimator:
    """Multi-strategy FLOPs estimation with graceful degradation.

    Strategy order (highest to lowest confidence):
    1. calflops library - direct measurement, works with most HF models
    2. Architecture-based - uses model config (hidden_size, num_layers, etc.)
    3. Parameter-based - simple 2*P approximation

    Note: For BitsAndBytes quantization, FLOPs = FP16 FLOPs because
    computation happens at FP16 after dequantization.
    """

    def __init__(self, timeout_sec: int = 30) -> None:
        """Initialize the FLOPs estimator.

        Args:
            timeout_sec: Timeout for calflops estimation (not currently used,
                         reserved for future async implementation).
        """
        self._timeout_sec = timeout_sec

    def estimate(
        self,
        model: Any,
        input_ids: torch.Tensor,
        config: ExperimentConfig | None = None,
    ) -> FlopsResult:
        """Estimate FLOPs using the fallback chain.

        Args:
            model: The model to measure (typically a HuggingFace model).
            input_ids: Tokenized input tensor.
            config: Optional experiment config for precision detection.

        Returns:
            FlopsResult with the estimate and provenance information.
        """
        seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]

        # Determine actual compute precision (BNB always computes at FP16)
        precision = self._get_compute_precision(config)

        # Strategy 1: calflops library (most accurate)
        result = self._try_calflops(model, seq_len, precision)
        if result is not None:
            return result

        # Strategy 2: Architecture-based calculation
        result = self._try_architecture(model, seq_len, precision)
        if result is not None:
            return result

        # Strategy 3: Parameter-based approximation
        return self._parameter_estimate(model, seq_len, precision)

    def _get_compute_precision(self, config: ExperimentConfig | None) -> str:
        """Determine actual compute precision.

        BitsAndBytes quantization stores weights compressed but computes at FP16.

        Args:
            config: Experiment configuration (may be None).

        Returns:
            Precision string (e.g., "fp16", "fp32").
        """
        if config is None:
            return "fp16"

        # Check for BNB quantization in PyTorch config
        pytorch_cfg = config.pytorch
        if pytorch_cfg and (pytorch_cfg.load_in_4bit or pytorch_cfg.load_in_8bit):
            # BNB always dequantizes to FP16 (or bfloat16 for 4bit compute dtype)
            return pytorch_cfg.bnb_4bit_compute_dtype if pytorch_cfg.load_in_4bit else "fp16"

        # Use the config's fp_precision
        precision = config.fp_precision.lower()
        if precision in ("float32", "fp32"):
            return "fp32"
        if precision in ("bfloat16", "bf16"):
            return "bf16"
        return "fp16"

    def _try_calflops(
        self,
        model: Any,
        seq_len: int,
        precision: str,
    ) -> FlopsResult | None:
        """Try to estimate FLOPs using calflops library.

        Args:
            model: The model to measure.
            seq_len: Sequence length.
            precision: Compute precision string.

        Returns:
            FlopsResult if successful, None otherwise.
        """
        try:
            from calflops import calculate_flops

            # calflops returns (flops, macs, params) or similar
            # The exact return format may vary by version
            result = calculate_flops(
                model=model,
                input_shape=(1, seq_len),
                transformer_tokenizer=None,
                print_results=False,
                output_as_string=False,
            )

            # Handle different return formats
            flops = result[0] if isinstance(result, tuple) else result

            # Convert to float if needed
            if isinstance(flops, str):
                # Parse string like "1.5 GFLOPS"
                flops = self._parse_flops_string(flops)

            if flops is not None and flops > 0:
                logger.debug(f"calflops estimation: {flops:.2e} FLOPs")
                return FlopsResult(
                    value=float(flops),
                    method="calflops",
                    confidence="high",
                    precision=precision,
                )

        except ImportError:
            logger.debug("calflops not installed, trying architecture-based")
        except Exception as e:
            logger.debug(f"calflops failed: {e}, trying architecture-based")

        return None

    def _try_architecture(
        self,
        model: Any,
        seq_len: int,
        precision: str,
    ) -> FlopsResult | None:
        """Try to estimate FLOPs from model architecture config.

        Uses the formula for decoder-only transformers:
        - Attention: 4 * seq * hidden (Q,K,V,O projections) + 2 * seq² * head_dim
        - FFN: 8 * hidden² (assuming 4x intermediate)

        Args:
            model: The model (must have .config attribute).
            seq_len: Sequence length.
            precision: Compute precision string.

        Returns:
            FlopsResult if successful, None otherwise.
        """
        try:
            model_config = model.config

            hidden = model_config.hidden_size
            layers = model_config.num_hidden_layers
            # num_attention_heads accessed to verify config exists
            _ = model_config.num_attention_heads

            # Get intermediate size (FFN), default to 4x hidden
            intermediate = getattr(model_config, "intermediate_size", hidden * 4)

            # Per-layer FLOPs (simplified for decoder-only):
            # Attention: Q, K, V, O projections + attention scores
            # QKV projections: 3 * 2 * hidden * hidden = 6 * hidden²
            # Output projection: 2 * hidden * hidden = 2 * hidden²
            # Attention scores: 2 * seq * seq * head_dim * heads = 2 * seq² * hidden
            attn_proj_flops = 8 * hidden * hidden
            attn_score_flops = 2 * seq_len * seq_len * hidden

            # FFN: up projection + down projection (with intermediate size)
            # Up: 2 * hidden * intermediate
            # Down: 2 * intermediate * hidden
            # If using GLU (e.g., LLaMA): add another up projection
            has_glu = getattr(model_config, "hidden_act", "").lower() in ["silu", "swiglu", "gelu"]
            ffn_multiplier = 3 if has_glu else 2
            ffn_flops = ffn_multiplier * 2 * hidden * intermediate

            per_layer = attn_proj_flops + attn_score_flops + ffn_flops
            total = layers * per_layer * seq_len

            model_type = getattr(model_config, "model_type", "unknown")
            logger.debug(
                f"Architecture estimation for {model_type}: "
                f"{total:.2e} FLOPs ({layers} layers, {hidden} hidden)"
            )

            return FlopsResult(
                value=float(total),
                method="architecture",
                confidence="medium",
                precision=precision,
                notes=f"Based on model config: {model_type}",
            )

        except AttributeError as e:
            logger.debug(f"Architecture estimation failed (missing attr): {e}")
        except Exception as e:
            logger.debug(f"Architecture estimation failed: {e}")

        return None

    def _parameter_estimate(
        self,
        model: Any,
        seq_len: int,
        precision: str,
    ) -> FlopsResult:
        """Estimate FLOPs based on parameter count (2*P approximation).

        This is the fallback when other methods fail. Uses the rule of thumb
        that inference requires ~2 FLOPs per parameter per token.

        Args:
            model: The model.
            seq_len: Sequence length.
            precision: Compute precision string.

        Returns:
            FlopsResult (always succeeds, may return 0 on error).
        """
        try:
            params = sum(p.numel() for p in model.parameters())
            flops = 2 * params * seq_len

            logger.debug(
                f"Parameter-based estimation: {flops:.2e} FLOPs "
                f"(2 * {params:,} params * {seq_len} tokens)"
            )

            return FlopsResult(
                value=float(flops),
                method="parameter_estimate",
                confidence="low",
                precision=precision,
                notes=f"Approximation: 2 * {params:,} params * {seq_len} tokens",
            )

        except Exception as e:
            logger.error(f"All FLOPs estimation methods failed: {e}")
            return FlopsResult(
                value=0.0,
                method="parameter_estimate",
                confidence="low",
                precision=precision,
                notes="Could not estimate FLOPs - all methods failed",
            )

    def _parse_flops_string(self, flops_str: str) -> float | None:
        """Parse a FLOPs string like '1.5 GFLOPS' to a float.

        Args:
            flops_str: String representation of FLOPs.

        Returns:
            FLOPs as float, or None if parsing fails.
        """
        try:
            # Remove common suffixes and parse
            multipliers = {
                "T": 1e12,
                "G": 1e9,
                "M": 1e6,
                "K": 1e3,
                "TFLOPS": 1e12,
                "GFLOPS": 1e9,
                "MFLOPS": 1e6,
                "KFLOPS": 1e3,
            }

            flops_str = flops_str.strip().upper()
            for suffix, mult in multipliers.items():
                if suffix in flops_str:
                    num_str = flops_str.replace(suffix, "").replace("FLOPS", "").strip()
                    return float(num_str) * mult

            # Try direct parse
            return float(flops_str)
        except (ValueError, AttributeError):
            return None


# Module-level convenience instance
_default_estimator: FlopsEstimator | None = None


def get_flops_estimator() -> FlopsEstimator:
    """Get or create the default FlopsEstimator instance."""
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = FlopsEstimator()
    return _default_estimator


def estimate_flops(
    model: Any,
    input_ids: torch.Tensor,
    config: ExperimentConfig | None = None,
) -> FlopsResult:
    """Convenience function to estimate FLOPs.

    Args:
        model: The model to measure.
        input_ids: Tokenized input tensor.
        config: Optional experiment config.

    Returns:
        FlopsResult with the estimate and provenance.
    """
    return get_flops_estimator().estimate(model, input_ids, config)
