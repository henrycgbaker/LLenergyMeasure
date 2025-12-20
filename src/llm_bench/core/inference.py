"""Inference engine for LLM benchmarking."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from llm_bench.config.models import ExperimentConfig
from llm_bench.core.prompts import (
    create_adaptive_batches,
    create_fixed_batches,
    tokenize_batch,
)
from llm_bench.domain.metrics import InferenceMetrics

if TYPE_CHECKING:
    from accelerate import Accelerator
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class InferenceResult:
    """Results from an inference run."""

    metrics: InferenceMetrics
    input_ids: torch.Tensor
    outputs: list[torch.Tensor] | None = None


@dataclass
class BatchResult:
    """Results from a single batch inference."""

    output_ids: torch.Tensor
    input_tokens: int
    generated_tokens: int
    latency_ms: float


def calculate_inference_metrics(
    num_prompts: int,
    latencies_ms: list[float],
    total_input_tokens: int,
    total_generated_tokens: int,
) -> InferenceMetrics:
    """Calculate inference performance metrics.

    Args:
        num_prompts: Number of prompts processed.
        latencies_ms: List of per-batch latencies in milliseconds.
        total_input_tokens: Total input tokens processed.
        total_generated_tokens: Total tokens generated.

    Returns:
        InferenceMetrics with calculated values.
    """
    total_time_sec = sum(latencies_ms) / 1000.0 if latencies_ms else 0.0
    tokens_per_sec = total_generated_tokens / total_time_sec if total_time_sec > 0 else 0.0
    latency_per_token_ms = 1000.0 / tokens_per_sec if tokens_per_sec > 0 else 0.0

    return InferenceMetrics(
        total_tokens=total_input_tokens + total_generated_tokens,
        input_tokens=total_input_tokens,
        output_tokens=total_generated_tokens,
        inference_time_sec=total_time_sec,
        tokens_per_second=tokens_per_sec,
        latency_per_token_ms=latency_per_token_ms,
    )


def run_inference(
    model: PreTrainedModel,
    config: ExperimentConfig,
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    accelerator: Accelerator,
) -> InferenceResult:
    """Run inference on prompts and collect metrics.

    Args:
        model: The language model.
        config: Experiment configuration.
        prompts: List of prompt strings.
        tokenizer: The tokenizer.
        accelerator: Accelerator for distributed setup.

    Returns:
        InferenceResult with metrics and optionally outputs.
    """
    device = accelerator.device
    max_input_tokens = config.max_input_tokens
    max_output_tokens = config.max_output_tokens

    # Create batches
    batches = _create_batches(prompts, tokenizer, config)

    # Process batches
    all_outputs: list[torch.Tensor] = []
    all_input_ids: list[torch.Tensor] = []
    latencies: list[float] = []
    total_generated_tokens = 0
    total_input_tokens = 0

    for batch_idx, batch in enumerate(batches):
        result = _process_batch(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
            device=device,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            config=config,
            batch_idx=batch_idx,
            total_batches=len(batches),
            accelerator=accelerator,
        )

        all_input_ids.append(result.output_ids[:, :max_input_tokens])
        latencies.append(result.latency_ms)
        total_input_tokens += result.input_tokens
        total_generated_tokens += result.generated_tokens

        if config.save_outputs:
            all_outputs.append(result.output_ids)

    concatenated_input_ids = torch.cat(all_input_ids, dim=0)

    metrics = calculate_inference_metrics(
        num_prompts=len(prompts),
        latencies_ms=latencies,
        total_input_tokens=total_input_tokens,
        total_generated_tokens=total_generated_tokens,
    )

    return InferenceResult(
        metrics=metrics,
        input_ids=concatenated_input_ids,
        outputs=all_outputs if config.save_outputs else None,
    )


def _create_batches(
    prompts: list[str],
    tokenizer: PreTrainedTokenizer,
    config: ExperimentConfig,
) -> list[list[str]]:
    """Create batches based on config settings."""
    batching = config.batching_options

    if batching.dynamic_batching:
        # Dynamic batching uses token budget
        return create_adaptive_batches(
            prompts=prompts,
            tokenizer=tokenizer,
            max_tokens_per_batch=config.max_input_tokens,
            max_prompt_tokens=config.max_input_tokens,
            max_batch_size=batching.batch_size,
        )
    else:
        return create_fixed_batches(
            prompts=prompts,
            batch_size=batching.batch_size,
        )


def _process_batch(
    model: PreTrainedModel,
    batch: list[str],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_input_tokens: int,
    max_output_tokens: int,
    config: ExperimentConfig,
    batch_idx: int,
    total_batches: int,
    accelerator: Accelerator,
) -> BatchResult:
    """Process a single batch of prompts."""
    # Tokenize batch
    tokenized = tokenize_batch(
        prompts=batch,
        tokenizer=tokenizer,
        max_length=max_input_tokens,
        batch_size=len(batch),
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    input_tokens = input_ids.numel()

    gpu_id = device.index if hasattr(device, "index") else 0
    logger.info(f"[GPU {gpu_id}] Tokenized batch {batch_idx + 1}/{total_batches}")

    # Calculate allowed output tokens
    total_allowed = tokenizer.model_max_length
    current_length = input_ids.shape[1]
    allowed_new = max(0, total_allowed - current_length)

    # Build generation kwargs
    generation_kwargs = _build_generation_kwargs(
        config=config,
        input_length=current_length,
        max_output_tokens=max_output_tokens,
        allowed_new_tokens=allowed_new,
    )

    # Apply latency simulation if configured
    _apply_latency_simulation(config, batch_idx)

    # Run inference
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

    torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    logger.info(
        f"[GPU {gpu_id}] Completed batch {batch_idx + 1}/{total_batches} " f"in {latency_ms:.1f}ms"
    )

    # Count generated tokens
    generated_tokens = 0
    for j in range(input_ids.size(0)):
        prompt_len = input_ids[j].shape[0]
        gen_len = outputs[j].shape[0] - prompt_len
        generated_tokens += gen_len

    return BatchResult(
        output_ids=outputs,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        latency_ms=latency_ms,
    )


def _build_generation_kwargs(
    config: ExperimentConfig,
    input_length: int,
    max_output_tokens: int,
    allowed_new_tokens: int,
) -> dict[str, Any]:
    """Build kwargs dict for model.generate()."""
    min_output = config.min_output_tokens or 0
    min_length = input_length + min_output

    generation_kwargs: dict[str, Any] = {
        "min_length": min_length,
        "max_new_tokens": min(max_output_tokens, allowed_new_tokens),
    }

    # Apply decoder settings
    decoder = config.decoder_config
    temp = decoder.temperature

    if temp and temp > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temp
        if decoder.top_k:
            generation_kwargs["top_k"] = decoder.top_k
        if decoder.top_p:
            generation_kwargs["top_p"] = decoder.top_p

    return generation_kwargs


def _apply_latency_simulation(config: ExperimentConfig, batch_idx: int) -> None:
    """Apply latency simulation delays if configured."""
    latency = config.latency_simulation

    if latency.enabled:
        delay_sec = random.uniform(latency.delay_min_ms, latency.delay_max_ms) / 1000.0
        time.sleep(delay_sec)
