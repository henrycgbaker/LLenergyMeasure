"""Inference engine for LLM benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.core.prompts import (
    create_adaptive_batches,
    create_fixed_batches,
    tokenize_batch,
)
from llm_energy_measure.domain.metrics import InferenceMetrics

if TYPE_CHECKING:
    from accelerate import Accelerator
    from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_energy_measure.core.traffic import TrafficGenerator


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

    Supports different parallelism strategies:
    - none: Standard inference with device_map='auto'
    - tensor_parallel: HF native TP - works transparently with model.generate()
    - pipeline_parallel: Experimental - uses standard inference path

    Args:
        model: The language model (may be wrapped for parallelism).
        config: Experiment configuration.
        prompts: List of prompt strings.
        tokenizer: The tokenizer.
        accelerator: Accelerator for distributed setup.

    Returns:
        InferenceResult with metrics and optionally outputs.
    """
    sharding_strategy = config.sharding_config.strategy

    # Log parallelism mode
    if sharding_strategy == "tensor_parallel":
        logger.info("Running inference with tensor parallelism (HF native tp_plan)")
    elif sharding_strategy == "pipeline_parallel":
        logger.info(
            "Running inference with pipeline parallelism (experimental - "
            "full PP scheduling not yet implemented)"
        )

    device = accelerator.device
    max_input_tokens = config.max_input_tokens
    max_output_tokens = config.max_output_tokens

    # Create batches
    batches = _create_batches(prompts, tokenizer, config)

    # Initialize traffic generator once for the entire experiment
    # This ensures proper Poisson sequence with seeded reproducibility
    traffic_generator = _create_traffic_generator(config)

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
            traffic_generator=traffic_generator,
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
    """Create batches based on config settings.

    Industry-standard strategies (per MLPerf/vLLM terminology):
    - static: Fixed batch size, pads to max_length (MLPerf offline scenario)
    - dynamic: Token-aware batching with max_tokens_per_batch (MLPerf server scenario)
    - sorted_static: Sort by length, then static batches (reduces padding waste)
    - sorted_dynamic: Sort by length + dynamic token budget (optimal packing)
    """
    batching = config.batching_options
    strategy = batching.strategy

    # Determine max tokens per batch
    max_tokens = batching.max_tokens_per_batch or config.max_input_tokens

    # Apply length sorting for relevant strategies
    working_prompts = prompts
    if strategy in ("sorted_static", "sorted_dynamic"):
        working_prompts = sorted(prompts, key=len)
        logger.debug(f"Sorted {len(prompts)} prompts by length for {strategy} strategy")

    # Create batches based on strategy
    if strategy == "static":
        return create_fixed_batches(
            prompts=working_prompts,
            batch_size=batching.batch_size,
        )
    elif strategy == "dynamic":
        return create_adaptive_batches(
            prompts=working_prompts,
            tokenizer=tokenizer,
            max_tokens_per_batch=max_tokens,
            max_prompt_tokens=config.max_input_tokens,
            max_batch_size=batching.batch_size,
        )
    elif strategy == "sorted_static":
        # Sort by length, then static batches
        return create_fixed_batches(
            prompts=working_prompts,
            batch_size=batching.batch_size,
        )
    elif strategy == "sorted_dynamic":
        # Sort by length + dynamic token budget
        return create_adaptive_batches(
            prompts=working_prompts,
            tokenizer=tokenizer,
            max_tokens_per_batch=max_tokens,
            max_prompt_tokens=config.max_input_tokens,
            max_batch_size=batching.batch_size,
        )
    else:
        # Fallback to static
        logger.warning(f"Unknown batching strategy '{strategy}', using static")
        return create_fixed_batches(
            prompts=working_prompts,
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
    traffic_generator: TrafficGenerator | None = None,
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

    # Apply traffic simulation if configured (MLPerf-style Poisson arrivals)
    _apply_traffic_simulation(config, batch_idx, traffic_generator)

    # Set seed for reproducible sampling (seed + batch_idx for varied but reproducible)
    if config.random_seed is not None:
        batch_seed = config.random_seed + batch_idx
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)

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
    """Build kwargs dict for model.generate().

    Applies all decoder configuration settings correctly:
    - Temperature=0 forces greedy decoding regardless of other settings
    - Respects do_sample config value
    - Applies repetition_penalty, min_p, no_repeat_ngram_size when enabled
    """
    min_output = config.min_output_tokens or 0
    decoder = config.decoder_config

    generation_kwargs: dict[str, Any] = {
        "min_length": input_length + min_output,
        "max_new_tokens": min(max_output_tokens, allowed_new_tokens),
    }

    # Temperature=0 forces greedy regardless of other settings
    if decoder.temperature == 0.0:
        generation_kwargs["do_sample"] = False
        return generation_kwargs

    # Apply sampling settings
    generation_kwargs["do_sample"] = decoder.do_sample
    generation_kwargs["temperature"] = decoder.temperature

    if decoder.do_sample:
        # Nucleus/top sampling (only when sampling is enabled)
        if decoder.top_k > 0:
            generation_kwargs["top_k"] = decoder.top_k
        if decoder.top_p < 1.0:
            generation_kwargs["top_p"] = decoder.top_p
        if decoder.min_p > 0.0:
            generation_kwargs["min_p"] = decoder.min_p

        # Repetition control
        if decoder.repetition_penalty != 1.0:
            generation_kwargs["repetition_penalty"] = decoder.repetition_penalty
        if decoder.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = decoder.no_repeat_ngram_size

    return generation_kwargs


def _create_traffic_generator(config: ExperimentConfig) -> TrafficGenerator | None:
    """Create a traffic generator for the experiment.

    Creates the generator once so it maintains state across batches,
    ensuring proper Poisson sequence generation.

    Args:
        config: Experiment configuration.

    Returns:
        TrafficGenerator if traffic simulation is enabled, None otherwise.
    """
    if not config.latency_simulation.enabled:
        return None
    return TrafficGenerator(config.latency_simulation)


def _apply_traffic_simulation(
    config: ExperimentConfig,
    batch_idx: int,
    generator: TrafficGenerator | None,
) -> float:
    """Apply MLPerf-style traffic simulation delays.

    Uses Poisson or constant arrival patterns instead of simple random delays.

    Args:
        config: Experiment configuration.
        batch_idx: Current batch index (0-indexed).
        generator: Pre-initialized traffic generator.

    Returns:
        The delay applied in seconds.
    """
    from llm_energy_measure.core.traffic import apply_traffic_delay

    return apply_traffic_delay(
        config=config.latency_simulation,
        batch_idx=batch_idx,
        generator=generator,
    )
