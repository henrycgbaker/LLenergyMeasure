"""
Inference engine for LLM text generation with comprehensive metrics tracking.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator

from llm_efficiency.config import ExperimentConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Manages LLM inference with batching, metrics tracking, and distributed support.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ExperimentConfig,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        Initialize inference engine.

        Args:
            model: Loaded model
            tokenizer: Tokenizer
            config: Experiment configuration
            accelerator: Optional Accelerator for distributed inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator

        self.device = next(model.parameters()).device
        logger.info(f"Inference engine initialized on device: {self.device}")

    def _prepare_generation_config(self) -> Dict:
        """
        Prepare generation configuration from experiment config.

        Returns:
            Dictionary with generation parameters
        """
        gen_config = {
            "max_new_tokens": self.config.max_output_tokens,
            "min_new_tokens": self.config.min_output_tokens,
            "do_sample": self.config.decoder.do_sample,
            "temperature": self.config.decoder.temperature,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Add mode-specific parameters
        if self.config.decoder.mode == "top_k":
            gen_config["top_k"] = self.config.decoder.top_k
        elif self.config.decoder.mode == "top_p":
            gen_config["top_p"] = self.config.decoder.top_p

        logger.debug(f"Generation config: {gen_config}")
        return gen_config

    def run_inference(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Run inference on prompts with batching.

        Args:
            prompts: List of input prompts
            batch_size: Batch size (uses config if None)

        Returns:
            Tuple of (generated_texts, metrics)
        """
        if batch_size is None:
            batch_size = self.config.batching.batch_size

        logger.info(f"Running inference on {len(prompts)} prompts (batch_size={batch_size})")

        gen_config = self._prepare_generation_config()

        all_outputs = []
        total_time = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_tokens,
            ).to(self.device)

            input_length = inputs["input_ids"].shape[1]
            total_input_tokens += inputs["input_ids"].numel()

            # Generate
            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config,
                )

            end_time = time.perf_counter()
            batch_time = end_time - start_time
            total_time += batch_time

            # Decode
            if self.config.decode_token_to_text:
                batch_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(batch_texts)
            else:
                all_outputs.extend(outputs.tolist())

            # Count output tokens
            output_length = outputs.shape[1] - input_length
            total_output_tokens += output_length * len(batch_prompts)

            logger.debug(
                f"Batch {i // batch_size + 1}: {batch_time:.3f}s, "
                f"{output_length * len(batch_prompts)} output tokens"
            )

        # Calculate metrics
        total_tokens = total_input_tokens + total_output_tokens
        metrics = {
            "total_time_seconds": total_time,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "num_prompts": len(prompts),
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "queries_per_second": len(prompts) / total_time if total_time > 0 else 0,
            "avg_latency_per_query": total_time / len(prompts) if len(prompts) > 0 else 0,
            "avg_tokens_per_prompt": total_tokens / len(prompts) if len(prompts) > 0 else 0,
        }

        logger.info(f"Inference complete: {metrics['tokens_per_second']:.2f} tokens/s")

        return all_outputs, metrics

    def warmup(self, num_iterations: int = 3) -> None:
        """
        Warmup model with dummy inputs.

        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Running {num_iterations} warmup iterations...")

        dummy_prompt = "This is a warmup prompt to initialize the model."

        for i in range(num_iterations):
            logger.debug(f"Warmup iteration {i + 1}/{num_iterations}")

            inputs = self.tokenizer(
                dummy_prompt,
                return_tensors="pt",
                max_length=32,
            ).to(self.device)

            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                )

            # Clear cache between iterations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Warmup complete")


def run_inference_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    config: ExperimentConfig,
    accelerator: Optional[Accelerator] = None,
    warmup: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Convenience function to run complete inference experiment.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        prompts: Input prompts
        config: Experiment configuration
        accelerator: Optional Accelerator
        warmup: Whether to run warmup

    Returns:
        Tuple of (outputs, metrics)

    Example:
        >>> outputs, metrics = run_inference_experiment(
        ...     model, tokenizer, prompts, config
        ... )
        >>> print(f"Throughput: {metrics['tokens_per_second']:.2f} tokens/s")
    """
    engine = InferenceEngine(model, tokenizer, config, accelerator)

    if warmup:
        engine.warmup()

    return engine.run_inference(prompts)
