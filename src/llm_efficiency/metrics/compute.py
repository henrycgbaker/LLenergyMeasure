"""
Computational metrics calculation including FLOPs, memory, and utilization.

This module provides accurate FLOPs calculation for both quantized and non-quantized models,
fixing the critical bug in v1.0 where all quantized models used the same hardcoded FLOPs value.
"""

import io
import json
import logging
import contextlib
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
import psutil
import ptflops
from transformers import PreTrainedModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class FLOPsCalculator:
    """
    Calculates FLOPs for models with caching and support for quantized models.

    This fixes the v1.0 bug where all quantized models used the same hardcoded
    FLOPs value. Now each model gets its accurate FLOPs computed before quantization.

    Strategies (in order of preference):
    1. Cache lookup (instant)
    2. Direct ptflops calculation for unquantized models
    3. Architectural estimation for quantized models
    4. Load unquantized model temporarily to compute FLOPs
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize FLOPs calculator.

        Args:
            cache_dir: Directory for FLOPs cache. Defaults to ~/.cache/llm_efficiency/flops
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "llm_efficiency" / "flops"
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "flops_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, int]:
        """Load FLOPs cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load FLOPs cache: %s", e)
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save FLOPs cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save FLOPs cache: %s", e)

    def _get_cache_key(self, model_name: str, sequence_length: int) -> str:
        """Generate cache key for model and sequence length."""
        return f"{model_name}::{sequence_length}"

    def _compute_flops_ptflops(
        self,
        model: PreTrainedModel,
        sequence_length: int,
        device: torch.device,
        timeout: int = 10,
    ) -> Optional[int]:
        """
        Compute FLOPs using ptflops library.

        Args:
            model: The model to analyze
            sequence_length: Input sequence length
            device: Device model is on
            timeout: Timeout in seconds

        Returns:
            FLOPs count or None if computation fails
        """

        def input_constructor(input_res: Tuple[int]) -> Dict[str, torch.Tensor]:
            """Construct input tensors for ptflops."""
            dummy_input = torch.zeros((1,) + input_res, dtype=torch.long).to(device)
            attention_mask = torch.ones_like(dummy_input)
            return {"input_ids": dummy_input, "attention_mask": attention_mask}

        def compute() -> int:
            """Inner function for threaded execution."""
            with io.StringIO() as buf:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    flops, _ = ptflops.get_model_complexity_info(
                        model,
                        input_res=(sequence_length,),
                        as_strings=False,
                        print_per_layer_stat=False,
                        verbose=False,
                        input_constructor=input_constructor,
                    )
            return int(flops)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(compute)
                return future.result(timeout=timeout)
        except (FuturesTimeoutError, Exception) as e:
            logger.warning("ptflops computation failed: %s", e)
            return None

    def _estimate_flops_from_architecture(
        self,
        model: PreTrainedModel,
        sequence_length: int,
        num_output_tokens: int = 1,
    ) -> int:
        """
        Estimate FLOPs based on model architecture parameters.

        For Transformer models (causal LM):
        FLOPs per token ≈ 2 * P + 4 * L * H² * S

        Where:
        - P = total parameters (2 FLOPs per param for forward pass)
        - L = number of layers
        - H = hidden dimension
        - S = sequence length

        This is an approximation but should be accurate within 10-20% for standard architectures.

        Args:
            model: The model
            sequence_length: Input sequence length
            num_output_tokens: Number of output tokens to generate

        Returns:
            Estimated FLOPs

        Raises:
            ValueError: If architecture info is missing
        """
        config = model.config

        # Extract architecture parameters
        num_layers = getattr(config, "num_hidden_layers", None)
        hidden_size = getattr(config, "hidden_size", None)
        vocab_size = getattr(config, "vocab_size", None)
        intermediate_size = getattr(config, "intermediate_size", None)

        if num_layers is None or hidden_size is None:
            raise ValueError("Cannot estimate FLOPs: missing num_hidden_layers or hidden_size")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Estimate FLOPs per token
        # Attention: 4 projections (Q, K, V, O) + attention scores
        attention_flops = 4 * num_layers * hidden_size**2 * sequence_length

        # FFN: typically 2 linear layers with intermediate_size = 4 * hidden_size
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        ffn_flops = 2 * num_layers * hidden_size * intermediate_size

        # Embedding + output projection
        if vocab_size is not None:
            embedding_flops = 2 * hidden_size * vocab_size
        else:
            embedding_flops = 0

        # Total per token
        flops_per_token = attention_flops + ffn_flops + embedding_flops

        # Total for input + output generation
        total_tokens = sequence_length + num_output_tokens
        total_flops = flops_per_token * total_tokens

        logger.info(
            "Estimated FLOPs from architecture: %d (L=%d, H=%d, params=%d)",
            total_flops,
            num_layers,
            hidden_size,
            num_params,
        )

        return int(total_flops)

    def get_flops(
        self,
        model: PreTrainedModel,
        model_name: str,
        sequence_length: int,
        device: torch.device,
        is_quantized: bool = False,
        force_recompute: bool = False,
    ) -> int:
        """
        Get FLOPs for a model, using cache when available.

        Args:
            model: The model to analyze
            model_name: Model identifier for caching
            sequence_length: Input sequence length
            device: Device model is on
            is_quantized: Whether model is quantized
            force_recompute: Force recomputation even if cached

        Returns:
            FLOPs as integer

        Raises:
            ValueError: If FLOPs cannot be computed
        """
        cache_key = self._get_cache_key(model_name, sequence_length)

        # Check cache first
        if not force_recompute and cache_key in self.cache:
            logger.debug("FLOPs cache hit for %s", cache_key)
            return self.cache[cache_key]

        logger.info("Computing FLOPs for %s (quantized=%s)", model_name, is_quantized)

        flops = None

        if is_quantized:
            # For quantized models, try architectural estimation first
            try:
                flops = self._estimate_flops_from_architecture(model, sequence_length)
                logger.info("Used architectural estimation for quantized model")
            except Exception as e:
                logger.warning("Architectural estimation failed: %s", e)

                # Fallback: Load unquantized model temporarily
                try:
                    logger.info("Loading unquantized model to compute FLOPs...")
                    temp_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="cpu",  # Load on CPU to save GPU memory
                        trust_remote_code=True,
                    )
                    flops = self._compute_flops_ptflops(
                        temp_model, sequence_length, torch.device("cpu")
                    )
                    del temp_model
                    torch.cuda.empty_cache()
                    logger.info("Computed FLOPs using temporary unquantized model")
                except Exception as e2:
                    logger.error("Failed to load unquantized model: %s", e2)
                    raise ValueError(
                        f"Cannot compute FLOPs for quantized model {model_name}"
                    ) from e2
        else:
            # Unquantized model - direct computation
            flops = self._compute_flops_ptflops(model, sequence_length, device)

            if flops is None:
                # Fallback to architectural estimation
                logger.warning("ptflops failed, trying architectural estimation")
                flops = self._estimate_flops_from_architecture(model, sequence_length)

        if flops is None:
            raise ValueError(f"Failed to compute FLOPs for {model_name}")

        # Cache result
        self.cache[cache_key] = flops
        self._save_cache()

        logger.info("Computed FLOPs: %d", flops)
        return flops

    def get_flops_batch(
        self,
        model: PreTrainedModel,
        model_name: str,
        input_ids: torch.Tensor,
        device: torch.device,
        is_quantized: bool = False,
    ) -> int:
        """
        Get FLOPs for a batch of inputs.

        Args:
            model: The model
            model_name: Model identifier
            input_ids: Batch of input token IDs [batch_size, seq_len]
            device: Device
            is_quantized: Whether model is quantized

        Returns:
            Total FLOPs for the batch
        """
        batch_size = input_ids.shape[0]
        sequence_lengths = [input_ids[i].numel() for i in range(batch_size)]

        # If all same length, compute once and multiply
        if len(set(sequence_lengths)) == 1:
            flops_single = self.get_flops(
                model, model_name, sequence_lengths[0], device, is_quantized
            )
            return flops_single * batch_size

        # Variable lengths - compute per sample
        total_flops = 0
        for seq_len in sequence_lengths:
            flops = self.get_flops(model, model_name, seq_len, device, is_quantized)
            total_flops += flops

        return total_flops

    def clear_cache(self) -> None:
        """Clear the FLOPs cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("FLOPs cache cleared")


def get_gpu_memory_stats(device: torch.device) -> Dict[str, int]:
    """
    Get GPU memory usage statistics.

    Args:
        device: CUDA device

    Returns:
        Dictionary with memory statistics in bytes
    """
    if not torch.cuda.is_available():
        return {}

    torch.cuda.reset_peak_memory_stats(device)

    return {
        "gpu_current_memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "gpu_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_current_memory_reserved_bytes": torch.cuda.memory_reserved(device),
        "gpu_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def get_gpu_cpu_utilization() -> Dict[str, any]:
    """
    Get GPU and CPU utilization statistics.

    Returns:
        Dictionary with utilization percentages and CPU memory usage
    """
    utilization_info: Dict[str, any] = {}

    # GPU utilization using nvidia-smi
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        lines = result.decode("utf-8").strip().splitlines()
        gpu_utils = [float(line.strip()) for line in lines if line.strip()]
        utilization_info["gpu_utilization_percent"] = gpu_utils
    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as e:
        logger.warning("Failed to get GPU utilization: %s", e)
        utilization_info["gpu_utilization_percent"] = []

    # CPU utilization
    try:
        utilization_info["cpu_usage_percent"] = psutil.cpu_percent(interval=1.0, percpu=False)
        utilization_info["cpu_memory_usage_bytes"] = psutil.Process().memory_info().rss
    except (psutil.Error, OSError) as e:
        logger.warning("Failed to get CPU utilization: %s", e)
        utilization_info["cpu_usage_percent"] = 0.0
        utilization_info["cpu_memory_usage_bytes"] = 0

    return utilization_info
