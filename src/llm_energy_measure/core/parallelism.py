"""Parallelism strategies for multi-GPU inference.

This module provides parallelism strategies that integrate with the model loading
and inference pipeline. Strategies modify how models are loaded and executed
across multiple GPUs.

Supported strategies:
- NoParallelism: Default device_map="auto" behaviour
- TensorParallelStrategy: HuggingFace native tensor parallelism (tp_plan="auto")
- PipelineParallelStrategy: PyTorch native pipeline parallelism

Usage:
    from llm_energy_measure.core.parallelism import get_parallelism_strategy

    strategy = get_parallelism_strategy(config.sharding_config)
    strategy.setup(config.sharding_config, gpu_list)
    model_kwargs = strategy.prepare_model_kwargs()
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = strategy.wrap_model(model)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from llm_energy_measure.config.models import ShardingConfig


# Models known to support HuggingFace native tensor parallelism
TP_SUPPORTED_MODEL_PATTERNS = [
    "llama",
    "mistral",
    "mixtral",
    "qwen",
    "phi",
    "gemma",
    "falcon",
    "mpt",
    "bloom",
    "opt",
]


class ParallelismStrategy(ABC):
    """Abstract base class for parallelism strategies.

    Parallelism strategies control how models are loaded and distributed
    across multiple GPUs. Each strategy implements:
    - setup(): Initialise distributed resources
    - prepare_model_kwargs(): Return kwargs for from_pretrained()
    - wrap_model(): Post-load model modifications
    - cleanup(): Release distributed resources
    """

    @abstractmethod
    def setup(self, config: ShardingConfig, gpu_list: list[int]) -> None:
        """Initialise the parallelism strategy.

        Args:
            config: Sharding configuration with strategy-specific options.
            gpu_list: List of GPU indices available for this experiment.

        Raises:
            ValueError: If configuration is invalid for available hardware.
        """
        ...

    @abstractmethod
    def prepare_model_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass to from_pretrained().

        Returns:
            Dictionary of kwargs to merge into model loading.
        """
        ...

    @abstractmethod
    def wrap_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Wrap or modify model after loading.

        Args:
            model: The loaded model.

        Returns:
            Possibly modified model.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up distributed resources."""
        ...

    @property
    @abstractmethod
    def requires_torchrun(self) -> bool:
        """Whether this strategy requires torchrun launcher instead of accelerate."""
        ...


class NoParallelism(ParallelismStrategy):
    """Default parallelism: single device or device_map='auto'.

    This strategy uses HuggingFace's automatic device mapping which
    spreads model layers across available GPUs if the model doesn't
    fit on a single device. Not true parallelism - each GPU handles
    different layers sequentially.
    """

    def setup(self, config: ShardingConfig, gpu_list: list[int]) -> None:
        """No-op for default strategy."""
        logger.debug("NoParallelism strategy: using device_map='auto'")

    def prepare_model_kwargs(self) -> dict[str, Any]:
        """Return device_map='auto' for automatic layer placement."""
        return {"device_map": "auto"}

    def wrap_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """No modification needed."""
        return model

    def cleanup(self) -> None:
        """No cleanup needed for default strategy."""
        pass

    @property
    def requires_torchrun(self) -> bool:
        """Default strategy uses accelerate launcher."""
        return False


class TensorParallelStrategy(ParallelismStrategy):
    """HuggingFace native tensor parallelism using tp_plan.

    Tensor parallelism splits individual layers across GPUs, allowing
    each GPU to process a portion of each layer in parallel. This is
    effective for very large models where even a single layer is too
    large for one GPU's memory.

    Requirements:
    - Model must support HF's tp_plan (Llama, Mistral, Qwen, etc.)
    - Must use torchrun launcher (not accelerate launch)
    - PyTorch 2.x with DTensor support

    Note:
        HuggingFace tensor parallelism is applied at load time via
        tp_plan="auto". No post-load wrapping is needed.
    """

    def __init__(self) -> None:
        self._num_shards: int = 1
        self._device_mesh_initialised: bool = False

    def setup(self, config: ShardingConfig, gpu_list: list[int]) -> None:
        """Initialise tensor parallelism with device mesh.

        Args:
            config: Sharding config with num_shards.
            gpu_list: Available GPUs.

        Raises:
            ValueError: If num_shards exceeds available GPUs.
        """
        if config.num_shards > len(gpu_list):
            raise ValueError(
                f"num_shards ({config.num_shards}) exceeds available GPUs ({len(gpu_list)}). "
                f"Tensor parallelism requires 1 GPU per shard."
            )

        self._num_shards = config.num_shards

        # Device mesh is initialised by torchrun/HF transformers automatically
        # We just validate and log here
        logger.info(
            f"TensorParallelStrategy configured: num_shards={self._num_shards}, "
            f"tp_plan={config.tp_plan}"
        )

    def prepare_model_kwargs(self) -> dict[str, Any]:
        """Return kwargs for tensor parallel model loading.

        Returns:
            Dict with tp_plan and recommended dtype for TP.
        """
        return {
            "tp_plan": "auto",
            # TP typically uses bfloat16 for better performance
            # Let dtype be overridden by config if specified
        }

    def wrap_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """No wrapping needed - TP is applied at load time."""
        logger.debug("TensorParallel model loaded via tp_plan - no wrapping needed")
        return model

    def cleanup(self) -> None:
        """Destroy process group if initialised."""
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.debug("TensorParallel process group destroyed")
            except Exception as e:
                logger.warning(f"Error destroying process group: {e}")

    @property
    def requires_torchrun(self) -> bool:
        """Tensor parallelism requires torchrun launcher."""
        return True


class PipelineParallelStrategy(ParallelismStrategy):
    """PyTorch native pipeline parallelism using torch.distributed.pipelining.

    Pipeline parallelism splits the model into sequential stages, with
    each stage running on a different GPU. Microbatching is used to
    improve utilisation by overlapping computation across stages.

    Schedules:
    - gpipe: Simple fill-drain schedule (lower memory, higher latency)
    - 1f1b: Interleaved schedule (steady state memory, better throughput)

    Note:
        Pipeline parallelism requires model splitting which may not work
        with all HuggingFace models. Falls back to layer-based splitting
        if torch.export fails.
    """

    def __init__(self) -> None:
        self._num_stages: int = 1
        self._schedule_type: str = "gpipe"
        self._num_microbatches: int = 4
        self._rank: int = 0
        self._world_size: int = 1

    def setup(self, config: ShardingConfig, gpu_list: list[int]) -> None:
        """Initialise pipeline parallelism.

        Args:
            config: Sharding config with pipeline options.
            gpu_list: Available GPUs.

        Raises:
            ValueError: If num_shards exceeds available GPUs.
        """
        if config.num_shards > len(gpu_list):
            raise ValueError(
                f"num_shards ({config.num_shards}) exceeds available GPUs ({len(gpu_list)}). "
                f"Pipeline parallelism requires 1 GPU per stage."
            )

        self._num_stages = config.num_shards
        self._schedule_type = config.pipeline_schedule
        self._num_microbatches = config.num_microbatches

        # Initialise distributed if not already done
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        logger.info(
            f"PipelineParallelStrategy configured: stages={self._num_stages}, "
            f"schedule={self._schedule_type}, microbatches={self._num_microbatches}, "
            f"rank={self._rank}/{self._world_size}"
        )

    def prepare_model_kwargs(self) -> dict[str, Any]:
        """Return kwargs for pipeline parallel loading.

        For PP, we need to load on CPU or meta device first, then split.
        """
        # Load on CPU first for splitting
        return {"device_map": "cpu", "low_cpu_mem_usage": True}

    def wrap_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Split model into pipeline stages.

        Attempts automatic splitting via torch.distributed.pipelining.
        Falls back to manual layer-based splitting if that fails.

        Args:
            model: Model loaded on CPU.

        Returns:
            This rank's stage of the model, moved to appropriate GPU.
        """
        try:
            return self._split_model_automatic(model)
        except Exception as e:
            logger.warning(f"Automatic pipeline split failed: {e}. Using layer-based fallback.")
            return self._split_model_manual(model)

    def _split_model_automatic(self, model: PreTrainedModel) -> PreTrainedModel:
        """Split model using torch.distributed.pipelining.

        This uses torch.export to trace the model and split it into stages.
        May not work with all HuggingFace models due to dynamic control flow.
        """
        try:
            from torch.distributed.pipelining import pipeline
        except ImportError as e:
            raise ImportError(
                "torch.distributed.pipelining not available. "
                "Requires PyTorch 2.2+ with pipelining support."
            ) from e

        # Create split points at regular intervals based on model structure
        split_spec = self._create_split_spec(model)

        # Create example input for tracing
        example_input = torch.zeros(1, 128, dtype=torch.long)

        # Build pipeline
        pipe = pipeline(
            module=model,
            mb_args=(example_input,),
            split_spec=split_spec,
        )

        # Get this rank's stage (pipe is dynamically typed from torch.distributed.pipelining)
        stage = pipe.get_stage(self._rank)  # type: ignore[operator]

        # Move to appropriate GPU
        device = torch.device(f"cuda:{self._rank}")
        stage_model = stage.submod.to(device)

        logger.info(f"Rank {self._rank}: Automatic split successful, stage on {device}")
        return stage_model  # type: ignore[no-any-return]

    def _split_model_manual(self, model: PreTrainedModel) -> PreTrainedModel:
        """Manual layer-based model splitting.

        This is a fallback for models that don't support torch.export.
        Splits based on the model's layer structure.
        """
        # Identify model layers (dynamic attribute access based on model architecture)
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama-style models
            layers = list(model.model.layers)  # type: ignore[arg-type]
            embed = model.model.embed_tokens  # type: ignore[union-attr]
            norm = model.model.norm  # type: ignore[union-attr]
            lm_head = model.lm_head  # type: ignore[union-attr]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-style models
            layers = list(model.transformer.h)  # type: ignore[arg-type]
            embed = model.transformer.wte  # type: ignore[union-attr]
            norm = model.transformer.ln_f  # type: ignore[union-attr]
            lm_head = model.lm_head  # type: ignore[union-attr]
        else:
            raise ValueError(
                f"Cannot identify layer structure for {type(model).__name__}. "
                "Manual pipeline splitting not supported for this model architecture."
            )

        n_layers = len(layers)
        layers_per_stage = n_layers // self._num_stages
        remainder = n_layers % self._num_stages

        # Calculate which layers this rank handles
        start_idx = self._rank * layers_per_stage + min(self._rank, remainder)
        end_idx = start_idx + layers_per_stage + (1 if self._rank < remainder else 0)

        logger.info(f"Rank {self._rank}: Manual split - layers {start_idx}-{end_idx} of {n_layers}")

        # Create a wrapper that only runs this rank's layers
        stage_model = _PipelineStageWrapper(
            rank=self._rank,
            world_size=self._world_size,
            layers=layers[start_idx:end_idx],
            embed=embed if self._rank == 0 else None,
            norm=norm if self._rank == self._world_size - 1 else None,
            lm_head=lm_head if self._rank == self._world_size - 1 else None,
        )

        device = torch.device(f"cuda:{self._rank}")
        stage_model = stage_model.to(device)

        return stage_model  # type: ignore[return-value]

    def _create_split_spec(self, model: PreTrainedModel) -> dict[str, Any]:
        """Create split specification for torch.distributed.pipelining.

        Returns a dictionary mapping layer names to split points.
        """
        # This is model-architecture dependent
        # For Llama-style: split at model.model.layers[N]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            n_layers = len(model.model.layers)
            layers_per_stage = n_layers // self._num_stages

            split_spec = {}
            for i in range(1, self._num_stages):
                split_idx = i * layers_per_stage
                split_spec[f"model.layers.{split_idx}"] = "split"

            return split_spec

        # Default: empty spec triggers auto-splitting
        return {}

    def cleanup(self) -> None:
        """Destroy process group."""
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
                logger.debug("PipelineParallel process group destroyed")
            except Exception as e:
                logger.warning(f"Error destroying process group: {e}")

    @property
    def requires_torchrun(self) -> bool:
        """Pipeline parallelism requires torchrun launcher."""
        return True


class _PipelineStageWrapper(torch.nn.Module):  # type: ignore[misc]
    """Wrapper for a pipeline stage containing a subset of layers.

    This is used for manual layer-based pipeline splitting when
    torch.export-based splitting fails.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        layers: list[torch.nn.Module],
        embed: torch.nn.Module | None = None,
        norm: torch.nn.Module | None = None,
        lm_head: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.layers = torch.nn.ModuleList(layers)
        self.embed = embed
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process input through this stage's layers.

        Args:
            hidden_states: Input tensor (embeddings from rank 0, or
                          hidden states from previous rank).

        Returns:
            Hidden states to pass to next rank, or logits from final rank.
        """
        # First rank: apply embedding
        if self.embed is not None:
            hidden_states = self.embed(hidden_states)

        # Process through this stage's layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]  # [0] for hidden_states only

        # Last rank: apply norm and lm_head
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            hidden_states = self.lm_head(hidden_states)

        return hidden_states


def get_parallelism_strategy(config: ShardingConfig) -> ParallelismStrategy:
    """Factory function to get appropriate parallelism strategy.

    Args:
        config: Sharding configuration specifying the strategy.

    Returns:
        Configured parallelism strategy instance.

    Raises:
        ValueError: If strategy is unknown.
    """
    strategy_map: dict[str, type[ParallelismStrategy]] = {
        "none": NoParallelism,
        "tensor_parallel": TensorParallelStrategy,
        "pipeline_parallel": PipelineParallelStrategy,
    }

    strategy_cls = strategy_map.get(config.strategy)
    if strategy_cls is None:
        raise ValueError(
            f"Unknown sharding strategy: {config.strategy}. "
            f"Supported: {list(strategy_map.keys())}"
        )

    return strategy_cls()


def is_model_tp_compatible(model_name: str) -> bool:
    """Check if a model likely supports HuggingFace tensor parallelism.

    Args:
        model_name: HuggingFace model name or path.

    Returns:
        True if model architecture is known to support TP.
    """
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in TP_SUPPORTED_MODEL_PATTERNS)
