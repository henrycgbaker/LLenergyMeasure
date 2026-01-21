"""Configuration models for LLM Bench experiments."""

import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from llm_energy_measure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

# Default dataset for experiments (AI Energy Score standardised benchmark)
DEFAULT_DATASET = "ai-energy-score"

# Built-in dataset aliases for prompt loading
BUILTIN_DATASETS: dict[str, dict[str, str]] = {
    "ai-energy-score": {
        "path": "AIEnergyScore/text_generation",
        "column": "text",
        "split": "train",
    },
    # Underscore variant for YAML convenience
    "ai_energy_score": {
        "path": "AIEnergyScore/text_generation",
        "column": "text",
        "split": "train",
    },
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "column": "instruction",
        "split": "train",
    },
    "sharegpt": {
        "path": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "column": "conversations",
        "split": "train",
    },
    "gsm8k": {
        "path": "gsm8k",
        "subset": "main",
        "column": "question",
        "split": "train",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "subset": "all",
        "column": "question",
        "split": "test",
    },
}

# Column names to try for auto-detection (order matters - first match wins)
AUTO_DETECT_COLUMNS = ["text", "prompt", "question", "instruction", "input", "content"]


class BatchingConfig(BaseModel):
    """Batching configuration for inference.

    Industry-standard batching strategies (per MLPerf/vLLM terminology):
    - static: Fixed batch size, pads to uniform length (MLPerf offline scenario)
    - dynamic: Token-aware batching, groups by token budget (MLPerf server scenario)
    - sorted_static: Sort by length then static batches (reduces padding waste)
    - sorted_dynamic: Sort by length + dynamic token budget (optimal packing)
    """

    batch_size: int = Field(default=1, ge=1, description="Max prompts per batch")
    strategy: Literal["static", "dynamic", "sorted_static", "sorted_dynamic"] = Field(
        default="static", description="Batching strategy (MLPerf terminology)"
    )
    max_tokens_per_batch: int | None = Field(
        default=None,
        description="Max tokens per batch (for dynamic strategies). Defaults to max_input_tokens.",
    )

    # Legacy field for backwards compatibility
    dynamic_batching: bool = Field(
        default=False,
        description="[Deprecated] Use strategy='dynamic' instead. Kept for backwards compat.",
    )

    @model_validator(mode="after")
    def handle_legacy_dynamic_batching(self) -> "BatchingConfig":
        """Map legacy dynamic_batching flag to strategy."""
        if self.dynamic_batching and self.strategy == "static":
            object.__setattr__(self, "strategy", "dynamic")
        return self


class ParallelismConfig(BaseModel):
    """Unified parallelism configuration for multi-GPU inference.

    This config consolidates GPU parallelism settings that were previously split
    between `num_processes`, `sharding.num_shards`, and `sharding.strategy`.

    Strategies:
    - none: Single GPU or device_map='auto' (sequential layer distribution)
    - tensor_parallel: Split layers horizontally across GPUs
    - pipeline_parallel: Split model vertically into sequential stages
    - data_parallel: Replicate model across GPUs, split batches

    Examples:
        # Single GPU
        parallelism:
          strategy: none

        # 2-way tensor parallelism
        parallelism:
          strategy: tensor_parallel
          degree: 2

        # 4-way data parallelism
        parallelism:
          strategy: data_parallel
          degree: 4
    """

    strategy: Literal["none", "tensor_parallel", "pipeline_parallel", "data_parallel"] = Field(
        default="none",
        description="Parallelism strategy",
    )
    degree: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs/workers for parallelism",
    )

    # Advanced options
    tp_plan: Literal["auto"] | None = Field(
        default=None,
        description="Tensor parallel plan ('auto' uses model's predefined config)",
    )

    @model_validator(mode="after")
    def validate_parallelism(self) -> "ParallelismConfig":
        """Validate parallelism settings and set defaults."""
        from loguru import logger

        # Warn if strategy=none but degree > 1 (contradiction)
        if self.strategy == "none" and self.degree > 1:
            logger.warning(
                f"parallelism.strategy='none' with degree={self.degree} is contradictory. "
                f"Setting degree=1. Use a parallelism strategy (tensor_parallel, "
                f"data_parallel, pipeline_parallel) for multi-GPU."
            )
            object.__setattr__(self, "degree", 1)

        # TP defaults to tp_plan="auto" if not specified
        if self.strategy == "tensor_parallel" and self.tp_plan is None:
            object.__setattr__(self, "tp_plan", "auto")

        return self


class ShardingConfig(BaseModel):
    """[DEPRECATED] Use ParallelismConfig instead.

    Model sharding configuration for multi-GPU parallelism.
    Kept for backwards compatibility - values are migrated to ParallelismConfig.

    Strategies:
    - none: Default device_map='auto' behaviour (sequential layer distribution)
    - tensor_parallel: Split layers horizontally across GPUs (HuggingFace native)
    - pipeline_parallel: Split model vertically into sequential stages
    """

    strategy: Literal["none", "tensor_parallel", "pipeline_parallel"] = Field(
        default="none", description="Sharding strategy"
    )
    num_shards: int = Field(default=1, ge=1, description="Number of GPUs for parallelism")

    # Tensor parallelism options
    tp_plan: Literal["auto"] | None = Field(
        default=None,
        description="Tensor parallel plan ('auto' uses model's predefined config)",
    )

    @model_validator(mode="after")
    def set_strategy_defaults(self) -> "ShardingConfig":
        """Set strategy-specific defaults."""
        # TP defaults to tp_plan="auto" if not specified
        if self.strategy == "tensor_parallel" and self.tp_plan is None:
            object.__setattr__(self, "tp_plan", "auto")
        return self

    def to_parallelism_config(self) -> ParallelismConfig:
        """Convert to new ParallelismConfig format."""
        return ParallelismConfig(
            strategy=self.strategy,  # type: ignore[arg-type]
            degree=self.num_shards,
            tp_plan=self.tp_plan,
        )


class TrafficSimulation(BaseModel):
    """MLPerf-style traffic simulation for realistic load testing.

    Modes:
    - constant: Fixed inter-arrival time (1/target_qps seconds)
    - poisson: Exponential inter-arrival times (MLPerf server scenario)

    The Poisson mode models real-world API traffic where requests arrive
    randomly following a Poisson process with rate λ = target_qps.
    """

    enabled: bool = Field(default=False, description="Enable traffic simulation")
    mode: Literal["constant", "poisson"] = Field(
        default="poisson", description="Traffic arrival pattern (MLPerf terminology)"
    )
    target_qps: float = Field(
        default=1.0, gt=0, description="Target queries per second (arrival rate λ)"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducible Poisson arrivals"
    )


# Backwards compatibility alias (deprecated, use TrafficSimulation)
LatencySimulation = TrafficSimulation

# Valid day names for schedule configuration
VALID_DAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
DAY_ALIASES = {"weekdays": ["mon", "tue", "wed", "thu", "fri"], "weekends": ["sat", "sun"]}


class ScheduleConfig(BaseModel):
    """Schedule configuration for daemon mode experiments.

    Supports interval-based scheduling, time-of-day scheduling, and day filtering.
    All options can be combined for flexible scheduling patterns.

    Examples:
        - interval: "6h" → run every 6 hours
        - at: "09:00" → run daily at 9am
        - at: "09:00", days: ["mon", "wed", "fri"] → 9am on Mon/Wed/Fri
        - interval: "12h", days: ["sat", "sun"] → every 12h on weekends
    """

    enabled: bool = Field(default=False, description="Enable scheduled mode")
    interval: str | None = Field(
        default=None,
        description="Interval between runs (e.g., '6h', '30m', '1d')",
    )
    at: str | None = Field(
        default=None,
        description="Specific time of day to run (e.g., '09:00', '14:30')",
    )
    days: list[str] | None = Field(
        default=None,
        description="Days to run on (e.g., ['mon', 'wed', 'fri'] or ['weekdays'])",
    )
    total_duration: str = Field(
        default="24h",
        description="Total duration to run daemon (e.g., '24h', '7d')",
    )

    @field_validator("days", mode="before")
    @classmethod
    def expand_day_aliases(cls, v: list[str] | str | None) -> list[str] | None:
        """Expand day aliases like 'weekdays' and 'weekends'."""
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        expanded: list[str] = []
        for day in v:
            day_lower = day.lower()
            if day_lower in DAY_ALIASES:
                expanded.extend(DAY_ALIASES[day_lower])
            elif day_lower in VALID_DAYS:
                expanded.append(day_lower)
            else:
                raise ValueError(
                    f"Invalid day '{day}'. Valid: {sorted(VALID_DAYS)} or {list(DAY_ALIASES.keys())}"
                )
        return expanded

    @model_validator(mode="after")
    def validate_schedule_has_timing(self) -> "ScheduleConfig":
        """Ensure at least one timing option is set when enabled."""
        if self.enabled and not self.interval and not self.at:
            raise ValueError("Schedule requires either 'interval' or 'at' to be set")
        return self


class IOConfig(BaseModel):
    """I/O configuration for experiment results and data paths.

    Allows per-experiment override of results directory. Precedence:
    1. CLI flag --results-dir (highest)
    2. Config YAML io.results_dir
    3. .env file LLM_ENERGY_RESULTS_DIR
    4. Default "results/" (lowest)
    """

    results_dir: str | None = Field(
        default=None,
        description="Results output directory (overrides .env default)",
    )


# Sampling presets aligned with industry best practices (vLLM, OpenAI, MLPerf)
SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "deterministic": {"temperature": 0.0, "do_sample": False},
    "standard": {"temperature": 1.0, "do_sample": True, "top_p": 0.95, "top_k": 50},
    "creative": {"temperature": 0.8, "do_sample": True, "top_p": 0.9, "repetition_penalty": 1.1},
    "factual": {"temperature": 0.3, "do_sample": True, "top_k": 10},
}


class BeamSearchConfig(BaseModel):
    """Beam search configuration for generation.

    Beam search explores multiple candidate sequences in parallel, selecting
    the most probable overall sequence. Generally produces higher quality
    output at the cost of throughput.

    Note: Beam search is typically mutually exclusive with sampling.
    """

    enabled: bool = Field(default=False, description="Enable beam search (disables sampling)")
    num_beams: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Beam width (1=greedy, >1=beam search)",
    )
    length_penalty: float = Field(
        default=1.0,
        description="Exponential length penalty (>1 favours longer, <1 favours shorter)",
    )
    early_stopping: bool = Field(
        default=False,
        description="Stop when num_beams best sequences complete",
    )
    no_repeat_ngram_size: int = Field(
        default=0,
        ge=0,
        description="Prevent n-gram repetition within beam (0=disabled)",
    )


class DecoderConfig(BaseModel):
    """Decoder/generation configuration.

    Supports industry-standard sampling parameters aligned with vLLM/HuggingFace.
    Use `preset` for common configurations or set individual parameters.

    Presets:
    - deterministic: Greedy decoding (temp=0, do_sample=False)
    - standard: Balanced sampling (temp=1.0, top_p=0.95, top_k=50)
    - creative: Higher variance (temp=0.8, top_p=0.9, repetition_penalty=1.1)
    - factual: Lower variance (temp=0.3, top_k=10)
    """

    # Core sampling
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature (0=greedy)"
    )
    do_sample: bool = Field(default=True, description="Enable sampling (ignored if temp=0)")

    # Nucleus/top sampling
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling (1.0=disabled)"
    )
    top_k: int = Field(default=50, ge=0, description="Top-k sampling (0=disabled)")
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Min probability relative to top token (0=disabled)",
    )

    # Repetition control
    repetition_penalty: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Repetition penalty (1.0=no penalty)"
    )
    no_repeat_ngram_size: int = Field(
        default=0, ge=0, description="Prevent n-gram repetition (0=disabled)"
    )

    # Beam search configuration (unified across backends)
    beam_search: BeamSearchConfig = Field(
        default_factory=BeamSearchConfig,
        description="Beam search configuration",
    )

    # Preset shortcut
    preset: Literal["deterministic", "standard", "creative", "factual"] | None = Field(
        default=None,
        description="Sampling preset (expands to preset values, overrides apply on top)",
    )

    @model_validator(mode="before")
    @classmethod
    def apply_preset(cls, data: Any) -> Any:
        """Expand preset, then apply explicit overrides on top."""
        if (
            isinstance(data, dict)
            and (preset_name := data.get("preset"))
            and preset_name in SAMPLING_PRESETS
        ):
            # Preset values first, then user overrides on top
            return {**SAMPLING_PRESETS[preset_name], **data}
        return data

    @property
    def is_deterministic(self) -> bool:
        """True if using greedy decoding (temp=0 or do_sample=False)."""
        return self.temperature == 0.0 or not self.do_sample

    @property
    def use_beam_search(self) -> bool:
        """True if beam search is enabled and num_beams > 1."""
        return self.beam_search.enabled and self.beam_search.num_beams > 1


class QuantizationConfig(BaseModel):
    """Quantization configuration for model loading."""

    quantization: bool = Field(default=False, description="Enable quantization")
    load_in_4bit: bool = Field(default=False, description="Load in 4-bit (BNB)")
    load_in_8bit: bool = Field(default=False, description="Load in 8-bit (BNB)")
    bnb_4bit_compute_dtype: str = Field(default="float16", description="Compute dtype for 4-bit")
    bnb_4bit_quant_type: str = Field(default="nf4", description="Quantization type (nf4, fp4)")
    bnb_4bit_use_double_quant: bool = Field(default=False, description="Use double quantization")

    @model_validator(mode="after")
    def validate_quantization_exclusivity(self) -> "QuantizationConfig":
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
        if (self.load_in_4bit or self.load_in_8bit) and not self.quantization:
            # Auto-enable quantization flag
            object.__setattr__(self, "quantization", True)
        return self


class FilePromptSource(BaseModel):
    """Load prompts from a text file (one per line)."""

    type: Literal["file"] = "file"
    path: str = Field(..., description="Path to prompts file")


class HuggingFacePromptSource(BaseModel):
    """Load prompts from a HuggingFace dataset.

    Supports built-in aliases (alpaca, gsm8k, mmlu, sharegpt) or any HF dataset path.
    Column auto-detection tries: text, prompt, question, instruction, input, content.
    """

    type: Literal["huggingface"] = "huggingface"
    dataset: str = Field(..., description="Dataset name: built-in alias or HuggingFace path")
    split: str = Field(default="train", description="Dataset split")
    subset: str | None = Field(default=None, description="Dataset subset/config name")
    column: str | None = Field(
        default=None, description="Column to extract (auto-detected if not set)"
    )
    sample_size: int | None = Field(default=None, ge=1, description="Limit number of prompts")
    shuffle: bool = Field(default=False, description="Shuffle before sampling")
    seed: int = Field(default=42, description="Random seed for shuffling")

    @model_validator(mode="after")
    def resolve_builtin_alias(self) -> "HuggingFacePromptSource":
        """Resolve built-in aliases to full dataset paths."""
        if self.dataset in BUILTIN_DATASETS:
            builtin = BUILTIN_DATASETS[self.dataset]
            # Only override if not explicitly set
            if self.column is None:
                object.__setattr__(self, "column", builtin.get("column"))
            if self.subset is None and "subset" in builtin:
                object.__setattr__(self, "subset", builtin["subset"])
            # Replace alias with full path
            object.__setattr__(self, "dataset", builtin["path"])
        return self


def _get_prompts_type(v: Any) -> str:
    """Discriminator function for prompts field union type."""
    if isinstance(v, dict):
        return str(v.get("type", "file"))
    return str(getattr(v, "type", "file"))


PromptSourceConfig = Annotated[
    Annotated[FilePromptSource, Tag("file")]
    | Annotated[HuggingFacePromptSource, Tag("huggingface")],
    Discriminator(_get_prompts_type),
]


class DatasetConfig(BaseModel):
    """Simple dataset configuration for convenience.

    A streamlined way to specify a dataset without the full PromptSourceConfig.
    For advanced options (shuffle, subset, custom seed), use the `prompts` field instead.

    Examples:
        dataset:
          name: alpaca
          sample_size: 100

        dataset:
          name: tatsu-lab/alpaca
          split: validation
          column: instruction
    """

    name: str = Field(..., description="Dataset name: built-in alias or HuggingFace path")
    sample_size: int | None = Field(default=None, ge=1, description="Limit number of prompts")
    split: str = Field(default="train", description="Dataset split")
    column: str | None = Field(
        default=None, description="Column for prompts (auto-detected if not set)"
    )


class ExperimentConfig(BaseModel):
    """Main experiment configuration.

    This is the central configuration object that controls all aspects
    of an LLM benchmarking experiment.
    """

    # Identity
    config_name: str = Field(..., min_length=1, description="Unique config identifier")
    model_name: str = Field(..., min_length=1, description="HuggingFace model name/path")
    adapter: str | None = Field(
        default=None,
        description="LoRA adapter: HuggingFace Hub ID or local path",
    )

    # Model properties
    is_encoder_decoder: bool = Field(default=False, description="Is encoder-decoder model")
    task_type: Literal["text_generation", "translation", "summarisation"] = Field(
        default="text_generation", description="Task type"
    )
    inference_type: Literal["pure_generative", "reasoning"] = Field(
        default="pure_generative", description="Inference type"
    )

    # Token limits
    max_input_tokens: int = Field(default=512, ge=1, description="Max input tokens")
    max_output_tokens: int = Field(default=128, ge=1, description="Max output tokens")
    min_output_tokens: int = Field(default=0, ge=0, description="Min output tokens")

    # Input configuration
    num_input_prompts: int = Field(default=1, ge=1, description="Number of prompts")
    save_outputs: bool = Field(default=False, description="Save generated outputs")
    decode_token_to_text: bool = Field(default=False, description="Decode tokens to text")

    # Dataset configuration (simple form - recommended for most use cases)
    dataset: DatasetConfig | None = Field(
        default=None,
        description="Simple dataset config. For advanced options, use 'prompts' instead.",
    )

    # Prompt source (advanced - for custom shuffle, subset, file source)
    prompts: PromptSourceConfig | None = Field(
        default=None,
        description="Advanced prompt source: file or huggingface dataset with full options",
    )

    # Distributed configuration
    gpus: list[int] = Field(
        default_factory=lambda: [0],
        description="GPU indices to use",
    )
    # DEPRECATED: Use parallelism.degree instead. Kept for backwards compatibility.
    num_processes: int = Field(
        default=1,
        ge=1,
        description="[Deprecated] Number of processes. Use parallelism.degree instead.",
    )

    # Sub-configurations (canonical names only)
    batching: BatchingConfig = Field(
        default_factory=BatchingConfig,
        description="Batching configuration",
    )
    sharding: ShardingConfig = Field(
        default_factory=ShardingConfig,
        description="[Deprecated] Use parallelism instead. Legacy sharding configuration.",
    )
    parallelism: ParallelismConfig = Field(
        default_factory=ParallelismConfig,
        description="Unified parallelism configuration for multi-GPU inference",
    )
    traffic_simulation: TrafficSimulation = Field(
        default_factory=TrafficSimulation,
        description="MLPerf-style traffic simulation",
    )
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig,
        description="Decoder/generation configuration",
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        description="Quantization configuration",
    )
    schedule: ScheduleConfig = Field(
        default_factory=ScheduleConfig,
        description="Schedule config for daemon mode",
    )
    io: IOConfig = Field(
        default_factory=IOConfig,
        description="I/O paths configuration",
    )

    # Precision and backend
    fp_precision: Literal["float32", "float16", "bfloat16"] = Field(
        default="float16", description="Floating point precision"
    )
    backend: Literal["pytorch", "tensorrt", "vllm"] = Field(
        default="pytorch", description="Inference backend"
    )

    # Streaming latency measurement (TTFT/ITL metrics)
    streaming: bool = Field(
        default=False,
        description="Enable streaming mode for TTFT/ITL latency measurement. "
        "Also a testable parameter - streaming may affect energy profile.",
    )
    streaming_warmup_requests: int = Field(
        default=5,
        ge=0,
        description="Warmup requests before streaming measurement (excluded from stats)",
    )

    # Backend-specific configurations
    # These are optional and only validated when the corresponding backend is selected
    vllm: "VLLMConfig | None" = Field(
        default=None,
        description="vLLM-specific configuration (only used when backend=vllm)",
    )
    pytorch: "PyTorchConfig | None" = Field(
        default=None,
        description="PyTorch-specific configuration (only used when backend=pytorch)",
    )
    tensorrt: "TensorRTConfig | None" = Field(
        default=None,
        description="TensorRT-LLM configuration (only used when backend=tensorrt)",
    )

    # Experiment tracking
    cycle_id: int | None = Field(default=None, description="Experiment cycle ID")
    num_cycles: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of cycles for statistical robustness (1-10)",
    )
    query_rate: float = Field(default=1.0, ge=0, description="Query rate (queries/sec)")

    # Reproducibility
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducibility (None = non-deterministic)"
    )

    # Extra metadata (for extensibility)
    extra_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_and_migrate_config(self) -> "ExperimentConfig":
        """Validate config and migrate legacy fields to new unified structure."""
        # Migrate legacy sharding config to new parallelism config
        # Only migrate if sharding was explicitly configured (not default)
        sharding_is_default = (
            self.sharding.strategy == "none"
            and self.sharding.num_shards == 1
            and self.sharding.tp_plan is None
        )
        parallelism_is_default = (
            self.parallelism.strategy == "none" and self.parallelism.degree == 1
        )

        if not sharding_is_default and parallelism_is_default:
            # Migrate from legacy sharding to new parallelism
            migrated = self.sharding.to_parallelism_config()
            object.__setattr__(self, "parallelism", migrated)

        # Migrate legacy num_processes to parallelism.degree for data parallelism
        if (
            self.num_processes > 1
            and self.parallelism.degree == 1
            and self.parallelism.strategy == "none"
        ):
            # Legacy num_processes implies data parallelism
            migrated = ParallelismConfig(
                strategy="data_parallel",
                degree=self.num_processes,
            )
            object.__setattr__(self, "parallelism", migrated)

        # Validate parallelism.degree <= len(gpus)
        if self.parallelism.degree > len(self.gpus):
            raise ValueError(
                f"parallelism.degree ({self.parallelism.degree}) must be <= "
                f"len(gpus) ({len(self.gpus)})"
            )

        # Legacy validation: num_processes <= len(gpus) (for backwards compatibility)
        if self.num_processes > len(self.gpus):
            raise ValueError(
                f"num_processes ({self.num_processes}) must be <= " f"len(gpus) ({len(self.gpus)})"
            )

        # Validate min_output_tokens <= max_output_tokens
        if self.min_output_tokens > self.max_output_tokens:
            raise ValueError(
                f"min_output_tokens ({self.min_output_tokens}) must be <= "
                f"max_output_tokens ({self.max_output_tokens})"
            )

        # Validate backend supports the requested parallelism strategy
        # PyTorch backend does not support pipeline parallelism for inference
        # (generate() requires full model access for token-by-token generation)
        if self.backend == "pytorch" and self.parallelism.strategy == "pipeline_parallel":
            raise ValueError(
                "Pipeline parallelism is not supported with PyTorch backend for inference. "
                "PyTorch's generate() requires full model access for autoregressive generation. "
                "Use backend='vllm' or backend='tensorrt' for pipeline parallel inference."
            )

        # Validate backend-specific config matches selected backend
        if self.vllm is not None and self.backend != "vllm":
            raise ValueError(
                f"vllm config provided but backend is '{self.backend}'. "
                "Set backend='vllm' or remove vllm config section."
            )
        if self.pytorch is not None and self.backend != "pytorch":
            raise ValueError(
                f"pytorch config provided but backend is '{self.backend}'. "
                "Set backend='pytorch' or remove pytorch config section."
            )
        if self.tensorrt is not None and self.backend != "tensorrt":
            raise ValueError(
                f"tensorrt config provided but backend is '{self.backend}'. "
                "Set backend='tensorrt' or remove tensorrt config section."
            )

        # Warn if both dataset and prompts are configured (redundant)
        if self.dataset is not None and self.prompts is not None:
            warnings.warn(
                "Both 'dataset' and 'prompts' are set. 'dataset' takes precedence. "
                "Consider using only one for clarity.",
                UserWarning,
                stacklevel=2,
            )

        return self

    @field_validator("gpus", mode="before")
    @classmethod
    def ensure_gpus_list(cls, v: Any) -> list[int]:
        """Ensure gpus is always a list of integers."""
        if isinstance(v, int):
            return [v]
        return list(v)


# Rebuild model to resolve forward references for backend configs
# Import here to avoid circular imports
def _rebuild_experiment_config() -> None:
    """Rebuild ExperimentConfig to resolve forward references."""
    from llm_energy_measure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

    ExperimentConfig.model_rebuild(
        _types_namespace={
            "VLLMConfig": VLLMConfig,
            "PyTorchConfig": PyTorchConfig,
            "TensorRTConfig": TensorRTConfig,
        }
    )


_rebuild_experiment_config()
