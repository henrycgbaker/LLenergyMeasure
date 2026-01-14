"""Configuration models for LLM Bench experiments."""

from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Discriminator,
    Field,
    Tag,
    field_validator,
    model_validator,
)

# Built-in dataset aliases for prompt loading
BUILTIN_DATASETS: dict[str, dict[str, str]] = {
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


class ShardingConfig(BaseModel):
    """Model sharding configuration for multi-GPU parallelism.

    Strategies:
    - none: Default device_map='auto' behaviour (sequential layer distribution)
    - tensor_parallel: Split layers across GPUs (HuggingFace native tp_plan)
    - pipeline_parallel: Split model into sequential stages across GPUs

    Tensor Parallelism:
        Requires torchrun launcher. Supported models: Llama, Mistral, Mixtral,
        Qwen, Phi, Gemma, Falcon, MPT, BLOOM, OPT.

    Pipeline Parallelism:
        Uses torch.distributed.pipelining with microbatching for throughput.
        Schedules: 'gpipe' (fill-drain) or '1f1b' (interleaved).
    """

    strategy: Literal["none", "tensor_parallel", "pipeline_parallel"] = Field(
        default="none", description="Sharding strategy"
    )
    num_shards: int = Field(default=1, ge=1, description="Number of shards/stages")

    # Tensor parallelism options
    tp_plan: Literal["auto"] | None = Field(
        default=None,
        description="Tensor parallel plan ('auto' uses model's predefined config)",
    )

    # Pipeline parallelism options
    pipeline_schedule: Literal["gpipe", "1f1b"] = Field(
        default="gpipe",
        description="Pipeline schedule: 'gpipe' (fill-drain) or '1f1b' (interleaved)",
    )
    num_microbatches: int = Field(
        default=4,
        ge=1,
        description="Number of microbatches for pipeline parallelism",
    )

    @model_validator(mode="after")
    def set_strategy_defaults(self) -> "ShardingConfig":
        """Set strategy-specific defaults."""
        # TP defaults to tp_plan="auto" if not specified
        if self.strategy == "tensor_parallel" and self.tp_plan is None:
            object.__setattr__(self, "tp_plan", "auto")
        return self


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


# Backwards compatibility alias
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


# Sampling presets aligned with industry best practices (vLLM, OpenAI, MLPerf)
SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "deterministic": {"temperature": 0.0, "do_sample": False},
    "standard": {"temperature": 1.0, "do_sample": True, "top_p": 0.95, "top_k": 50},
    "creative": {"temperature": 0.8, "do_sample": True, "top_p": 0.9, "repetition_penalty": 1.1},
    "factual": {"temperature": 0.3, "do_sample": True, "top_k": 10},
}


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


def _get_prompt_source_type(v: Any) -> str:
    """Discriminator function for PromptSourceConfig union."""
    if isinstance(v, dict):
        return str(v.get("type", "file"))
    return str(getattr(v, "type", "file"))


PromptSourceConfig = Annotated[
    Annotated[FilePromptSource, Tag("file")]
    | Annotated[HuggingFacePromptSource, Tag("huggingface")],
    Discriminator(_get_prompt_source_type),
]


class ExperimentConfig(BaseModel):
    """Main experiment configuration.

    This is the central configuration object that controls all aspects
    of an LLM benchmarking experiment.
    """

    # Identity
    config_name: str = Field(..., min_length=1, description="Unique config identifier")
    model_name: str = Field(..., min_length=1, description="HuggingFace model name/path")

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

    # Prompt source (optional - can also be specified via CLI)
    # YAML alias: "prompts" (preferred) or "prompt_source" (legacy)
    prompt_source: PromptSourceConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("prompts", "prompt_source"),
        description="Prompt source: file or huggingface dataset",
    )

    # Distributed configuration
    # YAML alias: "gpus" (preferred) or "gpu_list" (legacy)
    gpu_list: list[int] = Field(
        default_factory=lambda: [0],
        validation_alias=AliasChoices("gpus", "gpu_list"),
        description="GPU indices to use",
    )
    num_processes: int = Field(default=1, ge=1, description="Number of processes")

    # Sub-configurations
    # YAML aliases: short names (preferred) or full names (legacy)
    batching_options: BatchingConfig = Field(
        default_factory=BatchingConfig,
        validation_alias=AliasChoices("batching", "batching_options"),
        description="Batching config",
    )
    sharding_config: ShardingConfig = Field(
        default_factory=ShardingConfig,
        validation_alias=AliasChoices("sharding", "sharding_config"),
        description="Sharding config",
    )
    latency_simulation: LatencySimulation = Field(
        default_factory=LatencySimulation,
        validation_alias=AliasChoices("traffic_simulation", "latency_simulation"),
        description="Traffic simulation config",
    )
    decoder_config: DecoderConfig = Field(
        default_factory=DecoderConfig,
        validation_alias=AliasChoices("decoder", "decoder_config"),
        description="Decoder/generation config",
    )
    quantization_config: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        validation_alias=AliasChoices("quantization", "quantization_config"),
        description="Quantization config",
    )
    schedule_config: ScheduleConfig = Field(
        default_factory=ScheduleConfig,
        validation_alias=AliasChoices("schedule", "schedule_config"),
        description="Schedule config for daemon mode",
    )

    # Precision and backend
    fp_precision: Literal["float32", "float16", "bfloat16"] = Field(
        default="float16", description="Floating point precision"
    )
    backend: Literal["pytorch", "tensorrt", "vllm"] = Field(
        default="pytorch", description="Inference backend"
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
    def validate_config(self) -> "ExperimentConfig":
        # Validate num_processes <= len(gpu_list)
        if self.num_processes > len(self.gpu_list):
            raise ValueError(
                f"num_processes ({self.num_processes}) must be <= "
                f"len(gpu_list) ({len(self.gpu_list)})"
            )

        # Validate min_output_tokens <= max_output_tokens
        if self.min_output_tokens > self.max_output_tokens:
            raise ValueError(
                f"min_output_tokens ({self.min_output_tokens}) must be <= "
                f"max_output_tokens ({self.max_output_tokens})"
            )

        return self

    @field_validator("gpu_list", mode="before")
    @classmethod
    def ensure_gpu_list(cls, v: Any) -> list[int]:
        if isinstance(v, int):
            return [v]
        return list(v)
