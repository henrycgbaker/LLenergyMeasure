"""Configuration models for LLM Bench experiments.

This module defines the Tier 1 (Universal) configuration that applies identically
across all backends. Backend-specific parameters live in backend_configs.py.
"""

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
    from llenergymeasure.config.backend_configs import (
        PyTorchConfig,
        TensorRTConfig,
        VLLMConfig,
    )

# Default dataset for experiments (AI Energy Score standardised benchmark)
DEFAULT_DATASET = "ai-energy-score"

# Built-in dataset aliases for prompt loading (SSOT for CLI + docs)
# Each entry includes metadata used by: CLI listing, doc generation, tests
BUILTIN_DATASETS: dict[str, dict[str, str]] = {
    "ai-energy-score": {
        "path": "AIEnergyScore/text_generation",
        "column": "text",
        "split": "train",
        "description": "AI Energy Score benchmark (default)",
    },
    # Underscore variant for YAML convenience
    "ai_energy_score": {
        "path": "AIEnergyScore/text_generation",
        "column": "text",
        "split": "train",
        "description": "AI Energy Score benchmark (alias)",
    },
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "column": "instruction",
        "split": "train",
        "description": "Instruction-following prompts (52k)",
    },
    "sharegpt": {
        "path": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "column": "conversations",
        "split": "train",
        "description": "Real user conversations",
    },
    "gsm8k": {
        "path": "gsm8k",
        "subset": "main",
        "column": "question",
        "split": "train",
        "description": "Primary school maths reasoning",
    },
    "mmlu": {
        "path": "cais/mmlu",
        "subset": "all",
        "column": "question",
        "split": "test",
        "description": "Multi-task knowledge questions",
    },
}

# Column names to try for auto-detection (order matters - first match wins)
AUTO_DETECT_COLUMNS = ["text", "prompt", "question", "instruction", "input", "content"]


# =============================================================================
# Tier 1: Universal Configurations (identical semantics across all backends)
# =============================================================================


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
    "standard": {"temperature": 1.0, "do_sample": True, "top_p": 0.95},
    "creative": {"temperature": 0.8, "do_sample": True, "top_p": 0.9, "repetition_penalty": 1.1},
    "factual": {"temperature": 0.3, "do_sample": True},
}


class DecoderConfig(BaseModel):
    """Universal decoder/generation configuration.

    Contains parameters with identical semantics across all backends.
    Backend-specific decoder params (min_p, beam_search) are in backend configs.

    top_k (Universal):
        All backends support top_k with identical semantics: sample from top K tokens.
        The "disabled" convention differs across backends but we normalise it:

        | Backend    | Config Value | Disabled Value | Conversion            |
        |------------|--------------|----------------|-----------------------|
        | PyTorch    | 0            | 0              | Pass as-is            |
        | vLLM       | 0            | -1             | Convert 0 → -1        |
        | TensorRT   | 0            | 0              | Pass as-is            |

        Users set top_k=0 to disable; backends handle the conversion internally.

    Presets:
        - deterministic: Greedy decoding (temp=0, do_sample=False)
        - standard: Balanced sampling (temp=1.0, top_p=0.95)
        - creative: Higher variance (temp=0.8, top_p=0.9, repetition_penalty=1.1)
        - factual: Lower variance (temp=0.3)
    """

    # Core sampling (universal across all backends)
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Sampling temperature (0=greedy)"
    )
    do_sample: bool = Field(default=True, description="Enable sampling (ignored if temp=0)")

    # Top-k sampling (universal - all backends support with same semantics)
    top_k: int = Field(default=50, ge=0, description="Top-k sampling (0=disabled)")

    # Nucleus sampling (universal)
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p nucleus sampling (1.0=disabled)"
    )

    # Repetition control (universal)
    repetition_penalty: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Repetition penalty (1.0=no penalty)"
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


def _validate_sampling_presets() -> None:
    """Validate SAMPLING_PRESETS keys match DecoderConfig fields at import time.

    This ensures typos in preset definitions are caught immediately rather than
    silently ignored at runtime.
    """
    valid_fields = set(DecoderConfig.model_fields.keys())
    for preset_name, values in SAMPLING_PRESETS.items():
        invalid_keys = set(values.keys()) - valid_fields
        if invalid_keys:
            raise ValueError(
                f"SAMPLING_PRESETS['{preset_name}'] has invalid keys: {invalid_keys}. "
                f"Valid keys are: {valid_fields}"
            )


# Validate presets at import time (SSOT enforcement)
_validate_sampling_presets()


# =============================================================================
# Prompt/Dataset Configuration
# =============================================================================


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


# =============================================================================
# Measurement Configuration (Phase 1 extensions)
# =============================================================================


class WarmupConfig(BaseModel):
    """Warmup convergence configuration.

    Controls the warmup phase before measurement begins. Supports either
    CV-based convergence detection (stop when latency stabilises) or
    fixed iteration count.
    """

    enabled: bool = Field(default=True, description="Enable warmup phase before inference")
    convergence_detection: bool = Field(
        default=True,
        description="Use CV-based convergence detection (false = fixed iterations)",
    )
    cv_threshold: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Target CV threshold (default 5%)",
    )
    max_prompts: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum warmup iterations (safety cap)",
    )
    window_size: int = Field(
        default=5,
        ge=3,
        le=20,
        description="Rolling window size for CV calculation",
    )
    min_prompts: int = Field(
        default=5,
        ge=1,
        description="Minimum warmup prompts before checking convergence",
    )

    @model_validator(mode="after")
    def validate_window_size(self) -> "WarmupConfig":
        """Ensure window_size and min_prompts are within max_prompts."""
        if self.window_size > self.max_prompts:
            raise ValueError(
                f"window_size ({self.window_size}) must be <= max_prompts ({self.max_prompts})"
            )
        if self.min_prompts > self.max_prompts:
            raise ValueError(
                f"min_prompts ({self.min_prompts}) must be <= max_prompts ({self.max_prompts})"
            )
        return self


class BaselineConfig(BaseModel):
    """Baseline power measurement configuration.

    Controls whether and how idle GPU power is measured before experiments,
    enabling baseline-adjusted energy attribution.
    """

    enabled: bool = Field(default=True, description="Enable baseline power measurement")
    required: bool = Field(
        default=False,
        description="Fail experiment if baseline measurement fails (false = warn and continue)",
    )
    duration_sec: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Baseline measurement duration in seconds",
    )
    cache_ttl_sec: float = Field(
        default=3600.0,
        ge=60.0,
        description="Cache validity in seconds (default 1 hour)",
    )
    sample_interval_ms: int = Field(
        default=100,
        ge=50,
        le=1000,
        description="Sampling interval in milliseconds",
    )


class TimeSeriesConfig(BaseModel):
    """Time-series data collection configuration.

    Controls whether power/temperature/utilisation time-series data
    is collected and saved during experiments.
    """

    enabled: bool = Field(default=False, description="Enable time-series data collection")
    save: bool = Field(
        default=False,
        description="Save time-series to separate file (--save-timeseries)",
    )
    sample_interval_ms: int = Field(
        default=100,
        ge=50,
        le=5000,
        description="Sampling interval in ms (100ms = 10Hz, 1000ms = 1Hz)",
    )


# =============================================================================
# Main Experiment Configuration
# =============================================================================


class ExperimentConfig(BaseModel):
    """Main experiment configuration.

    This is the central configuration object that controls all aspects
    of an LLM benchmarking experiment.

    Structure:
    - Tier 1 (Universal): Parameters at the top level with identical semantics
      across all backends (model, tokens, dataset, decoder, schedule, etc.)
    - Tier 2 (Backend-specific): Parameters in backend sections (pytorch/vllm/tensorrt)
      that use native parameter names for each backend
    """

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------
    config_name: str = Field(..., min_length=1, description="Unique config identifier")
    model_name: str = Field(..., min_length=1, description="HuggingFace model name/path")
    adapter: str | None = Field(
        default=None,
        description="LoRA adapter: HuggingFace Hub ID or local path",
    )

    # -------------------------------------------------------------------------
    # Token Limits
    # -------------------------------------------------------------------------
    max_input_tokens: int = Field(default=512, ge=1, description="Max input tokens")
    max_output_tokens: int = Field(default=128, ge=1, description="Max output tokens")
    min_output_tokens: int = Field(default=0, ge=0, description="Min output tokens")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    gpus: list[int] = Field(
        default_factory=lambda: [0],
        description="GPU indices to use",
    )
    fp_precision: Literal["float32", "float16", "bfloat16"] = Field(
        default="float16", description="Floating point precision"
    )

    # -------------------------------------------------------------------------
    # Universal Sub-Configurations
    # -------------------------------------------------------------------------
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig,
        description="Universal decoder/generation configuration",
    )
    traffic_simulation: TrafficSimulation = Field(
        default_factory=TrafficSimulation,
        description="MLPerf-style traffic simulation",
    )
    schedule: ScheduleConfig = Field(
        default_factory=ScheduleConfig,
        description="Schedule config for daemon mode",
    )
    io: IOConfig = Field(
        default_factory=IOConfig,
        description="I/O paths configuration",
    )

    # -------------------------------------------------------------------------
    # Measurement Configuration (Phase 1)
    # -------------------------------------------------------------------------
    warmup: WarmupConfig = Field(
        default_factory=WarmupConfig,
        description="Warmup convergence configuration",
    )
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig,
        description="Baseline power measurement configuration",
    )
    timeseries: TimeSeriesConfig = Field(
        default_factory=TimeSeriesConfig,
        description="Time-series data collection configuration",
    )

    # -------------------------------------------------------------------------
    # Streaming (TTFT/ITL metrics)
    # -------------------------------------------------------------------------
    streaming: bool = Field(
        default=False,
        description="Enable streaming mode for TTFT/ITL latency measurement.",
    )
    streaming_warmup_requests: int = Field(
        default=5,
        ge=0,
        description="Warmup requests before streaming measurement (excluded from stats)",
    )

    # -------------------------------------------------------------------------
    # Backend Selection
    # -------------------------------------------------------------------------
    backend: Literal["pytorch", "tensorrt", "vllm"] = Field(
        default="pytorch", description="Inference backend"
    )

    # -------------------------------------------------------------------------
    # Backend-Specific Configurations (Tier 2)
    # -------------------------------------------------------------------------
    # These contain all backend-native parameters including batching,
    # quantization, parallelism, and decoder extensions
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

    # -------------------------------------------------------------------------
    # Experiment Tracking
    # -------------------------------------------------------------------------
    cycle_id: int | None = Field(default=None, description="Experiment cycle ID")
    num_cycles: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Number of cycles for statistical robustness (1-10). "
            "With 1 cycle, confidence intervals and robustness metrics "
            "cannot be computed. Use >= 3 cycles for basic statistical "
            "validity, >= 5 for publication-grade results."
        ),
    )
    query_rate: float = Field(default=1.0, ge=0, description="Query rate (queries/sec)")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducibility (None = non-deterministic)"
    )

    # -------------------------------------------------------------------------
    # Extra
    # -------------------------------------------------------------------------
    extra_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_config(self) -> "ExperimentConfig":
        """Validate config constraints."""
        # Validate min_output_tokens <= max_output_tokens
        if self.min_output_tokens > self.max_output_tokens:
            raise ValueError(
                f"min_output_tokens ({self.min_output_tokens}) must be <= "
                f"max_output_tokens ({self.max_output_tokens})"
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

        # TensorRT doesn't support float32
        if self.backend == "tensorrt" and self.fp_precision == "float32":
            raise ValueError(
                "float32 precision is not supported with TensorRT backend. "
                "TensorRT-LLM is optimised for lower precision. "
                "Use fp_precision='float16' or 'bfloat16' instead."
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
def _rebuild_experiment_config() -> None:
    """Rebuild ExperimentConfig to resolve forward references."""
    from llenergymeasure.config.backend_configs import (
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
