"""Configuration models for LLM efficiency measurement experiments (v2.0 schema).

This module defines the Tier 1 (Universal) configuration that applies identically
across all engines. Engine-specific parameters live in the generated
``llenergymeasure.engines.<engine>.config`` modules (one ``Config`` per engine,
projected from the mined schema, with ``engine_params`` / ``sampling_params``
sub-sections).

v2.0 field renames from v1.x:
    model_name         -> model
    fp_precision       -> dtype
    num_input_prompts  -> n
    extra_metadata     -> passthrough_kwargs

v2.0 removals:
    config_name, schema_version, TrafficSimulation, ScheduleConfig, IOConfig,
    query_rate, streaming, streaming_warmup_requests, save_outputs,
    decode_token_to_text, extra_metadata, gpus, min_output_tokens
"""

from __future__ import annotations

import difflib
import logging
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

from pydantic import BaseModel, Field, model_validator

from llenergymeasure.config.harness import HarnessConfig
from llenergymeasure.config.ssot import ALL_ENGINES, ENGINES, SAMPLING_PRESETS, Engine, engine_str
from llenergymeasure.config.warnings import ConfigValidationWarning

logger = logging.getLogger(__name__)

#: Valid energy sampler names for ``energy_sampler`` fields.
EnergySamplerName = Literal["auto", "nvml", "zeus", "codecarbon"]

#: Literal type of supported sampling presets (derived from SAMPLING_PRESETS keys).
SamplingPreset = Literal["deterministic", "standard", "creative", "factual"]

if TYPE_CHECKING:
    from llenergymeasure.config.engine_rules.loader import EngineRulesLoader
    from llenergymeasure.config.generated.tensorrt import Config as TensorRTConfig
    from llenergymeasure.config.generated.transformers import Config as TransformersConfig
    from llenergymeasure.config.generated.vllm import Config as VLLMConfig


@lru_cache(maxsize=1)
def _get_rules_loader() -> EngineRulesLoader:
    # Lazy import so module load doesn't read YAML off disk. Tests substitute
    # via ``monkeypatch.setattr(models, "_get_rules_loader", ...)``.
    from llenergymeasure.config.engine_rules.loader import EngineRulesLoader

    return EngineRulesLoader()


def _reset_rules_loader_cache() -> None:
    """Clear the memoised loader; used by tests that mutate the on-disk corpus."""
    _get_rules_loader.cache_clear()


# Soft-validation cutoff for difflib suggestions on extra keys. 0.8 keeps the
# suggestion conservative (clear typos like ``dtypee`` -> ``dtype``) without
# nagging on genuinely-new engine fields that happen to share a prefix.
_CLOSE_MATCH_CUTOFF = 0.8


def _nested_subsection_models(section: BaseModel) -> dict[str, type[BaseModel]]:
    """Return the ``engine_params`` / ``sampling_params`` sub-model classes.

    Detects the generated nested shape by field presence rather than engine
    name; a section lacking both sub-fields yields an empty dict.
    """
    models: dict[str, type[BaseModel]] = {}
    for sub_name in ("engine_params", "sampling_params"):
        field = type(section).model_fields.get(sub_name)
        if field is None:
            continue
        for candidate in get_args(field.annotation) or (field.annotation,):
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                models[sub_name] = candidate
                break
    return models


@lru_cache(maxsize=16)
def _discovered_field_names(engine: str, sub_name: str) -> frozenset[str]:
    """Discovered-schema field names for a sub-section; empty if no schema ships.

    Broadens the soft-validation vocabulary beyond the curated ``model_fields``
    so a typo of an un-curated-but-discovered passthrough field is still caught.
    Memoised per ``(engine, sub_name)`` (the discovered schema is a stable
    committed artifact) so a sweep validating many configs reads each schema
    once; tests that mutate the on-disk schema call
    ``_reset_discovered_field_cache``.
    """
    from llenergymeasure.config.schema_loader import SchemaLoader

    try:
        schema = SchemaLoader().load_schema(engine)
    except (FileNotFoundError, ValueError):
        return frozenset()
    return frozenset(getattr(schema, sub_name, {}) or {})


def _reset_discovered_field_cache() -> None:
    """Clear the memoised discovered-schema vocab; used by tests that mutate it."""
    _discovered_field_names.cache_clear()


# =============================================================================
# Warmup Configuration
# =============================================================================


class WarmupConfig(BaseModel):
    """Warmup configuration for the measurement phase.

    Controls the warmup phase before measurement begins. Fixed mode (the
    default) runs exactly n_prompts warmup inferences. Set
    convergence_detection=True for opt-in adaptive convergence: the loop runs
    until the latency coefficient of variation falls below cv_threshold,
    governed by min_prompts (warm-start floor), max_prompts (safety cap), and
    window_size. Either mode is followed by a thermal floor wait.

    # Confidence: n_prompts=5 HIGH (DeepSpeed 5-10, Zeus 10, AIEnergyScore 10)
    # thermal_floor_seconds=60: a conservative idle-settling default (chosen, not
    # externally mandated). MLPerf Power's 60s figure is a measurement-window
    # minimum for sampling adequacy, not a thermal-settling mandate; that
    # measurement-window floor is referenced in harness/windowing.py.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable warmup phase")

    n_prompts: int = Field(
        default=5,
        ge=1,
        description="Number of full-length warmup prompts in fixed mode",
    )
    thermal_floor_seconds: float = Field(
        default=60.0,
        ge=30.0,
        description="Minimum seconds to wait after warmup before measuring (thermal stabilisation). Minimum 30s enforced.",
    )

    # CV convergence detection (opt-in adaptive mode; replaces the fixed n_prompts count)
    convergence_detection: bool = Field(
        default=False,
        description="Enable CV-based adaptive convergence (governed by min_prompts, max_prompts, cv_threshold, window_size)",
    )
    cv_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="CV target for convergence (only used when convergence_detection=True)",
    )
    max_prompts: int = Field(
        default=20,
        ge=5,
        description="Maximum warmup prompts when CV mode is on (safety cap)",
    )
    window_size: int = Field(
        default=3,
        ge=3,
        description="Sliding window size for CV calculation (3 balances responsiveness and stability)",
    )
    min_prompts: int = Field(
        default=5,
        ge=1,
        description="Minimum prompts before checking convergence (warm start)",
    )


# =============================================================================
# Baseline Configuration
# =============================================================================


class BaselineConfig(BaseModel):
    """Baseline power measurement configuration.

    Controls whether and how idle GPU power is measured before experiments,
    enabling baseline-adjusted energy attribution.

    Strategies:
        cached: Disk-persisted baseline with configurable TTL (default).
            Host measures once, writes to JSON, mounts into Docker containers.
        validated: Same as cached but periodically spot-checks for drift.
            If drift exceeds threshold, re-measures full baseline.
        fresh: Every experiment measures its own baseline. Most accurate
            but wastes ~30s per experiment.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True, description="Enable baseline power measurement")
    duration_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Baseline measurement duration in seconds",
    )
    strategy: Literal["cached", "validated", "fresh"] = Field(
        default="validated",
        description=(
            "Baseline caching strategy: 'cached' (disk-persisted TTL), "
            "'validated' (cached with periodic spot-check), "
            "'fresh' (measure every experiment)"
        ),
    )
    cache_ttl_seconds: float = Field(
        default=7200.0,
        ge=60.0,
        description=(
            "How long a cached baseline remains valid before re-measurement, in seconds. "
            "Only used with strategy='cached' or 'validated'."
        ),
    )
    validation_interval: int = Field(
        default=5,
        ge=1,
        description=(
            "Re-validate baseline every N experiments. Only used with strategy='validated'."
        ),
    )
    drift_threshold: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description=(
            "Power drift threshold (fraction) to trigger re-measurement. "
            "Only used with strategy='validated'."
        ),
    )


# =============================================================================
# Dataset Configuration
# =============================================================================


class DatasetConfig(BaseModel):
    """Dataset configuration for experiment prompts.

    source is one of:
    - Built-in alias (e.g. "aienergyscore")
    - Path to a .jsonl file
    """

    model_config = {"extra": "forbid"}

    source: str = Field(
        default="aienergyscore",
        min_length=1,
        description="Dataset source: built-in alias or .jsonl file path",
        json_schema_extra={"display_label": "Dataset Source", "role": "workload"},
    )
    n_prompts: int = Field(
        default=100,
        ge=1,
        description="Number of prompts to load or generate",
        json_schema_extra={"display_label": "Prompts", "role": "workload"},
    )
    order: Literal["interleaved", "grouped", "shuffled"] = Field(
        default="interleaved",
        description=(
            "Prompt ordering: interleaved (round-robin by source, file order), "
            "grouped (sorted by source), shuffled (seed-based random)"
        ),
    )


# =============================================================================
# Task Configuration (what to measure)
# =============================================================================


class TaskConfig(BaseModel):
    """What to measure: model identity, dataset, and workload shape.

    These fields define the scientific workload - changing any of them means
    you're measuring a fundamentally different task.
    """

    model_config = {"extra": "forbid"}

    model: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model ID or local path",
        json_schema_extra={"display_label": "Model"},
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Dataset configuration",
        json_schema_extra={"display_label": "Dataset"},
    )
    max_input_tokens: int | None = Field(
        default=256,
        ge=1,
        description=(
            "Max input token length for truncation. Keeps computation workload "
            "constant across experiments for fair comparison. None = no truncation."
        ),
        json_schema_extra={"display_label": "Max Input Tokens"},
    )
    max_output_tokens: int | None = Field(
        default=256,
        ge=1,
        description=(
            "Max output tokens (max_new_tokens for generation). "
            "None = generate until EOS or model context limit."
        ),
        json_schema_extra={"display_label": "Max Output Tokens"},
    )
    random_seed: int = Field(
        default=42,
        description="Per-experiment seed for all stochasticity: inference RNG and dataset ordering.",
    )


# =============================================================================
# Measurement Configuration (how to measure)
# =============================================================================


class MeasurementConfig(BaseModel):
    """How to measure: warmup, baseline, and energy sampling strategy.

    These fields control the measurement methodology - changing them affects
    measurement quality/accuracy but not the workload itself.
    """

    model_config = {"extra": "forbid"}

    warmup: WarmupConfig = Field(
        default_factory=WarmupConfig, description="Warmup phase configuration"
    )
    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig, description="Baseline power measurement configuration"
    )
    energy_sampler: EnergySamplerName | None = Field(
        default="auto",
        description=(
            "Energy measurement backend. "
            "auto=best available (Zeus>NVML>CodeCarbon). null disables energy measurement."
        ),
        json_schema_extra={"display_label": "Sampler"},
    )
    latency_profiling: bool = Field(
        default=False,
        description=(
            "Opt-in per-token latency profiling. Default off. When enabled, the "
            "engine captures per-token timing (transformers via a streamer forced "
            "to batch_size=1; vLLM via decode-average inter-token latency); this "
            "overhead may perturb energy and latency, so profiled runs are tagged "
            "in measurement_warnings and energy figures are emitted as-is."
        ),
        json_schema_extra={"display_label": "Latency Profiling"},
    )
    measurement_methodology: Literal["total", "windowed", "steady_state"] = Field(
        default="total",
        description=(
            "Which part of the run to measure. 'total' (default) integrates the "
            "whole run unchanged. 'windowed' measures an explicit "
            "measurement_window. 'steady_state' discards a warm-up prefix and "
            "measures the remainder (optionally auto-detecting the onset)."
        ),
        json_schema_extra={"display_label": "Measurement Window Mode"},
    )
    measurement_window: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Explicit (start_sec, end_sec) window relative to inference start, in "
            "seconds. Required when measurement_methodology='windowed'; ignored "
            "otherwise."
        ),
        json_schema_extra={"display_label": "Measurement Window"},
    )
    warmup_discard_fraction: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Fraction of the run discarded as warm-up before measuring, for "
            "measurement_methodology='steady_state'. Default 0.1 (first 10%). "
            "Ignored unless warmup_discard_seconds is None."
        ),
        json_schema_extra={"display_label": "Warm-up Discard Fraction"},
    )
    warmup_discard_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Fixed warm-up duration in seconds to discard before measuring, for "
            "measurement_methodology='steady_state'. When set, takes precedence "
            "over warmup_discard_fraction. Ignored otherwise."
        ),
        json_schema_extra={"display_label": "Warm-up Discard Seconds"},
    )
    steady_state_auto_detect: bool = Field(
        default=False,
        description=(
            "Opt-in sliding-window stability detector for "
            "measurement_methodology='steady_state'. When enabled, a "
            "coefficient-of-variation / variance-ratio test over the cleaned power "
            "series locates the steady-state onset; on failure it falls back to the "
            "fixed warm-up discard and sets steady_state_not_detected in the result."
        ),
        json_schema_extra={"display_label": "Steady-state Auto-detect"},
    )

    @model_validator(mode="after")
    def _validate_measurement_window(self) -> MeasurementConfig:
        """Cross-field checks for the measurement-window modes."""
        if self.measurement_methodology == "windowed":
            if self.measurement_window is None:
                raise ValueError(
                    "measurement_window (start_sec, end_sec) is required when "
                    "measurement_methodology='windowed'"
                )
            start, end = self.measurement_window
            if start < 0.0:
                raise ValueError("measurement_window start_sec must be >= 0")
            if end <= start:
                raise ValueError("measurement_window end_sec must be > start_sec")
        return self


# =============================================================================
# Main Experiment Configuration (v2.0)
# =============================================================================


class ExperimentConfig(BaseModel):
    """v2.0 experiment configuration.

    Central configuration object controlling all aspects of a single LLM inference
    efficiency measurement. Organised into semantic groups:

    - task: What to measure (model, dataset, token limits, seed)
    - measurement: How to measure (warmup, baseline, energy sampler)
    - Engine sections (transformers:, vllm:, tensorrt:): How to execute

    The engine section must match the engine field. Providing a transformers:
    section when engine=vllm is a configuration error.
    """

    model_config = {"extra": "forbid"}

    # Task - what to measure
    task: TaskConfig = Field(..., description="Task configuration: model, dataset, workload shape")

    # Engine selection
    engine: Engine = Field(
        default=Engine.TRANSFORMERS,
        description="Inference engine",
        json_schema_extra={"display_label": "Engine"},
    )

    # Measurement - how to measure
    measurement: MeasurementConfig = Field(
        default_factory=MeasurementConfig,
        description="Measurement methodology: warmup, baseline, energy sampling",
    )

    # Sampling preset - expands into the active engine's sampling section
    sampling_preset: SamplingPreset | None = Field(
        default=None,
        description=(
            "Sampling preset. When set, preset values are merged into the active "
            "engine's sampling section at parse time; explicit YAML values take "
            "precedence over preset values."
        ),
    )

    # Engine sections (None = use engine's own defaults)
    transformers: TransformersConfig | None = Field(
        default=None,
        description="HuggingFace Transformers engine configuration (only used when engine=transformers)",
    )
    vllm: VLLMConfig | None = Field(
        default=None,
        description="vLLM-specific configuration (only used when engine=vllm)",
    )
    tensorrt: TensorRTConfig | None = Field(
        default=None,
        description="TensorRT-LLM configuration (only used when engine=tensorrt)",
    )

    # llem-orchestration knobs that have no engine-native API (transformers-only
    # residual today: prompt-batching, torch.compile, TF32, autocast). Engine
    # config proper lives in the engine sections above.
    harness: HarnessConfig | None = Field(
        default=None,
        description="Per-engine llem-orchestration knobs (batching, torch.compile, TF32, autocast).",
    )

    # Escape hatch - explicitly declared for extra="forbid" compatibility
    passthrough_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Extra kwargs passed through to engine at execution time. "
        "Keys must not collide with ExperimentConfig top-level fields.",
    )

    # -------------------------------------------------------------------------
    # Active-engine accessors (all three engines share the generated nested shape)
    # -------------------------------------------------------------------------

    def active_engine_params(self) -> Any:
        """Return the active engine's ``engine_params`` sub-model, or None.

        One config-layer accessor for the nested shape every engine now shares,
        replacing the per-site ``cfg.<engine>.engine_params if cfg.<engine> is
        not None else None`` idiom. Returns None when the engine section is
        absent (``engine: null``).
        """
        section = getattr(self, self.engine.value, None)
        return getattr(section, "engine_params", None) if section is not None else None

    def active_sampling_params(self) -> Any:
        """Return the active engine's ``sampling_params`` sub-model, or None."""
        section = getattr(self, self.engine.value, None)
        return getattr(section, "sampling_params", None) if section is not None else None

    def active_harness(self) -> Any:
        """Return the active engine's harness block (llem-orchestration knobs), or None.

        Mirrors ``active_engine_params()`` for the hand-written orchestration
        layer. Only transformers has a harness residual today (batch_size,
        torch.compile, TF32, autocast); vllm and tensorrt drive those through
        native engine APIs, so ``harness`` carries no block for them and this
        returns None.
        """
        return getattr(self.harness, self.engine.value, None) if self.harness is not None else None

    def _harness_sourced_batch_size(self) -> int:
        """The harness ``batch_size`` knob (default 1) shared by both batch semantics.

        For harness-sourced engines (transformers) the declared capacity bound and
        the static configured batch are the same fact - the orchestration knob on
        ``harness.<engine>.batch_size`` - so both accessors defer here.
        """
        harness = self.active_harness()
        if harness is not None and harness.batch_size is not None:
            return int(harness.batch_size)
        return 1

    def capacity_batch_size(self) -> int:
        """Declared worst-case batch/concurrency bound for the active engine.

        The largest number of sequences the engine may hold at once, used to size a
        worst-case VRAM estimate. Reads the per-engine
        :class:`~llenergymeasure.config.ssot.BatchSizeModel` descriptor: the harness
        ``batch_size`` for transformers, the declared capacity field (``max_num_seqs``
        for vLLM, ``max_batch_size`` for tensorrt) otherwise. Defaults to 1 when unset.
        """
        model = ENGINES[self.engine].batch
        if model.harness_sourced:
            return self._harness_sourced_batch_size()
        if model.capacity_field is not None:
            params = self.active_engine_params()
            if params is not None:
                value = getattr(params, model.capacity_field, None)
                if value is not None:
                    return int(value)
        return 1

    def static_batch_size(self) -> int | None:
        """Fixed configured batch size for the active engine, or None.

        The single static batch size to report on the measured result. Continuous-
        batching engines (vLLM) have no static batch and return None - the effective
        batch there is derived from the realised prompt/batch counts instead. Reads
        the per-engine :class:`~llenergymeasure.config.ssot.BatchSizeModel` descriptor.
        """
        model = ENGINES[self.engine].batch
        if model.harness_sourced:
            return self._harness_sourced_batch_size()
        if model.static_field is not None:
            params = self.active_engine_params()
            if params is not None:
                value = getattr(params, model.static_field, None)
                return int(value) if value is not None else None
        return None

    def engine_sub_dict(self, name: str) -> dict[str, Any] | None:
        """Return a non-empty ``engine_params`` sub-config dict by name, or None.

        The curated discovery-debt containers (vllm ``attention`` /
        ``beam_search``, tensorrt ``quant_config`` / ``kv_cache_config`` /
        ``scheduler_config``) are Any-typed on the current pins, so they arrive
        as plain dicts; this is the shared accessor the engine plugins read them
        through.
        """
        engine_params = self.active_engine_params()
        value = getattr(engine_params, name, None) if engine_params is not None else None
        return value if isinstance(value, dict) and value else None

    # -------------------------------------------------------------------------
    # Pre-validators (run before field parsing)
    # -------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def expand_sampling_preset(cls, data: Any) -> Any:
        """Merge ``sampling_preset`` values into the active engine's sampling section.

        Explicit YAML values take precedence over preset values (each preset key
        is applied via ``setdefault``). The preset name itself stays on the
        top-level model so it remains inspectable after parsing.
        """
        if not isinstance(data, dict):
            return data
        preset_name = data.get("sampling_preset")
        if not preset_name or preset_name not in SAMPLING_PRESETS:
            return data
        engine = data.get("engine", Engine.TRANSFORMERS)
        engine_key = engine_str(engine)
        # Ensure the engine section and its sampling_params sub-dict exist as
        # dicts (an explicit ``engine: null`` in YAML would otherwise leave None
        # here). All three engines use the generated nested shape.
        engine_section = data.get(engine_key)
        if not isinstance(engine_section, dict):
            engine_section = {}
            data[engine_key] = engine_section
        sampling_section = engine_section.get("sampling_params")
        if not isinstance(sampling_section, dict):
            sampling_section = {}
            engine_section["sampling_params"] = sampling_section
        for key, value in SAMPLING_PRESETS[preset_name].items():
            sampling_section.setdefault(key, value)
        return data

    # -------------------------------------------------------------------------
    # Cross-validators
    # -------------------------------------------------------------------------

    _FLASH_ATTENTION_IMPLS: ClassVar[set[str]] = {"flash_attention_2", "flash_attention_3"}

    @model_validator(mode="after")
    def validate_engine_section_match(self) -> ExperimentConfig:
        """Engine section must match the engine field.

        A transformers: section with engine=vllm is a configuration error - it indicates
        the researcher copied the wrong config block. Fail explicitly rather than
        silently ignoring the mismatched section.
        """
        for engine in ALL_ENGINES:
            if getattr(self, engine) is not None and self.engine != engine:
                raise ValueError(
                    f"{engine}: config section provided but engine={self.engine!r}. "
                    f"Remove the {engine}: section or set engine: {engine}."
                )
        return self

    @model_validator(mode="after")
    def validate_engine_section_extras(self) -> ExperimentConfig:
        """Soft/hard validation of extra keys on the active engine's nested section.

        Two checks against the generated nested shape (``engine_params`` /
        ``sampling_params`` sub-models):

        - Wrapper-level extras (keys on the section itself) are ERRORS: the
          engine never sees them, so silently accepting them would let a user
          measure something other than what they configured. If a key names a
          known nested field it is a pre-nested-shape flat config, so the
          message is a migration hint pointing at the correct nested location;
          otherwise it is a typo and the message carries a ``did you mean``
          suggestion.
        - Extras inside ``engine_params`` / ``sampling_params`` DO pass through
          to the engine (``extra="allow"``), so a genuinely-new engine field is
          legitimate. We only WARN, and only when the key is a close typo of a
          known field, using the discovered schema plus generated fields as the
          vocabulary.
        """
        section = getattr(self, self.engine.value, None)
        if not isinstance(section, BaseModel):
            return self
        sub_models = _nested_subsection_models(section)
        if not sub_models:
            return self

        nested_field_names = {name for model in sub_models.values() for name in model.model_fields}

        # (1) Wrapper-level extras: always an error.
        for key in section.model_extra or {}:
            for sub_name, model in sub_models.items():
                if key in model.model_fields:
                    raise ValueError(
                        f"{self.engine.value}.{key} moved to "
                        f"{self.engine.value}.{sub_name}.{key} (flat engine config was "
                        "replaced by the nested shape in v0.10)."
                    )
            suggestion = difflib.get_close_matches(key, sorted(nested_field_names), n=1)
            hint = f"; did you mean engine_params.{suggestion[0]}?" if suggestion else ""
            raise ValueError(f"unknown field {key!r} on {self.engine.value}{hint}")

        # (2) Sub-section extras: pass through to the engine - warn on close typos.
        for sub_name, model in sub_models.items():
            sub_section = getattr(section, sub_name, None)
            if sub_section is None:
                continue
            extras = sub_section.model_extra
            if not extras:
                # No extra keys to vet: skip building the vocabulary (which
                # reads the discovered-schema JSON off disk). The common case.
                continue
            vocabulary = sorted(
                set(model.model_fields) | _discovered_field_names(self.engine.value, sub_name)
            )
            for key in extras:
                if key in vocabulary:
                    # Discovered-but-uncurated engine field: legitimate
                    # passthrough, no warning.
                    continue
                suggestion = difflib.get_close_matches(
                    key, vocabulary, n=1, cutoff=_CLOSE_MATCH_CUTOFF
                )
                if suggestion:
                    warnings.warn(
                        f"unknown field {key!r} in {self.engine.value}.{sub_name}; "
                        f"did you mean {suggestion[0]}?",
                        ConfigValidationWarning,
                        stacklevel=2,
                    )
        return self

    @model_validator(mode="after")
    def validate_passthrough_kwargs_no_collision(self) -> ExperimentConfig:
        """passthrough_kwargs keys must not collide with ExperimentConfig fields.

        If a researcher writes passthrough_kwargs: {model: gpt2}, they intended to
        set the model field directly. Collisions are always a misconfiguration.
        """
        if self.passthrough_kwargs:
            top_level_fields = set(ExperimentConfig.model_fields.keys())
            collisions = set(self.passthrough_kwargs.keys()) & top_level_fields
            if collisions:
                raise ValueError(
                    f"passthrough_kwargs keys collide with ExperimentConfig fields: "
                    f"{sorted(collisions)}. Use the named fields instead."
                )
        return self

    # The old hand-written vLLM/TRT-LLM dtype Literals rejected float32 at parse
    # time; that rejection was over-narrow and was dropped with the generated
    # configs. vLLM's ModelDType genuinely accepts float32, so the generated vLLM
    # dtype enum (now the authoritative source) includes it; the tensorrt dtype is
    # un-narrowed (plain str). float32 is a valid-but-rarely-useful choice on these
    # engines for inference, not an error, so no hand-written rule replaces them.

    @model_validator(mode="after")
    def validate_transformers_flash_attn_dtype(self) -> ExperimentConfig:
        """FlashAttention (FA2/FA3) requires float16 or bfloat16 dtype (not float32).

        Retained as a hand-written validator until a ``PreTrainedModel``
        introspection miner can derive this rule programmatically (the check
        lives in ``_autoset_attn_implementation``, not in
        ``GenerationConfig.validate``).
        """
        ep = self.active_engine_params() if self.engine == "transformers" else None
        if (
            ep is not None
            and ep.attn_implementation in self._FLASH_ATTENTION_IMPLS
            and (ep.dtype or "bfloat16") == "float32"
        ):
            raise ValueError(
                f"attn_implementation='{ep.attn_implementation}' requires "
                "dtype='float16' or dtype='bfloat16'. FlashAttention does not support "
                "float32 computation."
            )
        return self

    @model_validator(mode="after")
    def validate_tensorrt_engine_path_backend(self) -> ExperimentConfig:
        """A prebuilt TensorRT engine_path requires backend='trt'.

        ``engine_path`` points the TRT-LLM loader at a directory of compiled
        ``rank*.engine`` files (a prebuilt TensorRT engine). Only the trt
        constructor (``tensorrt_llm._tensorrt_engine.LLM``) reads that format;
        the pytorch backend (``tensorrt_llm.LLM``) would misread the directory
        as a HuggingFace checkpoint and silently construct the wrong flow.

        ``backend`` defaults to 'pytorch' when unset, so an engine_path with no
        explicit backend is rejected too: we never flip the constructor class
        silently based on a passthrough field. This is a hand-written cross-field
        constraint on the curated surface because ``engine_path`` is an
        extra='allow' passthrough (not a mined field) and the constraint is a
        llem-orchestration semantic (which constructor to use), not an
        engine-native validation the rules corpus mines. Mirrors the sibling
        backend-applicability guards for fast_build / quant_config.
        """
        if self.engine != "tensorrt":
            return self
        ep = self.active_engine_params()
        if ep is None:
            return self
        engine_path = getattr(ep, "engine_path", None)
        backend = getattr(ep, "backend", None)
        if engine_path is not None and backend != "trt":
            raise ValueError(
                f"engine_path requires backend='trt' (got backend={backend!r}). "
                "engine_path loads a prebuilt compiled-TensorRT engine directory, "
                "which only the trt constructor can read; the pytorch backend "
                "would misinterpret it as a checkpoint. Set backend: trt, or drop "
                "engine_path to build from the model checkpoint."
            )
        return self

    @model_validator(mode="after")
    def _apply_rules(self) -> ExperimentConfig:
        # ``object.__setattr__`` bypasses Pydantic's ``extra='forbid'``;
        # consumers read via ``cfg._dormant_observations`` (dict keyed by
        # rule.id). Missing corpus is non-fatal - the rules layer is additive.
        from llenergymeasure.config.probe import DormantField

        dormant_observations: dict[str, DormantField] = {}
        try:
            rules = _get_rules_loader().load_rules(self.engine.value).rules
        except FileNotFoundError:
            logger.debug("No rules corpus for engine %r; skipping.", self.engine.value)
            rules = ()

        for rule in rules:
            match = rule.try_match(self)
            if match is None:
                continue
            annotated = f"[{rule.id}] {rule.render_message(match)}"
            if rule.severity == "error":
                raise ValueError(annotated)
            if rule.severity == "dormant":
                dormant_observations[rule.id] = DormantField(
                    declared_value=match.declared_value,
                    effective_value=match.effective_value,
                    reason=annotated,
                )

        object.__setattr__(self, "_dormant_observations", dormant_observations)
        return self


# Rebuild to resolve forward references for engine configs
def _rebuild_experiment_config() -> None:
    """Rebuild ExperimentConfig to resolve forward references.

    All three engines resolve to their generated nested ``Config``
    (engine_params / sampling_params shape, projected from the mined schema).
    """
    from llenergymeasure.config.generated.tensorrt import Config as TensorRTConfig
    from llenergymeasure.config.generated.transformers import Config as TransformersConfig
    from llenergymeasure.config.generated.vllm import Config as VLLMConfig

    ExperimentConfig.model_rebuild(
        _types_namespace={
            "VLLMConfig": VLLMConfig,
            "TransformersConfig": TransformersConfig,
            "TensorRTConfig": TensorRTConfig,
        }
    )


_rebuild_experiment_config()


class OutputConfig(BaseModel):
    """Study-level output configuration.

    Controls where results are written and what auxiliary artefacts are persisted.
    Lives on StudyConfig only - experiments don't own output config because output
    is an operational concern, not part of the scientific specification.

    Resolution chain (highest wins):
        study YAML output.results_dir > user_config.output.results_dir > "./results"
    """

    model_config = {"extra": "forbid"}

    results_dir: str | None = Field(
        default=None,
        description=(
            "Base directory for study results. A timestamped study subdirectory "
            "is created within this path. Resolved identically for local and "
            "Docker runs (Docker results are always written back to host). "
            "None = defer to user config or built-in default (./results)."
        ),
    )
    save_timeseries: bool = Field(
        default=True,
        description=(
            "Persist GPU power/thermal/memory timeseries as Parquet sidecar. "
            "NVML telemetry is always collected for throttle detection; this "
            "controls whether the full timeseries is written to disk."
        ),
    )


class ExecutionConfig(BaseModel):
    """Execution controls for a study (cycle repetition, ordering, gaps).

    Controls how many times the experiment list is repeated (n_cycles), the order
    in which experiments are executed across cycles (experiment_order), optional gaps
    between configs and cycles for thermal stabilisation, and an explicit shuffle
    seed override (default: derived from study_design_hash for reproducibility).

    Pydantic defaults are conservative (1 cycle, sequential, no gaps). The CLI
    will apply research-appropriate effective defaults (e.g. 3 cycles, shuffle).
    """

    model_config = {"extra": "forbid"}

    n_cycles: int = Field(
        default=1, ge=1, description="Number of times to repeat the experiment list"
    )
    experiment_order: Literal["sequential", "interleave", "shuffle", "reverse", "latin_square"] = (
        Field(
            default="sequential",
            description=(
                "Ordering strategy across cycles. "
                "sequential: [A,A,A,B,B,B], interleave: [A,B,A,B,A,B], "
                "shuffle: random per-cycle order."
            ),
        )
    )
    experiment_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Seconds to wait between individual experiments. "
            "None = use machine default from user config."
        ),
    )
    cycle_gap_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Longer thermal-equalisation pause at cycle boundaries (only when n_cycles >= 2). "
            "For sequential order it fires between per-config repetition blocks ([A,A,A|B,B,B]); "
            "for interleave/reverse/shuffle/latin_square it fires between full passes over the "
            "configs. None = use machine default from user config."
        ),
    )
    shuffle_seed: int | None = Field(
        default=None,
        description=(
            "Explicit seed for shuffle experiment_order. "
            "None = derived from study_design_hash (same study always shuffles identically)."
        ),
    )
    skip_preflight: bool = Field(
        default=False,
        description=(
            "Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility). "
            "Useful for remote Docker daemon setups or CI environments. "
            "The --skip-preflight CLI flag always overrides this setting."
        ),
    )
    max_consecutive_failures: int = Field(
        default=10,
        ge=0,
        description=(
            "Circuit breaker threshold: abort after N consecutive failures. "
            "0 = disabled. 1 = fail-fast (no cooldown)."
        ),
    )
    circuit_breaker_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Cooldown pause before half-open probe experiment.",
    )
    wall_clock_timeout_hours: float | None = Field(
        default=None,
        gt=0.0,
        description="Study wall-clock timeout in hours. null = no limit.",
    )
    experiment_timeout_seconds: float = Field(
        default=600.0,
        gt=0.0,
        description=(
            "Per-experiment wall-clock timeout in seconds. Applies to both the "
            "local subprocess path and the Docker container path. Experiments "
            "that exceed this budget are killed and recorded as TimeoutError; "
            "the circuit breaker counts them toward max_consecutive_failures."
        ),
    )
    stdout_silence_timeout_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description=(
            "Maximum stdout/stderr-silent stretch tolerated by the Docker watchdog "
            "before the container is killed. Catches hangs that wouldn't trip "
            "experiment_timeout_seconds (stuck CUDA kernels, deadlocked NCCL, "
            "frozen compilation). Set 0 to disable. Raise (e.g. 600-900s) for "
            "fresh TRT-LLM engine builds with infrequent compile progress lines."
        ),
    )
    deduplicate_equivalent: bool = Field(
        default=True,
        description=(
            "When true (default), sweep expansion applies effective-config resolution to each "
            "declared ExperimentConfig via engine-rules dormant-rule application and drops "
            "duplicates that share a resolved_config_hash. When false, every declared "
            "config runs - effective-config resolution still populates equivalence-group "
            "metadata for the sidecar but no configs are elided. The --no-dedup "
            "CLI flag is the equivalent."
        ),
    )
    gpu_indices: list[int] | None = Field(
        default=None,
        description=(
            "HOST GPU indices (as `nvidia-smi` shows) to scope llem's Docker containers to "
            "via `docker run --gpus device=<indices>`, scoped at the docker level so "
            "in-container CUDA and NVML indices stay consistent (see the docker-setup docs). "
            "null = `--gpus all` (every visible GPU, the historical default). This is "
            "placement/dispatch metadata, NOT part of the declared-config or study-design "
            "hash, so pinning a study to different physical GPUs never changes dedup grouping. "
            "The LLEM_DOCKER_GPUS env var overrides this (env>config); when both are set the "
            "env wins and a warning is logged."
        ),
    )

    @model_validator(mode="after")
    def _validate_gpu_indices(self) -> ExecutionConfig:
        """Reject empty, negative, or duplicate GPU indices (fail loudly).

        Absence is expressed as ``None`` (``--gpus all``); an empty list is a
        mistake, not "all". Negative indices and duplicates cannot name real
        distinct host devices.
        """
        if self.gpu_indices is None:
            return self
        if not self.gpu_indices:
            raise ValueError(
                "study_execution.gpu_indices must not be empty; omit it (null) to use all GPUs."
            )
        if any(i < 0 for i in self.gpu_indices):
            raise ValueError(
                f"study_execution.gpu_indices must be non-negative host device indices, "
                f"got {self.gpu_indices}."
            )
        if len(set(self.gpu_indices)) != len(self.gpu_indices):
            raise ValueError(
                f"study_execution.gpu_indices must not contain duplicates, got {self.gpu_indices}."
            )
        return self


class StudyConfig(BaseModel):
    """Thin resolved container for a study (list of experiments + execution config).

    Populated by the study loader after sweep expansion. The experiments list
    contains fully-validated ExperimentConfig objects ready for execution.
    skipped_configs records any grid points that failed Pydantic validation so
    they can be displayed to the researcher in pre-flight output.
    """

    model_config = {"extra": "forbid"}

    experiments: list[ExperimentConfig] = Field(
        ..., min_length=1, description="Resolved list of experiments to run"
    )
    study_name: str | None = Field(
        default=None, description="Study name (used in output directory naming)"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Study-level output configuration (results_dir, save_timeseries)",
    )
    study_execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Cycle repetition and ordering controls",
    )
    runners: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-engine runner configuration. Keys are engine names "
            "('transformers', 'vllm', 'tensorrt'), values are runner strings "
            "('process', 'container', or 'container:<image>'; the legacy 'local'/'docker' "
            "vocabulary was renamed in v0.7 and is now rejected with a migration error). "
            "None = use user config / auto-detection. "
            "Runner is metadata - not part of the experiment config hash."
        ),
    )
    images: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-engine Docker image overrides (orthogonal to runners). "
            "Keys are engine names, values are image references "
            "(e.g. 'ghcr.io/org/img:tag'). None = use smart default "
            "(local build → registry fallback). "
            "Image is metadata - not part of the experiment config hash."
        ),
    )
    study_design_hash: str | None = Field(
        default=None,
        description=(
            "16-char SHA-256 hex of the resolved experiment list (execution block excluded). "
            "Set by the loader after expansion; None before expansion."
        ),
    )
    skipped_configs: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Grid points that failed Pydantic validation during expansion. "
            "Persisted for post-hoc review and pre-flight display."
        ),
    )
    dedup_mode: Literal["resolved", "off"] = Field(
        default="resolved",
        description=(
            "Effective-config resolution dedup mode. 'resolved' applies dormant-rule "
            "effective-config resolution at expansion and collapses "
            "resolved-config-hash-equivalent configs to a single run. 'off' runs every "
            "declared config regardless of equivalence. Set via "
            "ExecutionConfig.deduplicate_equivalent / --no-dedup."
        ),
    )
    pre_run_equivalence_groups: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Pre-run equivalence groups computed at sweep-expansion time. "
            "Each group records the resolved_config_hash, canonical excerpt, and member "
            "declared-indices. Written to 'equivalence_groups.json' alongside "
            "the results bundle."
        ),
    )
    declared_resolved_config_hashes: list[str] = Field(
        default_factory=list,
        description=(
            "Per-declared-config resolved_config_hashes (parallel to the pre-resolved "
            "sweep input). Harness consults this to tag each experiment with "
            "its equivalence group at sidecar-write time."
        ),
    )
    dormant_observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Distinct auto-normalised settings applied during effective-config resolution "
            "(keys: engine, rule_id, field_path, normalisation). These are fields the "
            "engine silently rewrites in the executed config, so they are surfaced in "
            "'llem study plan' and preflight output. Empty when nothing was normalised."
        ),
    )
