"""Single source of truth for project-wide constants.

Centralises engine capabilities, runner modes, environment variable names,
temp file prefixes, and infrastructure timeout values. Consumers include:
- ExperimentConfig cross-validators (structural validation)
- config/introspection.py (engine capability metadata)
- CLI help generation
- Infrastructure modules (Docker runner, runner resolution, image registry)
- Study runner and container lifecycle management

Do not inline these values in validators or infrastructure code - always
import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Literal

# ---------------------------------------------------------------------------
# Sampling presets (shared across engines)
# ---------------------------------------------------------------------------
# Preset values are aligned with industry conventions (vLLM, OpenAI, MLPerf).
# do_sample is intentionally absent from presets - it's HF-specific; the
# transformers engine builder infers do_sample=False from temperature=0.

SAMPLING_PRESETS: dict[str, dict[str, Any]] = {
    "deterministic": {"temperature": 0.0},
    "standard": {"temperature": 1.0, "top_p": 0.95},
    "creative": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1},
    "factual": {"temperature": 0.3},
}

# ---------------------------------------------------------------------------
# Engine enum - the ONE source of truth for backend engine identifiers.
# Add a new backend by adding a member here; everything else derives from it
# (Pydantic validation, membership checks, ordered iteration, CLI choices).
# ---------------------------------------------------------------------------


class Engine(str, Enum):
    """Supported inference backends.

    Uses (str, Enum) so members compare and hash equal to their string values
    on Python 3.10+. The __str__ override ensures str(Engine.X) returns the
    raw value (e.g. "transformers"), not "Engine.TRANSFORMERS".
    StrEnum (3.11+) is intentionally avoided to keep requires-python = ">=3.10".
    """

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TENSORRT = "tensorrt"

    def __str__(self) -> str:
        return self.value


ALL_ENGINES: Final[frozenset[Engine]] = frozenset(Engine)
"""Unordered engine set - use for O(1) membership checks (``engine in ALL_ENGINES``)."""


def engine_str(engine: Any) -> str:
    """Coerce an Engine enum (or plain string) to its string value."""
    return engine.value if hasattr(engine, "value") else str(engine)


# ---------------------------------------------------------------------------
# Runner mode constants
# ---------------------------------------------------------------------------

RUNNER_LOCAL: Final = "local"
RUNNER_DOCKER: Final = "docker"
CONTAINER_EXCHANGE_DIR: Final = "/run/llem"
"""Mount point inside Docker containers for config/result exchange."""

# RunnerSpec.source tags - which layer of the precedence chain produced a runner.
SOURCE_ENV: Final = "env"
SOURCE_YAML: Final = "yaml"
SOURCE_USER_CONFIG: Final = "user_config"
SOURCE_AUTO_DETECTED: Final = "auto_detected"
SOURCE_DEFAULT: Final = "default"
SOURCE_MULTI_ENGINE_ELEVATION: Final = "multi_engine_elevation"
"""RunnerSpec source tag when an engine is auto-elevated to Docker for multi-engine isolation."""

EXPLICIT_RUNNER_SOURCES: Final[frozenset[str]] = frozenset(
    {SOURCE_ENV, SOURCE_YAML, SOURCE_USER_CONFIG}
)
"""Runner source tags that represent an explicit user pin (env var, study YAML, or user
config). In a multi-engine study these win over Docker elevation; only auto-resolved
runners (``auto_detected`` / ``default``) are elevated to Docker for isolation."""

RunnerMode = Literal["local", "docker"]

DOCKER_PULL_TIMEOUT: Final = 1800
"""Maximum seconds to wait for ``docker pull`` (30 min - generous for large images like TensorRT ~10 GB)."""

# ---------------------------------------------------------------------------
# Environment variable name constants
# ---------------------------------------------------------------------------

ENV_RUNNER_PREFIX: Final = "LLEM_RUNNER_"
"""Prefix for per-engine runner override env vars (e.g. ``LLEM_RUNNER_TRANSFORMERS=docker``)."""

ENV_IMAGE_PREFIX: Final = "LLEM_IMAGE_"
"""Prefix for per-engine image override env vars (e.g. ``LLEM_IMAGE_VLLM=custom:tag``)."""

ENV_CARBON_INTENSITY: Final = "LLEM_CARBON_INTENSITY"
ENV_DATACENTER_PUE: Final = "LLEM_DATACENTER_PUE"
ENV_NO_PROMPT: Final = "LLEM_NO_PROMPT"
ENV_HF_TOKEN: Final = "HF_TOKEN"
ENV_OUTPUT_DIR: Final = "LLEM_OUTPUT_DIR"
ENV_SAVE_TIMESERIES: Final = "LLEM_SAVE_TIMESERIES"
ENV_CONFIG_PATH: Final = "LLEM_CONFIG_PATH"
ENV_BASELINE_SPEC_PATH: Final = "LLEM_BASELINE_SPEC_PATH"
"""Path inside a baseline container where the entrypoint reads its spec JSON."""
ENV_LOG_LEVEL: Final = "LLEM_LOG_LEVEL"
ENV_TABLE_ROWS: Final = "LLEM_TABLE_ROWS"

# Engine dispatch (container_entrypoint.sh + docker_runner.py).
ENV_ENGINE: Final = "LLEM_ENGINE"
"""Engine value (transformers/vllm/tensorrt) read by the container entrypoint
script so it can route tensorrt dispatches through nvidia_entrypoint.sh."""
ENV_HOST_UID: Final = "LLEM_HOST_UID"
"""Host user UID, passed in so the in-container entrypoint can chown the
deps cache after priming (container runs as root by default, so without
this the host can't clean the cache without sudo)."""
ENV_HOST_GID: Final = "LLEM_HOST_GID"
"""Host user GID; paired with ENV_HOST_UID for chown."""
ENV_DEPS_CACHE_DIR: Final = "LLEM_DEPS_CACHE_DIR"
"""Override for the host-side runtime-deps cache directory. Defaults to
``platformdirs.user_cache_dir('llem')/deps`` when unset."""
ENV_ENTRY_MODULE: Final = "LLEM_ENTRY_MODULE"
"""Framework module the container entrypoint script exec's with ``python3 -m``.
Defaults to ``llenergymeasure.entrypoints.container`` (the experiment path);
the baseline dispatch sets it to ``llenergymeasure.entrypoints.baseline_measure``
so both share the same package-mount + dep-prime bootstrap."""

# ---------------------------------------------------------------------------
# Temp file/directory prefixes
# ---------------------------------------------------------------------------

TEMP_PREFIX_EXCHANGE: Final = "llem-"
"""Prefix for exchange directory created by DockerRunner."""

TEMP_PREFIX_ENV_FILE: Final = "llem-env"
"""Prefix for env-file temp files used to pass secrets to Docker."""

TEMP_PREFIX_TIMESERIES: Final = "llem-ts-"
"""Prefix for temp directories holding timeseries parquet files."""

STAGE_LINE_PREFIX: Final = "[llem.baseline]"
"""Wire-protocol line prefix for baseline stage markers: emitted by
entrypoints/baseline_measure._emit_stage and parsed by study/baseline_container."""

# ---------------------------------------------------------------------------
# Subprocess / thread timeout constants (seconds)
# ---------------------------------------------------------------------------

# Docker CLI subprocess timeouts
TIMEOUT_DOCKER_CLI: Final = 5
"""Quick Docker CLI calls: ``docker ps``, ``docker image inspect`` (cache check)."""

TIMEOUT_DOCKER_INSPECT: Final = 10
"""``docker image inspect`` in ensure_image / study runner image preparation."""

TIMEOUT_DOCKER_STOP: Final = 10
"""``docker stop`` graceful shutdown."""

# NVIDIA tool subprocess timeouts
TIMEOUT_NVIDIA_SMI: Final = 10
"""``nvidia-smi`` query subprocess."""

# Background task timeouts
TIMEOUT_ENV_SNAPSHOT: Final = 10
"""Environment snapshot collection future."""

# Thread / process lifecycle timeouts
TIMEOUT_THREAD_JOIN: Final = 5
"""Thread joins, process teardown."""

TIMEOUT_SIGTERM_GRACE: Final = 2
"""Grace period after SIGTERM before SIGKILL."""

TIMEOUT_INTERRUPT_POLL: Final = 1
"""Interrupt event wait loop tick."""

# ---------------------------------------------------------------------------
# Per-engine descriptor registry
# ---------------------------------------------------------------------------
# ENGINES is the single source of truth for per-engine facts. Every fact an
# engine carries (identity package, availability probe, plugin location,
# supported dtypes, parallelism model, default-image version source) lives in
# one shape here, so a backend is described in exactly one place. Consumers
# read the descriptor directly, or take a narrow derived view (ENGINE_PACKAGES).


@dataclass(frozen=True)
class ParallelismModel:
    """How an engine's config fields map to the set of GPUs to monitor.

    ``multiply_fields`` names the active engine-params attributes whose product
    (each defaulting to 1 when unset) gives the GPU count - e.g. vLLM multiplies
    tensor- by pipeline-parallel size. ``all_visible_field``, when set, names an
    attribute that (when non-None) means the engine shards across every
    NVML-visible GPU - e.g. transformers ``device_map``. The two are mutually
    exclusive; an engine with neither always monitors a single GPU.
    """

    multiply_fields: tuple[str, ...] = ()
    all_visible_field: str | None = None


@dataclass(frozen=True)
class EngineDescriptor:
    """Consolidated per-engine facts (see ``ENGINES``)."""

    package: str
    """Importable package that identifies the engine (version + preflight)."""

    availability_probe: str
    """Package imported to test whether the engine can run. Transformers probes
    ``torch`` (its GPU stack), not the always-present ``transformers`` package."""

    plugin_module: str
    """Module holding the engine's ``EnginePlugin`` implementation."""

    plugin_class: str
    """Class name of the engine's ``EnginePlugin`` implementation."""

    dtypes: tuple[str, ...]
    """Precision modes the engine supports ("float32" = full, "float16" = half,
    "bfloat16" = brain float16). fp16/bf16 require GPU; GPU detection and
    cpu-dtype cross-validation happen at pre-flight."""

    parallelism: ParallelismModel
    """How the engine's config maps to the set of GPUs to monitor."""

    image_version_source: Literal["package", "engine"]
    """Which version tags the engine's default image: the llenergymeasure
    ``package`` version (first-party GHCR image) or the pinned ``engine`` version
    (upstream image)."""


ENGINES: dict[Engine, EngineDescriptor] = {
    Engine.TRANSFORMERS: EngineDescriptor(
        package="transformers",
        availability_probe="torch",
        plugin_module="llenergymeasure.engines.transformers",
        plugin_class="TransformersEngine",
        dtypes=("float32", "float16", "bfloat16"),
        parallelism=ParallelismModel(all_visible_field="device_map"),
        image_version_source="package",
    ),
    Engine.VLLM: EngineDescriptor(
        package="vllm",
        availability_probe="vllm",
        plugin_module="llenergymeasure.engines.vllm",
        plugin_class="VLLMEngine",
        dtypes=("float16", "bfloat16"),  # vLLM does not support fp32 inference
        parallelism=ParallelismModel(
            multiply_fields=("tensor_parallel_size", "pipeline_parallel_size")
        ),
        image_version_source="engine",
    ),
    Engine.TENSORRT: EngineDescriptor(
        package="tensorrt_llm",
        availability_probe="tensorrt_llm",
        plugin_module="llenergymeasure.engines.tensorrt",
        plugin_class="TensorRTEngine",
        dtypes=("float16", "bfloat16"),  # TRT-LLM does not support fp32 inference
        parallelism=ParallelismModel(multiply_fields=("tensor_parallel_size",)),
        image_version_source="engine",
    ),
}

# Map each engine to the importable package that identifies it - a narrow view
# derived from ENGINES. Used by preflight checks, health reporting, and the
# version handshake.
ENGINE_PACKAGES: dict[Engine, str] = {
    engine: descriptor.package for engine, descriptor in ENGINES.items()
}

__all__ = [
    "ALL_ENGINES",
    "CONTAINER_EXCHANGE_DIR",
    "DOCKER_PULL_TIMEOUT",
    "ENGINES",
    "ENGINE_PACKAGES",
    "ENV_BASELINE_SPEC_PATH",
    "ENV_CARBON_INTENSITY",
    "ENV_CONFIG_PATH",
    "ENV_DATACENTER_PUE",
    "ENV_DEPS_CACHE_DIR",
    "ENV_ENGINE",
    "ENV_HF_TOKEN",
    "ENV_HOST_GID",
    "ENV_HOST_UID",
    "ENV_IMAGE_PREFIX",
    "ENV_LOG_LEVEL",
    "ENV_NO_PROMPT",
    "ENV_OUTPUT_DIR",
    "ENV_RUNNER_PREFIX",
    "ENV_SAVE_TIMESERIES",
    "ENV_TABLE_ROWS",
    "EXPLICIT_RUNNER_SOURCES",
    "RUNNER_DOCKER",
    "RUNNER_LOCAL",
    "SAMPLING_PRESETS",
    "SOURCE_AUTO_DETECTED",
    "SOURCE_DEFAULT",
    "SOURCE_ENV",
    "SOURCE_MULTI_ENGINE_ELEVATION",
    "SOURCE_USER_CONFIG",
    "SOURCE_YAML",
    "STAGE_LINE_PREFIX",
    "TEMP_PREFIX_ENV_FILE",
    "TEMP_PREFIX_EXCHANGE",
    "TEMP_PREFIX_TIMESERIES",
    "TIMEOUT_DOCKER_CLI",
    "TIMEOUT_DOCKER_INSPECT",
    "TIMEOUT_DOCKER_STOP",
    "TIMEOUT_ENV_SNAPSHOT",
    "TIMEOUT_INTERRUPT_POLL",
    "TIMEOUT_NVIDIA_SMI",
    "TIMEOUT_SIGTERM_GRACE",
    "TIMEOUT_THREAD_JOIN",
    "Engine",
    "EngineDescriptor",
    "ParallelismModel",
    "RunnerMode",
]
