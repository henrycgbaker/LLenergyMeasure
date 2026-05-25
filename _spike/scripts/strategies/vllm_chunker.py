"""vLLM source extraction + chunking for LLM strategies.

Mirrors ``transformers_chunker.py``'s pattern for vllm 0.7.3.

Source acquisition
------------------
vllm is NOT installed in the project venv (Phase 2.5 cannot install CUDA
extensions cleanly). Source is read directly from a pre-unpacked wheel
at ``/tmp/vllm-unpacked/vllm/`` (already present from Phase 1 Day 1's
miner-lift work).

Phase 3 should generalise to a venv path passed in by trial_runner so
the matrix runner can target bumped vllm versions. For Phase 2.5 the
hard-coded path is the unblocker.

Chunkable units (per Phase 2.5 spec)
------------------------------------
Per-class chunks:
- SamplingParams (sampling_params.py): the user-facing request-time
  validation. GuidedDecodingParams is inlined as a companion (it's
  referenced by SamplingParams).
- EngineArgs (engine/arg_utils.py): the CLI dataclass; the dataclass
  definition is huge (~125 fields) so we split into the dataclass body
  and the __post_init__ separately.
- ModelConfig (config.py): per-method _verify_tokenizer_mode,
  _verify_quantization, _verify_cuda_graph, _verify_bnb_config.
- CacheConfig (config.py): _verify_args, _verify_cache_dtype,
  _verify_prefix_caching.
- SchedulerConfig (config.py): __post_init__ + _verify_args.
- ParallelConfig (config.py): __post_init__ + _verify_args.
- LoRAConfig (config.py): __post_init__.
- TokenizerPoolConfig (config.py): __post_init__.
- PromptAdapterConfig (config.py): __post_init__.

Companion classes inlined per chunk:
- GuidedDecodingParams in SamplingParams chunk.
- (vllm doesn't have a deep BitsAndBytesConfig equivalent; BNB is
  wrapped per-class via _verify_bnb_config method.)

Per-method chunks (validators):
- SamplingParams._verify_args + _verify_greedy_sampling (combined; 60 lines).
- Each _verify_* method on ModelConfig/CacheConfig is its own chunk
  (validators are 5-30 lines each; well within budget).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Source path config
# ---------------------------------------------------------------------------

VLLM_SOURCE_ROOT = Path("/tmp/vllm-unpacked/vllm")
"""Pre-unpacked vllm 0.7.3 source tree (from Phase 1 Day 1 miner-lift)."""


# ---------------------------------------------------------------------------
# ChunkSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpec:
    """One LLM-extraction chunk. Same shape as transformers_chunker's spec
    for unified handling in llm_extractor.

    Note: kept independent from transformers_chunker.ChunkSpec rather than
    imported. Two reasons: (a) future engine chunkers may want different
    fields; (b) Python's frozen-dataclass equality only works across the
    same class.
    """

    name: str
    description: str
    source: str
    expected_namespaces: list[str]
    expected_kind: str = "schema"


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------


def _read_source(rel_path: str) -> str:
    """Read a file under VLLM_SOURCE_ROOT."""
    p = VLLM_SOURCE_ROOT / rel_path
    if not p.exists():
        return ""
    return p.read_text()


def _extract_class_body(source: str, class_name: str, max_lines: int = 400) -> str:
    """Return the source for `class <class_name>:` up to (but not including)
    the next top-level class/def definition or end of file.

    Stops at the next non-indented `class ` or `def ` (heuristic).
    Caps at ``max_lines`` lines.
    """
    lines = source.splitlines()
    out: list[str] = []
    in_class = False
    for i, line in enumerate(lines):
        if not in_class:
            m = re.match(rf"^class\s+{re.escape(class_name)}\b", line)
            if m:
                in_class = True
                out.append(line)
            continue
        # We're in the class. Stop if we hit a new top-level definition.
        # vllm uses 4-space indent; methods are indented under the class.
        if line.startswith("class ") or (
            line.startswith("def ") and not line.startswith("    def ")
        ):
            break
        out.append(line)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def _extract_method_body(
    source: str, class_name: str, method_name: str, max_lines: int = 100
) -> str:
    """Return source for a method within a class.

    Walks the class body until it finds ``def <method_name>(`` at any
    indentation, then captures lines until indentation returns to <=
    that of the def line.
    """
    lines = source.splitlines()
    in_class = False
    in_method = False
    method_indent: int | None = None
    out: list[str] = []
    for line in lines:
        if not in_class:
            if re.match(rf"^class\s+{re.escape(class_name)}\b", line):
                in_class = True
            continue
        # Hit a new top-level class -> stop.
        if line.startswith("class ") and not line.startswith("    class "):
            break
        if not in_method:
            m = re.match(rf"^(\s+)def\s+{re.escape(method_name)}\s*\(", line)
            if m:
                in_method = True
                method_indent = len(m.group(1))
                out.append(line)
            continue
        # in_method
        if line.strip() == "":
            out.append(line)
            continue
        # Stop when indentation returns to <= method's def indent
        stripped_indent = len(line) - len(line.lstrip())
        if stripped_indent <= (method_indent or 0):
            break
        out.append(line)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def get_active_sources() -> dict[str, str]:
    """Pull source text for the chunked units of vllm 0.7.3.

    Returns a dict keyed by canonical name. Empty string for missing
    members.
    """
    sources: dict[str, str] = {}

    sampling_src = _read_source("sampling_params.py")
    config_src = _read_source("config.py")
    arg_utils_src = _read_source("engine/arg_utils.py")

    if not sampling_src or not config_src or not arg_utils_src:
        sources["_extraction_error"] = f"vllm source not found under {VLLM_SOURCE_ROOT}"
        return sources

    sources["SamplingParams.class"] = _extract_class_body(
        sampling_src, "SamplingParams", max_lines=300
    )
    sources["SamplingParams._verify_args"] = _extract_method_body(
        sampling_src, "SamplingParams", "_verify_args", max_lines=80
    )
    sources["SamplingParams._verify_greedy_sampling"] = _extract_method_body(
        sampling_src, "SamplingParams", "_verify_greedy_sampling", max_lines=20
    )
    sources["SamplingParams.__post_init__"] = _extract_method_body(
        sampling_src, "SamplingParams", "__post_init__", max_lines=80
    )
    sources["GuidedDecodingParams.class"] = _extract_class_body(
        sampling_src, "GuidedDecodingParams", max_lines=80
    )

    # config.py classes
    for cls in (
        "ModelConfig",
        "CacheConfig",
        "TokenizerPoolConfig",
        "ParallelConfig",
        "SchedulerConfig",
        "LoRAConfig",
        "PromptAdapterConfig",
        "LoadConfig",
        "DecodingConfig",
        "ObservabilityConfig",
    ):
        sources[f"{cls}.class"] = _extract_class_body(config_src, cls, max_lines=120)

    # validator methods (separate chunks for invariants)
    for cls, methods in [
        (
            "ModelConfig",
            [
                "_verify_tokenizer_mode",
                "_verify_quantization",
                "_verify_cuda_graph",
                "_verify_bnb_config",
            ],
        ),
        (
            "CacheConfig",
            [
                "_verify_args",
                "_verify_cache_dtype",
                "_verify_prefix_caching",
            ],
        ),
        ("SchedulerConfig", ["__post_init__", "_verify_args"]),
        ("ParallelConfig", ["__post_init__", "_verify_args"]),
        ("LoRAConfig", ["__post_init__"]),
        ("PromptAdapterConfig", ["__post_init__"]),
        ("TokenizerPoolConfig", ["__post_init__"]),
        ("LoadConfig", ["__post_init__"]),
        ("DecodingConfig", ["__post_init__"]),
        ("ObservabilityConfig", ["__post_init__"]),
    ]:
        for m in methods:
            sources[f"{cls}.{m}"] = _extract_method_body(config_src, cls, m, max_lines=80)

    # EngineArgs is large (124+ field lines); allow generous cap.
    sources["EngineArgs.class"] = _extract_class_body(arg_utils_src, "EngineArgs", max_lines=180)
    sources["EngineArgs.__post_init__"] = _extract_method_body(
        arg_utils_src, "EngineArgs", "__post_init__", max_lines=60
    )

    return sources


# ---------------------------------------------------------------------------
# Schema chunking
# ---------------------------------------------------------------------------


def schema_chunks() -> list[ChunkSpec]:
    """Return schema-extraction chunks for active vllm 0.7.3.

    Strategy:
    - 1 chunk per config dataclass (EngineArgs, SamplingParams,
      ModelConfig, CacheConfig, SchedulerConfig, ParallelConfig,
      LoRAConfig, TokenizerPoolConfig, PromptAdapterConfig,
      LoadConfig, DecodingConfig).
    - GuidedDecodingParams companion inlined into SamplingParams chunk.

    SamplingParams emits to ``sampling_params``; everything else to
    ``engine_params``. The reference's schema layout has all the
    config-class fields under ``engine_params`` (vllm config namespaces
    flatten there); SamplingParams alone gets ``sampling_params``.
    """
    sources = get_active_sources()
    if "_extraction_error" in sources:
        return [
            ChunkSpec(
                name="source_extraction_failed",
                description=sources["_extraction_error"],
                source="",
                expected_namespaces=[],
                expected_kind="schema",
            )
        ]

    chunks: list[ChunkSpec] = []

    # SamplingParams + GuidedDecodingParams companion
    chunks.append(
        ChunkSpec(
            name="sampling_params_class",
            description="vllm.SamplingParams msgspec.Struct fields + GuidedDecodingParams companion",
            source=(
                "=== SOURCE: vllm.SamplingParams (msgspec.Struct definition) ===\n"
                + sources["SamplingParams.class"][:12000]
                + "\n\n"
                + "=== COMPANION SOURCE: vllm.GuidedDecodingParams (referenced by SamplingParams.guided_decoding) ===\n"
                + sources["GuidedDecodingParams.class"][:3000]
                + "\n\n"
            ),
            expected_namespaces=["sampling_params", "$defs.GuidedDecodingParams"],
            expected_kind="schema",
        )
    )

    # EngineArgs (the big CLI dataclass)
    chunks.append(
        ChunkSpec(
            name="engine_args_class",
            description="vllm.EngineArgs dataclass (CLI args -> engine_params)",
            source=(
                "=== SOURCE: vllm.EngineArgs (dataclass field list) ===\n"
                + sources["EngineArgs.class"][:12000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # ModelConfig
    chunks.append(
        ChunkSpec(
            name="model_config_class",
            description="vllm.ModelConfig dataclass (model-side engine_params)",
            source=(
                "=== SOURCE: vllm.ModelConfig (dataclass field list) ===\n"
                + sources["ModelConfig.class"][:12000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # CacheConfig
    chunks.append(
        ChunkSpec(
            name="cache_config_class",
            description="vllm.CacheConfig dataclass (KV cache engine_params)",
            source=(
                "=== SOURCE: vllm.CacheConfig (dataclass field list) ===\n"
                + sources["CacheConfig.class"][:8000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # SchedulerConfig
    chunks.append(
        ChunkSpec(
            name="scheduler_config_class",
            description="vllm.SchedulerConfig dataclass (scheduler engine_params)",
            source=(
                "=== SOURCE: vllm.SchedulerConfig (dataclass field list) ===\n"
                + sources["SchedulerConfig.class"][:8000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # ParallelConfig
    chunks.append(
        ChunkSpec(
            name="parallel_config_class",
            description="vllm.ParallelConfig dataclass (distributed engine_params)",
            source=(
                "=== SOURCE: vllm.ParallelConfig (dataclass field list) ===\n"
                + sources["ParallelConfig.class"][:8000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # LoRA + PromptAdapter + TokenizerPool + Decoding combined (small classes)
    small_classes = (
        "=== SOURCE: vllm.LoRAConfig ===\n"
        + sources["LoRAConfig.class"][:4000]
        + "\n\n"
        + "=== SOURCE: vllm.PromptAdapterConfig ===\n"
        + sources["PromptAdapterConfig.class"][:4000]
        + "\n\n"
        + "=== SOURCE: vllm.TokenizerPoolConfig ===\n"
        + sources["TokenizerPoolConfig.class"][:3000]
        + "\n\n"
        + "=== SOURCE: vllm.DecodingConfig ===\n"
        + sources["DecodingConfig.class"][:2000]
        + "\n\n"
    )
    chunks.append(
        ChunkSpec(
            name="small_config_classes",
            description="vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig, DecodingConfig}",
            source=small_classes,
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    return chunks


# ---------------------------------------------------------------------------
# Invariants chunking
# ---------------------------------------------------------------------------


def invariants_chunks() -> list[ChunkSpec]:
    """Return invariants-extraction chunks for active vllm 0.7.3.

    Strategy:
    - 1 chunk per validator method (most are 5-30 lines; well within
      LLM budget).
    - Two combined chunks for the tight 'sampling' methods (SamplingParams
      _verify_args + _verify_greedy_sampling + __post_init__) to keep
      context unified.
    """
    sources = get_active_sources()
    if "_extraction_error" in sources:
        return [
            ChunkSpec(
                name="source_extraction_failed",
                description=sources["_extraction_error"],
                source="",
                expected_namespaces=[],
                expected_kind="invariants",
            )
        ]

    chunks: list[ChunkSpec] = []

    # SamplingParams validation (combined: __post_init__ + _verify_args + _verify_greedy_sampling)
    chunks.append(
        ChunkSpec(
            name="sampling_params_invariants",
            description="SamplingParams __post_init__ + _verify_args + _verify_greedy_sampling",
            source=(
                "=== CONTEXT ===\n"
                "vllm.SamplingParams enforces validation via `_verify_args` "
                "called from `__post_init__`. The `_verify_args` body contains "
                "the bulk of the `if not <cond>: raise ValueError(...)` checks. "
                "`__post_init__` also adds dormant warning paths (e.g. "
                "temperature below threshold). Use namespace `vllm.sampling`.\n\n"
                "=== SOURCE: SamplingParams.__post_init__ ===\n"
                + sources["SamplingParams.__post_init__"][:5000]
                + "\n\n"
                + "=== SOURCE: SamplingParams._verify_args ===\n"
                + sources["SamplingParams._verify_args"][:5000]
                + "\n\n"
                + "=== SOURCE: SamplingParams._verify_greedy_sampling ===\n"
                + sources["SamplingParams._verify_greedy_sampling"][:2000]
                + "\n\n"
            ),
            expected_namespaces=["vllm.sampling"],
            expected_kind="invariants",
        )
    )

    # GuidedDecodingParams __post_init__ (the mutual-exclusion check)
    gdp_post = _extract_method_body(
        _read_source("sampling_params.py"),
        "GuidedDecodingParams",
        "__post_init__",
        max_lines=30,
    )
    chunks.append(
        ChunkSpec(
            name="guided_decoding_params_invariants",
            description="GuidedDecodingParams.__post_init__ mutual-exclusion check",
            source=(
                "=== CONTEXT ===\n"
                "GuidedDecodingParams enforces mutual exclusion of its guide "
                "fields (only one of json/regex/choice/grammar/json_object may "
                "be set). Use namespace `vllm.sampling` for the field key.\n\n"
                "=== SOURCE: GuidedDecodingParams.__post_init__ ===\n" + gdp_post + "\n\n"
            ),
            expected_namespaces=["vllm.sampling"],
            expected_kind="invariants",
        )
    )

    # ModelConfig validators (each one separately - they're small)
    for vmethod in (
        "_verify_tokenizer_mode",
        "_verify_quantization",
        "_verify_cuda_graph",
        "_verify_bnb_config",
    ):
        key = f"ModelConfig.{vmethod}"
        body = sources.get(key, "")
        if not body.strip():
            continue
        chunks.append(
            ChunkSpec(
                name=f"model_config_{vmethod.lstrip('_')}",
                description=f"vllm.ModelConfig.{vmethod}",
                source=(
                    f"=== CONTEXT ===\n"
                    f"vllm.ModelConfig.{vmethod} - emit one invariant per "
                    f"`if <cond>: raise ...` block. Use namespace `vllm` for "
                    f"engine_params fields.\n\n"
                    f"=== SOURCE: ModelConfig.{vmethod} ===\n" + body + "\n\n"
                ),
                expected_namespaces=["vllm"],
                expected_kind="invariants",
            )
        )

    # CacheConfig validators (in a combined chunk - they share the cache_dtype field family)
    cache_chunk_src = (
        "=== CONTEXT ===\n"
        "vllm.CacheConfig has 3 validator methods (_verify_args, "
        "_verify_cache_dtype, _verify_prefix_caching). Emit one invariant "
        "per `if <cond>: raise ...` block. Use namespace `vllm` for "
        "engine_params fields.\n\n"
        "=== SOURCE: CacheConfig._verify_args ===\n"
        + sources.get("CacheConfig._verify_args", "")
        + "\n\n"
        + "=== SOURCE: CacheConfig._verify_cache_dtype ===\n"
        + sources.get("CacheConfig._verify_cache_dtype", "")
        + "\n\n"
        + "=== SOURCE: CacheConfig._verify_prefix_caching ===\n"
        + sources.get("CacheConfig._verify_prefix_caching", "")
        + "\n\n"
    )
    chunks.append(
        ChunkSpec(
            name="cache_config_invariants",
            description="vllm.CacheConfig._verify_args + _verify_cache_dtype + _verify_prefix_caching",
            source=cache_chunk_src,
            expected_namespaces=["vllm"],
            expected_kind="invariants",
        )
    )

    # SchedulerConfig invariants
    chunks.append(
        ChunkSpec(
            name="scheduler_config_invariants",
            description="vllm.SchedulerConfig __post_init__ + _verify_args",
            source=(
                "=== CONTEXT ===\n"
                "vllm.SchedulerConfig validates `max_num_batched_tokens` vs "
                "`max_num_seqs` and the `policy` enum. Use namespace `vllm`.\n\n"
                "=== SOURCE: SchedulerConfig.__post_init__ ===\n"
                + sources.get("SchedulerConfig.__post_init__", "")
                + "\n\n"
                + "=== SOURCE: SchedulerConfig._verify_args ===\n"
                + sources.get("SchedulerConfig._verify_args", "")
                + "\n\n"
            ),
            expected_namespaces=["vllm"],
            expected_kind="invariants",
        )
    )

    # ParallelConfig invariants
    chunks.append(
        ChunkSpec(
            name="parallel_config_invariants",
            description="vllm.ParallelConfig __post_init__ + _verify_args",
            source=(
                "=== CONTEXT ===\n"
                "vllm.ParallelConfig validates `distributed_executor_backend` "
                "enum and `pipeline_parallel_size`/`tensor_parallel_size`.\n\n"
                "=== SOURCE: ParallelConfig.__post_init__ ===\n"
                + sources.get("ParallelConfig.__post_init__", "")
                + "\n\n"
                + "=== SOURCE: ParallelConfig._verify_args ===\n"
                + sources.get("ParallelConfig._verify_args", "")
                + "\n\n"
            ),
            expected_namespaces=["vllm"],
            expected_kind="invariants",
        )
    )

    # LoRAConfig + PromptAdapterConfig + TokenizerPoolConfig (small validators)
    chunks.append(
        ChunkSpec(
            name="lora_prompt_adapter_invariants",
            description="vllm.{LoRAConfig, PromptAdapterConfig, TokenizerPoolConfig}.__post_init__",
            source=(
                "=== CONTEXT ===\n"
                "Three small dataclasses share namespace `vllm`. Each "
                "`__post_init__` emits ValueError on bad shapes (max_loras<1, "
                "max_cpu_loras<max_loras, max_prompt_adapters<1, "
                "max_prompt_adapter_token==0, pool_type not in {ray}, "
                "extra_config not a dict).\n\n"
                "=== SOURCE: LoRAConfig.__post_init__ ===\n"
                + sources.get("LoRAConfig.__post_init__", "")
                + "\n\n"
                + "=== SOURCE: PromptAdapterConfig.__post_init__ ===\n"
                + sources.get("PromptAdapterConfig.__post_init__", "")
                + "\n\n"
                + "=== SOURCE: TokenizerPoolConfig.__post_init__ ===\n"
                + sources.get("TokenizerPoolConfig.__post_init__", "")
                + "\n\n"
            ),
            expected_namespaces=["vllm"],
            expected_kind="invariants",
        )
    )

    return chunks
