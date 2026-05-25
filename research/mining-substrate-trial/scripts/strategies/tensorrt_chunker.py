"""tensorrt_llm source extraction + chunking for LLM strategies.

Mirrors ``transformers_chunker.py``'s pattern for tensorrt_llm 0.21.0.

Source acquisition
------------------
tensorrt_llm is NOT installed in the project venv. Source is read from a
pre-unpacked wheel at ``/tmp/trt-llm-0.21.0/tensorrt_llm/`` (already
present from Phase 1 Day 2 miner-lift work).

Phase 3 should generalise to a venv-path-from-spec, mirroring the vllm
chunker pattern.

Chunkable units (per Phase 2.5 spec)
------------------------------------
tensorrt_llm uses Pydantic v2 validators, NOT the `if X: raise` pattern
of transformers/vllm. Each chunk MUST include:
- The class definition (so the LLM sees the field types + defaults).
- The validator decorators + bodies (so the LLM extracts the validation
  predicates).

Per-class chunks (each class definition + its in-class validators):
- TrtLlmArgs (llm_args.py): the main TRT engine args. Heavy class
  (~270 lines class body + ~70 lines of validators).
- BaseLlmArgs (llm_args.py): the parent class. ~280 lines class body +
  ~120 lines of validators with many @model_validator(mode="after")
  cross-field checks.
- CalibConfig (llm_args.py): pure dataclass-style with Literal[...] for
  device (no method validators in this class; the Literal is the
  invariant).
- SchedulerConfig (llm_args.py): StrEnum-typed fields.
- KvCacheConfig (llm_args.py): float Field with implicit bounds via
  description (not Pydantic-enforced).
- BuildCacheConfig + BuildCache (build_cache.py): the only class with
  the classic `if X: raise` pattern (max_records < 1 in BuildCache.__init__).
- LookaheadDecodingConfig (llm_args.py): @field_validator on three
  fields (max_window_size, max_ngram_size, max_verification_set_size).

Per-StrEnum chunks (for enum allowlists):
- BatchingType (llm_args.py)
- CapacitySchedulerPolicy (llm_args.py)
- ContextChunkingPolicy (llm_args.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Source path config
# ---------------------------------------------------------------------------

TENSORRT_SOURCE_ROOT = Path("/tmp/trt-llm-0.21.0/tensorrt_llm")
"""Pre-unpacked tensorrt_llm 0.21.0 source tree (from Phase 1 Day 2)."""


# ---------------------------------------------------------------------------
# ChunkSpec (mirror of transformers_chunker shape; kept independent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSpec:
    name: str
    description: str
    source: str
    expected_namespaces: list[str]
    expected_kind: str = "schema"


# ---------------------------------------------------------------------------
# Source extraction helpers
# ---------------------------------------------------------------------------


def _read_source(rel_path: str, source_root: Path | None = None) -> str:
    """Read a file under the tensorrt source root.

    Parameters
    ----------
    rel_path : str
        Path relative to the tensorrt_llm package root.
    source_root : Path | None
        If provided, read from this root. If None, fall back to the
        module-level ``TENSORRT_SOURCE_ROOT`` constant (the pre-staged
        active v0.21.0 path).
    """
    root = source_root if source_root is not None else TENSORRT_SOURCE_ROOT
    p = root / rel_path
    if not p.exists():
        return ""
    return p.read_text()


def _extract_class_with_validators(
    source: str,
    class_name: str,
    max_lines: int = 350,
) -> str:
    """Return the source for `class <class_name>(...):` up to the next
    top-level class/def definition or end of file.

    Crucially this INCLUDES the in-class @field_validator + @model_validator
    decorators (which would be skipped by a naive method-only extractor).
    """
    lines = source.splitlines()
    out: list[str] = []
    in_class = False
    for line in lines:
        if not in_class:
            m = re.match(rf"^class\s+{re.escape(class_name)}\b", line)
            if m:
                in_class = True
                out.append(line)
            continue
        # Stop at the next top-level (non-indented) definition.
        if line and not line.startswith(" ") and not line.startswith("\t"):
            if line.startswith("class ") or line.startswith("def ") or line.startswith("@"):
                break
        out.append(line)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def _extract_validators_block(source: str, class_name: str, max_chars: int = 14000) -> str:
    """Extract all validator-method blocks of a class (decorator + body).

    Walks the class linearly. When it sees `@field_validator` or
    `@model_validator` (chained decorators OK), starts capturing through
    the end of the next `def ...` method body. Skips non-validator
    methods.

    Returns the concatenated validator section (one block per validator).
    Capped at ``max_chars`` to keep chunks bounded.
    """
    lines = source.splitlines()
    in_class = False
    in_block = False
    block_def_indent: int | None = None
    pending_def = False
    out: list[str] = []
    out_chars = 0
    for line in lines:
        if not in_class:
            if re.match(rf"^class\s+{re.escape(class_name)}\b", line):
                in_class = True
            continue
        # Stop at next top-level class
        if line and not line.startswith(" ") and not line.startswith("\t"):
            if line.startswith("class "):
                break
        # Start a new block on validator decorator
        if not in_block:
            if re.match(r"^\s+@(field_validator|model_validator|validator)\b", line):
                in_block = True
                pending_def = True
                out.append(line)
                out_chars += len(line) + 1
                continue
            continue
        # We're in a validator block
        if pending_def:
            # Allow chained decorators (@classmethod after @field_validator) + the def line
            stripped = line.strip()
            if stripped.startswith("@"):
                out.append(line)
                out_chars += len(line) + 1
                continue
            if re.match(r"^\s+def\s+", line):
                # The validator method's def line
                m_indent = re.match(r"^(\s+)", line)
                block_def_indent = len(m_indent.group(1)) if m_indent else 4
                pending_def = False
                out.append(line)
                out_chars += len(line) + 1
                continue
            # Unexpected line (shouldn't normally happen)
            out.append(line)
            out_chars += len(line) + 1
            continue
        # Inside the validator's method body
        if line.strip() == "":
            out.append(line)
            out_chars += 1
            continue
        stripped_indent = len(line) - len(line.lstrip())
        if stripped_indent <= (block_def_indent or 0):
            # Method body ended; check if we're now at another validator
            if re.match(r"^\s+@(field_validator|model_validator|validator)\b", line):
                in_block = True
                pending_def = True
                out.append(line)
                out_chars += len(line) + 1
                continue
            # Some other class member - end the block
            in_block = False
            block_def_indent = None
            continue
        out.append(line)
        out_chars += len(line) + 1
        if out_chars >= max_chars:
            break
    return "\n".join(out)


def _extract_enum_class(source: str, class_name: str) -> str:
    """Return source for a small StrEnum/Enum class definition (10-20 lines)."""
    return _extract_class_with_validators(source, class_name, max_lines=30)


def get_active_sources(source_root: Path | None = None) -> dict[str, str]:
    """Pull source text for the chunked units of tensorrt_llm.

    Parameters
    ----------
    source_root : Path | None
        If None, sources come from the pre-staged active v0.21.0 tree at
        ``TENSORRT_SOURCE_ROOT``. If provided, sources come from this root -
        used by bumped-version cells (Phase 3a.2) where the chunker reads
        from a source-only venv path like
        ``/tmp/trial_tensorrt_v1_0_0_venv/src/tensorrt_llm/``.
    """
    sources: dict[str, str] = {}

    llm_args_src = _read_source("llmapi/llm_args.py", source_root=source_root)
    build_cache_src = _read_source("llmapi/build_cache.py", source_root=source_root)

    if not llm_args_src or not build_cache_src:
        root = source_root if source_root is not None else TENSORRT_SOURCE_ROOT
        sources["_extraction_error"] = f"tensorrt_llm source not found under {root}"
        return sources

    # Pydantic class definitions with their decorators
    for cls in (
        "BaseLlmArgs",
        "TrtLlmArgs",
        "TorchLlmArgs",
        "CalibConfig",
        "SchedulerConfig",
        "KvCacheConfig",
        "PeftCacheConfig",
        "LookaheadDecodingConfig",
        "DynamicBatchConfig",
        "CacheTransceiverConfig",
        "ExtendedRuntimePerfKnobConfig",
        "DecodingBaseConfig",
        "MedusaDecodingConfig",
        "EagleDecodingConfig",
        "NGramDecodingConfig",
        "MTPDecodingConfig",
    ):
        sources[cls] = _extract_class_with_validators(llm_args_src, cls, max_lines=350)

    # StrEnum allowlist classes
    for enum_cls in ("BatchingType", "CapacitySchedulerPolicy", "ContextChunkingPolicy"):
        sources[f"enum.{enum_cls}"] = _extract_enum_class(llm_args_src, enum_cls)

    # build_cache.py classes
    for cls in ("BuildCacheConfig", "BuildCache"):
        sources[f"build_cache.{cls}"] = _extract_class_with_validators(
            build_cache_src, cls, max_lines=120
        )

    return sources


# ---------------------------------------------------------------------------
# Schema chunking
# ---------------------------------------------------------------------------


def schema_chunks(source_root: Path | None = None) -> list[ChunkSpec]:
    """Return schema-extraction chunks for tensorrt_llm.

    Strategy:
    - 1 chunk per major Pydantic config class (BaseLlmArgs, TrtLlmArgs,
      CalibConfig, KvCacheConfig, SchedulerConfig, PeftCacheConfig,
      BuildCacheConfig).
    - 1 chunk for the StrEnum allowlist classes (BatchingType etc.) -
      these feed `enum` constraints into engine_params field specs.
    - 1 chunk for decoding configs (Lookahead, Medusa, Eagle, NGram, MTP)
      since they're small Pydantic dataclasses.

    The reference's schema layout puts all TRT engine fields under
    ``engine_params``. SamplingParams equivalent exists on TrtLlmArgs
    via sampling_params (TRT has no separate SamplingParams class in the
    same shape).

    Parameters
    ----------
    source_root : Path | None
        If None, sources come from the pre-staged active tree at
        ``TENSORRT_SOURCE_ROOT``. If provided, sources come from this root -
        used for bumped-version cells (Phase 3a.2).
    """
    sources = get_active_sources(source_root=source_root)
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

    # BaseLlmArgs is the parent of TrtLlmArgs; contains the bulk of fields.
    chunks.append(
        ChunkSpec(
            name="base_llm_args_class",
            description="tensorrt_llm.BaseLlmArgs Pydantic dataclass (engine_params)",
            source=(
                "=== CONTEXT ===\n"
                "BaseLlmArgs is the parent class of TrtLlmArgs and TorchLlmArgs. "
                "Most fields used at the LLM-API surface live here. Pydantic "
                "Field(default=..., description=...) is the common pattern. "
                "Extract each field as engine_params with its type + default.\n\n"
                "=== SOURCE: BaseLlmArgs (Pydantic class body) ===\n"
                + sources["BaseLlmArgs"][:14000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # TrtLlmArgs (the TRT-specific subclass)
    chunks.append(
        ChunkSpec(
            name="trt_llm_args_class",
            description="tensorrt_llm.TrtLlmArgs Pydantic dataclass (TRT-specific engine_params)",
            source=(
                "=== CONTEXT ===\n"
                "TrtLlmArgs extends BaseLlmArgs with TRT-specific build/calib "
                "fields (extended_runtime_perf_knob_config, calib_config, "
                "build_config, fast_build, enable_build_cache). Extract each "
                "Field(...) as engine_params.\n\n"
                "=== SOURCE: TrtLlmArgs ===\n" + sources["TrtLlmArgs"][:10000] + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # CalibConfig + KvCacheConfig (smallish Pydantic classes)
    chunks.append(
        ChunkSpec(
            name="calib_kv_cache_configs",
            description="tensorrt_llm.{CalibConfig, KvCacheConfig} Pydantic classes",
            source=(
                "=== CONTEXT ===\n"
                "CalibConfig and KvCacheConfig are nested Pydantic configs. "
                "Each Field(...) declaration is an engine_params entry. "
                "Note the `Literal['cuda', 'cpu']` on CalibConfig.device - "
                "emit this as `enum: [cuda, cpu]` in the schema entry.\n\n"
                "=== SOURCE: CalibConfig ===\n"
                + sources["CalibConfig"][:4000]
                + "\n\n"
                + "=== SOURCE: KvCacheConfig ===\n"
                + sources["KvCacheConfig"][:6000]
                + "\n\n"
            ),
            expected_namespaces=[
                "engine_params",
                "$defs.CalibConfig",
                "$defs.KvCacheConfig",
            ],
            expected_kind="schema",
        )
    )

    # SchedulerConfig + PeftCacheConfig + DynamicBatchConfig
    chunks.append(
        ChunkSpec(
            name="scheduler_peft_configs",
            description="tensorrt_llm.{SchedulerConfig, PeftCacheConfig, DynamicBatchConfig}",
            source=(
                "=== CONTEXT ===\n"
                "Three more Pydantic nested configs. SchedulerConfig has "
                "StrEnum-typed fields (capacity_scheduler_policy, "
                "context_chunking_policy); these gate the value to one of the "
                "enum members.\n\n"
                "=== SOURCE: SchedulerConfig ===\n"
                + sources["SchedulerConfig"][:3000]
                + "\n\n"
                + "=== SOURCE: PeftCacheConfig ===\n"
                + sources["PeftCacheConfig"][:5000]
                + "\n\n"
                + "=== SOURCE: DynamicBatchConfig ===\n"
                + sources["DynamicBatchConfig"][:3000]
                + "\n\n"
            ),
            expected_namespaces=[
                "engine_params",
                "$defs.SchedulerConfig",
                "$defs.PeftCacheConfig",
                "$defs.DynamicBatchConfig",
            ],
            expected_kind="schema",
        )
    )

    # BuildCacheConfig (non-Pydantic; classic dataclass)
    chunks.append(
        ChunkSpec(
            name="build_cache_config_class",
            description="tensorrt_llm.BuildCacheConfig + BuildCache (classic class with __init__)",
            source=(
                "=== CONTEXT ===\n"
                "BuildCacheConfig is a CLASSIC class (not Pydantic) - the "
                "constructor takes cache_root, max_records, max_cache_storage_gb. "
                "BuildCache's __init__ does the validation (`if max_records < 1: "
                "raise ValueError`). Treat the class init signature for schema.\n\n"
                "=== SOURCE: BuildCacheConfig ===\n"
                + sources["build_cache.BuildCacheConfig"][:3500]
                + "\n\n"
                + "=== SOURCE: BuildCache (companion; only __init__ relevant for schema) ===\n"
                + sources["build_cache.BuildCache"][:3500]
                + "\n\n"
            ),
            expected_namespaces=["engine_params", "$defs.BuildCacheConfig"],
            expected_kind="schema",
        )
    )

    # StrEnum allowlists (BatchingType, CapacitySchedulerPolicy, ContextChunkingPolicy)
    chunks.append(
        ChunkSpec(
            name="enum_allowlists",
            description="tensorrt_llm StrEnum classes (BatchingType, CapacitySchedulerPolicy, ContextChunkingPolicy)",
            source=(
                "=== CONTEXT ===\n"
                "These StrEnum classes are the allowlists for the corresponding "
                "field on TrtLlmArgs/SchedulerConfig. Extract each as a $defs "
                "entry, where the type is `string` and the `enum` is the list "
                "of enum values (e.g. for BatchingType: ['INFLIGHT', 'STATIC']).\n\n"
                "=== SOURCE: BatchingType ===\n"
                + sources["enum.BatchingType"]
                + "\n\n"
                + "=== SOURCE: CapacitySchedulerPolicy ===\n"
                + sources["enum.CapacitySchedulerPolicy"]
                + "\n\n"
                + "=== SOURCE: ContextChunkingPolicy ===\n"
                + sources["enum.ContextChunkingPolicy"]
                + "\n\n"
            ),
            expected_namespaces=[
                "$defs.BatchingType",
                "$defs.CapacitySchedulerPolicy",
                "$defs.ContextChunkingPolicy",
            ],
            expected_kind="schema",
        )
    )

    # Decoding configs (Lookahead, Medusa, Eagle, NGram, MTP)
    chunks.append(
        ChunkSpec(
            name="decoding_configs",
            description="tensorrt_llm decoding configs (Lookahead, Medusa, Eagle, NGram, MTP)",
            source=(
                "=== CONTEXT ===\n"
                "Speculative decoding configs (one per algorithm). Each has a "
                "small Pydantic field set; LookaheadDecodingConfig has a "
                "@field_validator on three positive-value fields.\n\n"
                "=== SOURCE: DecodingBaseConfig ===\n"
                + sources["DecodingBaseConfig"][:2000]
                + "\n\n"
                + "=== SOURCE: LookaheadDecodingConfig ===\n"
                + sources["LookaheadDecodingConfig"][:3000]
                + "\n\n"
                + "=== SOURCE: MedusaDecodingConfig ===\n"
                + sources["MedusaDecodingConfig"][:1500]
                + "\n\n"
                + "=== SOURCE: EagleDecodingConfig ===\n"
                + sources["EagleDecodingConfig"][:3000]
                + "\n\n"
                + "=== SOURCE: NGramDecodingConfig ===\n"
                + sources["NGramDecodingConfig"][:3000]
                + "\n\n"
                + "=== SOURCE: MTPDecodingConfig ===\n"
                + sources["MTPDecodingConfig"][:1500]
                + "\n\n"
            ),
            expected_namespaces=[
                "engine_params",
                "$defs.LookaheadDecodingConfig",
                "$defs.MedusaDecodingConfig",
                "$defs.EagleDecodingConfig",
                "$defs.NGramDecodingConfig",
                "$defs.MTPDecodingConfig",
            ],
            expected_kind="schema",
        )
    )

    return chunks


# ---------------------------------------------------------------------------
# Invariants chunking
# ---------------------------------------------------------------------------


def invariants_chunks(source_root: Path | None = None) -> list[ChunkSpec]:
    """Return invariants-extraction chunks for tensorrt_llm.

    Strategy:
    - 1 chunk per CLASS that has @field_validator or @model_validator
      decorators (the validator IS the invariant; the chunk must include
      the decorator + the validator body).
    - 1 chunk for the StrEnum allowlists - each StrEnum's members are
      the allowlist for the corresponding field; emit one invariant
      per (field, enum_class) pair.
    - 1 chunk for BuildCacheConfig + BuildCache (the only classic-style
      `if X: raise` invariant: BuildCache.__init__ checks max_records < 1).

    Per Phase 2.5 spec: invariant chunks MUST include the validator
    decorators in the source text, otherwise the LLM can't tell what's
    a Pydantic validator vs a plain method.

    Parameters
    ----------
    source_root : Path | None
        If None, sources come from the pre-staged active tree at
        ``TENSORRT_SOURCE_ROOT``. If provided, sources come from this root -
        used for bumped-version cells (Phase 3a.2).
    """
    sources = get_active_sources(source_root=source_root)
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

    # BaseLlmArgs validators: extract validator-only blocks (the class body
    # is 700+ lines / 30k+ chars; full body would blow the chunk budget).
    # Send TWO chunks for BaseLlmArgs validators (split by line range) so
    # ALL 14 validators get coverage.
    llm_args_src = _read_source("llmapi/llm_args.py", source_root=source_root)
    base_validators_block = _extract_validators_block(llm_args_src, "BaseLlmArgs", max_chars=28000)
    # Split roughly in half
    half = len(base_validators_block) // 2
    # Find a safe split point at the next "@" line
    split_idx = base_validators_block.find("\n    @", half)
    if split_idx == -1:
        split_idx = half
    base_v_top = base_validators_block[:split_idx]
    base_v_bot = base_validators_block[split_idx:]
    chunks.append(
        ChunkSpec(
            name="base_llm_args_validators_top",
            description="tensorrt_llm.BaseLlmArgs @field_validator + first @model_validators",
            source=(
                "=== CONTEXT ===\n"
                "tensorrt_llm uses Pydantic v2 validators (NOT `if X: raise` "
                "patterns). Each `@field_validator(field)` decorator + method "
                "is ONE invariant; each `@model_validator(mode='after')` "
                "decorator + method may contain multiple `raise ValueError` "
                "branches (each is its own invariant). Emit one invariant "
                "per `raise` statement OR per @field_validator method. "
                "Use namespace `tensorrt_llm`.\n\n"
                "Examples of validator forms to extract:\n"
                "- `@field_validator('model')\\ndef validate_model(...):\\n"
                "    if not isinstance(v, ...): raise ValueError(...)` -> "
                "  severity=error, predicate=type_is_not.\n"
                "- `@model_validator(mode='after')\\ndef "
                "validate_build_config_with_runtime_params(self):\\n"
                "    if self.max_batch_size > self.build_config.max_batch_size: "
                "raise ValueError(...)` -> severity=error, cross-field check.\n\n"
                "NOTE: this chunk shows the FIRST HALF of BaseLlmArgs validators; "
                "the rest are in a separate chunk.\n\n"
                "=== SOURCE: BaseLlmArgs validators (top half) ===\n" + base_v_top + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )
    chunks.append(
        ChunkSpec(
            name="base_llm_args_validators_bottom",
            description="tensorrt_llm.BaseLlmArgs remaining @model_validators",
            source=(
                "=== CONTEXT ===\n"
                "Continuation of BaseLlmArgs validators (second half). Same "
                "rules as the top half. Use namespace `tensorrt_llm`.\n\n"
                "=== SOURCE: BaseLlmArgs validators (bottom half) ===\n" + base_v_bot + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )

    # TrtLlmArgs validators (calib_config validator + 3 model_validators)
    trt_validators_block = _extract_validators_block(llm_args_src, "TrtLlmArgs", max_chars=10000)
    chunks.append(
        ChunkSpec(
            name="trt_llm_args_validators",
            description="tensorrt_llm.TrtLlmArgs @field_validator + @model_validator methods",
            source=(
                "=== CONTEXT ===\n"
                "TrtLlmArgs has 1 @field_validator (calib_config init) + 3 "
                "@model_validator decorators. The validate_enable_build_cache "
                "method has the most pertinent `raise ValueError(...)` block.\n\n"
                "=== SOURCE: TrtLlmArgs validators ===\n" + trt_validators_block + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )

    # LookaheadDecodingConfig @field_validator (positive values on 3 fields)
    chunks.append(
        ChunkSpec(
            name="lookahead_validator",
            description="tensorrt_llm.LookaheadDecodingConfig @field_validator (positive values)",
            source=(
                "=== CONTEXT ===\n"
                "LookaheadDecodingConfig has ONE @field_validator decorator "
                "applied to THREE fields (max_window_size, max_ngram_size, "
                "max_verification_set_size). This MUST emit THREE separate "
                "invariants - one per field. Each invariant has predicate "
                "`<= 0` (i.e. `<= 0` triggers ValueError). Use namespace "
                "`tensorrt_llm`.\n\n"
                "=== SOURCE: LookaheadDecodingConfig ===\n"
                + sources["LookaheadDecodingConfig"][:3000]
                + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )

    # CalibConfig is implicit-validator via Literal['cuda', 'cpu']
    chunks.append(
        ChunkSpec(
            name="calib_config_literal",
            description="tensorrt_llm.CalibConfig Literal['cuda', 'cpu'] (Pydantic-enforced enum)",
            source=(
                "=== CONTEXT ===\n"
                "CalibConfig.device is typed `Literal['cuda', 'cpu']`. Pydantic "
                "enforces this at construction time - any value other than "
                "'cuda' or 'cpu' raises ValidationError. Emit this as ONE "
                "invariant with predicate=not_in, severity=error.\n\n"
                "=== SOURCE: CalibConfig ===\n" + sources["CalibConfig"][:3000] + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )

    # StrEnum allowlists -> one invariant per (field, allowed_values) pair
    chunks.append(
        ChunkSpec(
            name="enum_allowlist_invariants",
            description="tensorrt_llm StrEnum allowlists as Pydantic-enforced invariants",
            source=(
                "=== CONTEXT ===\n"
                "These StrEnum classes are the type for fields on TrtLlmArgs "
                "(batching_type uses BatchingType) and SchedulerConfig "
                "(capacity_scheduler_policy uses CapacitySchedulerPolicy, "
                "context_chunking_policy uses ContextChunkingPolicy). Pydantic "
                "rejects any value outside the enum. Emit ONE invariant per "
                "field: predicate=not_in, severity=error, namespace=tensorrt_llm.\n\n"
                "=== SOURCE: BatchingType (field: batching_type on TrtLlmArgs) ===\n"
                + sources["enum.BatchingType"]
                + "\n\n"
                + "=== SOURCE: CapacitySchedulerPolicy (field: capacity_scheduler_policy on SchedulerConfig) ===\n"
                + sources["enum.CapacitySchedulerPolicy"]
                + "\n\n"
                + "=== SOURCE: ContextChunkingPolicy (field: context_chunking_policy on SchedulerConfig) ===\n"
                + sources["enum.ContextChunkingPolicy"]
                + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )

    # BuildCache (classic if X: raise pattern)
    chunks.append(
        ChunkSpec(
            name="build_cache_invariants",
            description="tensorrt_llm.BuildCache.__init__ (classic if X: raise pattern)",
            source=(
                "=== CONTEXT ===\n"
                "BuildCache.__init__ contains the only classic-style "
                "`if X: raise` invariant in the TRT codebase: `if "
                "config.max_records < 1: raise ValueError(...)`. Emit this as "
                "ONE invariant: namespace=tensorrt_llm, field=max_records (on "
                "BuildCacheConfig), predicate=lt 1, severity=error.\n\n"
                "=== SOURCE: BuildCacheConfig ===\n"
                + sources["build_cache.BuildCacheConfig"][:3000]
                + "\n\n"
                + "=== SOURCE: BuildCache (__init__ does the validation) ===\n"
                + sources["build_cache.BuildCache"][:3500]
                + "\n\n"
            ),
            expected_namespaces=["tensorrt_llm"],
            expected_kind="invariants",
        )
    )

    return chunks
