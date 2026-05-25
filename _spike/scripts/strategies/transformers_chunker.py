"""Transformers source extraction + chunking for LLM strategies.

Chunking principles per Phase 2 design:
- One chunk = one logical extraction unit (class + companions, OR
  validator method).
- Each chunk targets <2k input tokens of source (8k char rough proxy).
- Companion classes (e.g. BitsAndBytesConfig for from_pretrained,
  CompileConfig for GenerationConfig) are inlined into the same chunk
  - the LLM does NOT follow imports per Bake-off B lesson.

Two chunk lists are produced:

* ``schema_chunks()`` - chunks for SCHEMA extraction (fields + types +
  defaults). Returns ``[ChunkSpec, ...]`` keyed for the schema prompt.
* ``invariants_chunks()`` - chunks for INVARIANT extraction. Splits the
  GenerationConfig.validate() body into logical sections (decoding,
  cache, performance, watermarking, sampling-only-when-greedy,
  beam-only-when-greedy, cache-related, etc.) PLUS __init__ for
  type-mismatch invariants.

Active version is read via ``inspect.getsource``. Bumped versions
(Phase 3a.2) are read from a file-based ``source_root`` path - the
caller passes the unpacked-wheel root, the chunker AST-parses
configuration_utils.py + modeling_utils.py + quantization configs
directly. If the AST parse fails (e.g. API rename in transformers 5.x),
the failure surfaces as ``_extraction_error`` in the chunks list - that
IS the brittleness signal we measure.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkSpec:
    """One LLM-extraction chunk.

    The ``source`` field contains the full text to inline into the
    prompt. ``expected_namespaces`` tells the LLM which top-level
    namespaces it should emit fields under (e.g. ``["engine_params"]``
    or ``["sampling_params", "$defs.CompileConfig"]``).
    """

    name: str
    """Filesystem-safe chunk name, used for transcript filenames."""

    description: str
    """One-line human description, e.g. "BitsAndBytesConfig quantisation fields"."""

    source: str
    """Concatenated source code (with section headers)."""

    expected_namespaces: list[str]
    """Which schema namespaces this chunk's fields land under (schema
    task only; invariants task uses a different field)."""

    expected_kind: str = "schema"
    """One of ``"schema"`` or ``"invariants"`` - which extractor task."""


# ---------------------------------------------------------------------------
# Source extraction (active version; reads via inspect)
# ---------------------------------------------------------------------------


def get_active_sources() -> dict[str, str]:
    """Pull source via ``inspect.getsource`` for the project's installed
    transformers (active version on the project venv).

    Returns a dict keyed by canonical name. Empty strings for missing
    members (so the caller can branch).
    """
    sources: dict[str, str] = {}
    try:
        from transformers import (
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            GenerationConfig,
            PreTrainedModel,
        )
        from transformers.generation.configuration_utils import (
            CompileConfig,
            SynthIDTextWatermarkingConfig,
            WatermarkingConfig,
        )

        # from_pretrained source: PreTrainedModel is the canonical
        # location; AutoModelForCausalLM's is a thin wrapper.
        sources["PreTrainedModel.from_pretrained"] = inspect.getsource(
            PreTrainedModel.from_pretrained
        )
        sources["AutoModelForCausalLM.from_pretrained"] = inspect.getsource(
            AutoModelForCausalLM.from_pretrained
        )
        sources["BitsAndBytesConfig"] = inspect.getsource(BitsAndBytesConfig)
        sources["BitsAndBytesConfig.__init__"] = inspect.getsource(BitsAndBytesConfig.__init__)

        # GenerationConfig: __init__ (sampling params) + validate (invariants)
        sources["GenerationConfig.__init__"] = inspect.getsource(GenerationConfig.__init__)
        sources["GenerationConfig.validate"] = inspect.getsource(GenerationConfig.validate)
        sources["GenerationConfig.__doc__"] = GenerationConfig.__doc__ or ""

        sources["CompileConfig"] = inspect.getsource(CompileConfig)
        sources["WatermarkingConfig"] = inspect.getsource(WatermarkingConfig)
        sources["SynthIDTextWatermarkingConfig"] = inspect.getsource(SynthIDTextWatermarkingConfig)

    except Exception as exc:
        sources["_extraction_error"] = f"{type(exc).__name__}: {exc}"

    return sources


# ---------------------------------------------------------------------------
# Source extraction (bumped versions; AST-parse from file path)
# ---------------------------------------------------------------------------


def _ast_get_class_source(file_src: str, class_name: str) -> str:
    """Return the source text of a class declaration (incl. body) by AST.

    Returns "" if not found. Falls back to text-based extraction if AST
    parse fails (e.g. file uses syntax newer than the host Python).
    """
    try:
        tree = ast.parse(file_src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(file_src, node) or ""
    return ""


def _ast_get_method_source(file_src: str, class_name: str, method_name: str) -> str:
    """Return the source text of a method inside a class by AST. Empty if not found."""
    try:
        tree = ast.parse(file_src)
    except SyntaxError:
        return ""
    for cls in ast.walk(tree):
        if not (isinstance(cls, ast.ClassDef) and cls.name == class_name):
            continue
        for member in cls.body:
            if (
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == method_name
            ):
                return ast.get_source_segment(file_src, member) or ""
    return ""


def _ast_get_class_docstring(file_src: str, class_name: str) -> str:
    """Return the docstring of a class (the leading triple-string), or ""."""
    try:
        tree = ast.parse(file_src)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_docstring(node) or ""
    return ""


def get_sources_from_path(source_root: Path) -> dict[str, str]:
    """Pull source via AST file-parse from a source-only venv path.

    Used for bumped-version cells in Phase 3a.2. The chunker's chunked
    units are read from canonical file locations in the transformers
    package layout:

    - ``modeling_utils.py``: ``PreTrainedModel.from_pretrained``.
    - ``models/auto/modeling_auto.py``: ``AutoModelForCausalLM.from_pretrained``
      (best-effort; modelling-auto path can shift across versions).
    - ``generation/configuration_utils.py``: ``GenerationConfig.__init__``,
      ``GenerationConfig.validate``, ``GenerationConfig`` docstring,
      ``CompileConfig``, ``WatermarkingConfig``, ``SynthIDTextWatermarkingConfig``.
    - ``utils/quantization_config.py``: ``BitsAndBytesConfig``.

    If any required file is missing (e.g. major-version rename) or AST
    parse fails (e.g. syntax not supported by this Python), the relevant
    source comes back empty. The chunker then emits a chunk with an
    empty ``source`` - the LLM will see no content for that chunk and
    fail extraction, which is reported as the cell's brittleness mode.

    Returns a dict shaped identically to ``get_active_sources()``.
    """
    sources: dict[str, str] = {}

    def _read(rel_path: str) -> str:
        p = source_root / rel_path
        if not p.exists():
            return ""
        try:
            return p.read_text()
        except Exception:
            return ""

    modeling_utils_src = _read("modeling_utils.py")
    config_utils_src = _read("generation/configuration_utils.py")
    quant_src = _read("utils/quantization_config.py")

    # Best-effort modelling_auto location (varies across versions).
    automod_candidates = [
        "models/auto/modeling_auto.py",
        "models/auto/auto_factory.py",
    ]
    automod_src = ""
    for cand in automod_candidates:
        automod_src = _read(cand)
        if automod_src:
            break

    if not modeling_utils_src and not config_utils_src:
        sources["_extraction_error"] = (
            f"transformers source not found under {source_root} - "
            f"expected modeling_utils.py and generation/configuration_utils.py"
        )
        return sources

    # PreTrainedModel.from_pretrained
    sources["PreTrainedModel.from_pretrained"] = _ast_get_method_source(
        modeling_utils_src, "PreTrainedModel", "from_pretrained"
    )

    # AutoModelForCausalLM.from_pretrained - may be inherited from a base
    # class; best-effort extraction. Empty result is acceptable (the
    # chunker only references this for completeness).
    if automod_src:
        sources["AutoModelForCausalLM.from_pretrained"] = _ast_get_method_source(
            automod_src, "AutoModelForCausalLM", "from_pretrained"
        )
    else:
        sources["AutoModelForCausalLM.from_pretrained"] = ""

    # BitsAndBytesConfig
    if quant_src:
        sources["BitsAndBytesConfig"] = _ast_get_class_source(quant_src, "BitsAndBytesConfig")
        sources["BitsAndBytesConfig.__init__"] = _ast_get_method_source(
            quant_src, "BitsAndBytesConfig", "__init__"
        )
    else:
        sources["BitsAndBytesConfig"] = ""
        sources["BitsAndBytesConfig.__init__"] = ""

    # GenerationConfig + companions
    sources["GenerationConfig.__init__"] = _ast_get_method_source(
        config_utils_src, "GenerationConfig", "__init__"
    )
    sources["GenerationConfig.validate"] = _ast_get_method_source(
        config_utils_src, "GenerationConfig", "validate"
    )
    sources["GenerationConfig.__doc__"] = _ast_get_class_docstring(
        config_utils_src, "GenerationConfig"
    )
    sources["CompileConfig"] = _ast_get_class_source(config_utils_src, "CompileConfig")
    sources["WatermarkingConfig"] = _ast_get_class_source(
        config_utils_src, "WatermarkingConfig"
    )
    sources["SynthIDTextWatermarkingConfig"] = _ast_get_class_source(
        config_utils_src, "SynthIDTextWatermarkingConfig"
    )

    # If the critical sources are empty, record an error.
    critical = [
        "PreTrainedModel.from_pretrained",
        "GenerationConfig.__init__",
        "GenerationConfig.validate",
    ]
    missing = [k for k in critical if not sources[k]]
    if missing:
        sources["_extraction_error"] = (
            f"transformers AST extraction failed for {missing} under {source_root}"
        )

    return sources


# ---------------------------------------------------------------------------
# Schema chunking
# ---------------------------------------------------------------------------


def _truncate_signature_source(src: str, max_chars: int = 6000) -> str:
    """Truncate a long method source to its signature + first chunk of body.

    PreTrainedModel.from_pretrained is ~43k chars. We don't need the
    full body for schema extraction - the signature lists the named
    params; the docstring + the kwarg-pop pattern documents the rest.

    Strategy: take the signature line through the closing ``)`` of the
    parameter list, plus the docstring (everything until the first
    non-string-body line). Cap at max_chars.
    """
    # Find the signature end - the ``) ->`` or ``) :`` after the def
    m = re.search(r"(def\s+\w+\([^)]*\)(?:\s*->\s*[^:]+)?\s*:)", src, re.DOTALL)
    if not m:
        return src[:max_chars]
    sig = m.group(1)
    after_sig = src[m.end() :]
    # Take signature + docstring (everything in triple-quote pair)
    doc_match = re.search(r'^\s*(r?"""(.*?)""")', after_sig, re.DOTALL)
    if doc_match:
        result = sig + "\n" + doc_match.group(1)
    else:
        result = sig
    # Then add the first ~3k chars of body (where the kwarg-pop pattern
    # explicitly names common kwargs like _commit_hash, adapter_kwargs).
    if len(after_sig) > 0 and doc_match is not None:
        body_start = doc_match.end()
        body_extract = after_sig[body_start : body_start + 3000]
        result += "\n" + body_extract
    return result[:max_chars]


def schema_chunks(source_root: Path | None = None) -> list[ChunkSpec]:
    """Return the list of schema-extraction chunks for transformers.

    Round 2+ uses 5 chunks (expanded docstring coverage per round-1
    failure analysis - many fields were documented in the docstring
    body we'd truncated):

    1. **from_pretrained signature** -> engine_params (NAMED args only).
    2. **from_pretrained docstring kwargs** -> engine_params (the
       docstring documents 20+ kwargs via Sphinx kwargs that the
       signature elides; round 1 missed 21/40 engine_params here).
    3. **BitsAndBytesConfig + nested configs** -> engine_params.
    4. **GenerationConfig.__init__** -> sampling_params (NAMED args).
    5. **GenerationConfig docstring (full)** -> sampling_params (the
       docstring documents all the kwargs.pop'd fields; round 1
       missed 33/68 sampling_params here).

    Parameters
    ----------
    source_root : Path | None
        If None, sources come from ``inspect.getsource`` on the project
        venv's installed transformers (active-version path). If provided,
        sources come from AST-parsing files under that root (bumped-
        version path, Phase 3a.2).
    """
    sources = get_active_sources() if source_root is None else get_sources_from_path(source_root)

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

    # Chunk 1: from_pretrained signature + first 4k of docstring
    fp_src = _truncate_signature_source(sources["PreTrainedModel.from_pretrained"], max_chars=6000)
    chunks.append(
        ChunkSpec(
            name="from_pretrained_signature",
            description="PreTrainedModel.from_pretrained signature + leading docstring",
            source=(
                "=== SOURCE: PreTrainedModel.from_pretrained (signature + leading docstring) ===\n"
                + fp_src
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # Chunk 2: from_pretrained REMAINING docstring (kwargs section)
    # The Sphinx kwargs section (after "Parameters:") documents
    # device_map, dtype, attn_implementation, tp_plan etc.
    full_fp = sources["PreTrainedModel.from_pretrained"]
    # Find the body of the docstring after the signature
    import re as _re

    m = _re.search(r'r?"""(.*?)"""', full_fp, _re.DOTALL)
    docstring_body = m.group(1) if m else ""
    # Take the LATTER half of the docstring (kwargs section)
    half_point = len(docstring_body) // 2
    docstring_kwargs = docstring_body[half_point:][:12000]
    chunks.append(
        ChunkSpec(
            name="from_pretrained_docstring_kwargs",
            description="PreTrainedModel.from_pretrained docstring kwargs (engine_params via Sphinx)",
            source=(
                "=== CONTEXT ===\n"
                "Below is the LATTER HALF of PreTrainedModel.from_pretrained's "
                "docstring. It documents `**kwargs` that the signature does NOT "
                "name (device_map, dtype, attn_implementation, tp_plan, "
                "trust_remote_code, etc.). Extract these as engine_params, "
                "using the Sphinx-style `name (type, *optional*, defaults to X)` "
                "pattern to determine type + default.\n\n"
                "=== SOURCE: PreTrainedModel.from_pretrained docstring (kwargs section) ===\n"
                + docstring_kwargs
                + "\n\n"
            ),
            expected_namespaces=["engine_params"],
            expected_kind="schema",
        )
    )

    # Chunk 3: BitsAndBytesConfig + companion CompileConfig
    chunks.append(
        ChunkSpec(
            name="bitsandbytes_compile_configs",
            description="BitsAndBytesConfig + CompileConfig + WatermarkingConfig fields",
            source=(
                "=== SOURCE: BitsAndBytesConfig (quantisation fields that "
                "surface as engine_params kwargs) ===\n"
                + sources["BitsAndBytesConfig"][:6000]
                + "\n\n"
                + "=== SOURCE: CompileConfig (used in sampling_params; "
                "$defs entry) ===\n"
                + sources["CompileConfig"][:3000]
                + "\n\n"
                + "=== SOURCE: WatermarkingConfig ===\n"
                + sources["WatermarkingConfig"][:3000]
                + "\n\n"
            ),
            expected_namespaces=["engine_params", "$defs.CompileConfig"],
            expected_kind="schema",
        )
    )

    # Chunk 4: GenerationConfig.__init__ source
    gc_init = sources["GenerationConfig.__init__"]
    chunks.append(
        ChunkSpec(
            name="generation_config_init",
            description="GenerationConfig.__init__ (sampling_params via kwargs.pop)",
            source=("=== SOURCE: GenerationConfig.__init__ ===\n" + gc_init + "\n\n"),
            expected_namespaces=["sampling_params"],
            expected_kind="schema",
        )
    )

    # Chunk 5: GenerationConfig FULL docstring
    # Round 1 truncated this to 8000 chars; full doc is 19k chars and
    # documents ALL kwargs. Send in full (still fits in 32k ctx).
    gc_doc = sources["GenerationConfig.__doc__"]
    chunks.append(
        ChunkSpec(
            name="generation_config_docstring",
            description="GenerationConfig docstring FULL (documents all kwargs.pop'd sampling params)",
            source=(
                "=== CONTEXT ===\n"
                "Below is GenerationConfig's FULL docstring. It documents "
                "every sampling param the __init__ kwargs.pop's. Extract "
                "as sampling_params using the Sphinx pattern "
                "`name (type, *optional*, defaults to X)`.\n\n"
                "=== SOURCE: GenerationConfig docstring ===\n" + gc_doc[:16000] + "\n\n"
            ),
            expected_namespaces=["sampling_params"],
            expected_kind="schema",
        )
    )

    return chunks


# ---------------------------------------------------------------------------
# Invariants chunking
# ---------------------------------------------------------------------------


def _split_validate_into_sections(validate_src: str) -> list[tuple[str, str]]:
    """Split GenerationConfig.validate() into logical sections based on
    the "# 1.x" / "# 2.x" comment headers in the source.

    Returns list of (section_label, source_text). If no sections detected,
    returns the full body as a single chunk.
    """
    # Match "# 1.1.", "# 2.5.", etc. as section markers.
    # Capture text from each marker to the next.
    pattern = re.compile(
        r"(#\s+\d+(?:\.\d+)?\.\s+.*?)(?=\n\s+#\s+\d+(?:\.\d+)?\.\s+|\n\n|\Z)", re.DOTALL
    )
    matches = list(pattern.finditer(validate_src))
    if not matches:
        return [("validate_full", validate_src)]
    sections: list[tuple[str, str]] = []
    for m in matches:
        section_text = m.group(0).strip()
        # First line is the label
        first_line = section_text.split("\n", 1)[0]
        label = re.sub(r"[^\w_.]+", "_", first_line.strip("# .")).strip("_")[:60]
        sections.append((label, section_text))
    return sections


def invariants_chunks(source_root: Path | None = None) -> list[ChunkSpec]:
    """Return the list of invariant-extraction chunks for transformers.

    Strategy:
    - **One chunk per logical section of validate()** (decoding,
      cache, performance, watermarking, sampling-only-greedy,
      beam-only-greedy, num_return_sequences cross-field, cache cross-
      field). 5-8 chunks.
    - **One chunk for __init__-time validation** (type checks for
      cache_implementation, compile_config, etc.).

    Per Phase 2 chunking lesson: by sending logical sections separately,
    the LLM doesn't truncate. By sending the COMPANION classes
    (CompileConfig referenced from validate) inline in the relevant
    chunks, the LLM doesn't fail to "follow imports".

    Parameters
    ----------
    source_root : Path | None
        If None, sources via ``inspect`` on the project's installed
        transformers. If provided, AST-parse from this source-only-venv
        path (Phase 3a.2 bumped versions).
    """
    sources = get_active_sources() if source_root is None else get_sources_from_path(source_root)
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

    # __init__ invariants: type checks + early field validations
    chunks.append(
        ChunkSpec(
            name="generation_config_init_invariants",
            description="GenerationConfig.__init__ raise/warn patterns",
            source=(
                "=== SOURCE: GenerationConfig.__init__ ===\n"
                + sources["GenerationConfig.__init__"]
                + "\n\n"
                + "=== COMPANION SOURCE: CompileConfig (referenced in init) ===\n"
                + sources["CompileConfig"][:2000]
                + "\n\n"
            ),
            expected_namespaces=["transformers.sampling"],
            expected_kind="invariants",
        )
    )

    # ROUND 2 ADDITION: BitsAndBytesConfig type checks.
    # Round 1 missed 8 type_is_not invariants here. Same source we sent
    # for schema chunk 2, but THIS time prompt for INVARIANTS extraction.
    chunks.append(
        ChunkSpec(
            name="bitsandbytes_config_invariants",
            description="BitsAndBytesConfig.__init__ type checks (load_in_4bit, bnb_4bit_*, llm_int8_*)",
            source=(
                "=== CONTEXT ===\n"
                "BitsAndBytesConfig.__init__ has 8+ `if not isinstance(...): "
                "raise TypeError(...)` style type checks. Each is a separate "
                "invariant with severity=error, predicate=type_is_not. The "
                "namespace should be `transformers` (NOT transformers.sampling, "
                "since these are engine_params).\n\n"
                "=== SOURCE: BitsAndBytesConfig (full) ===\n"
                + sources["BitsAndBytesConfig"][:9000]
                + "\n\n"
            ),
            expected_namespaces=["transformers"],
            expected_kind="invariants",
        )
    )

    # validate() sectioned chunks - with round-2 dormancy improvement
    validate_src = sources["GenerationConfig.validate"]
    sections = _split_validate_into_sections(validate_src)
    for i, (label, section_src) in enumerate(sections):
        # Round 2 prompt improvement: for the sampling-only-when-greedy
        # section, explicitly tell the LLM to emit ONE invariant PER
        # FIELD (round 1 collapsed all into one keyed by do_sample).
        extra_context = ""
        if "sampling_only" in label.lower() or "beam_only" in label.lower():
            extra_context = (
                "\n\nIMPORTANT: Each `if self.<field> is not None and <cond>:` "
                "block must produce ONE invariant per <field>. Do NOT collapse "
                "multiple field checks into one invariant keyed by `do_sample` "
                "or `num_beams`. Each per-field check is its OWN dormant "
                "invariant. Key by the FIELD being flagged (temperature, top_p, "
                "min_p, etc.), NOT by the gating field (do_sample / num_beams).\n"
            )
        chunks.append(
            ChunkSpec(
                name=f"validate_section_{i:02d}_{label}",
                description=f"GenerationConfig.validate section: {label}",
                source=(
                    "=== CONTEXT: GenerationConfig.validate() ===\n"
                    "This is one logical section. Other sections are mined "
                    "separately. Focus only on validations visible in THIS "
                    "section." + extra_context + "\n\n"
                    f"=== SOURCE: validate() section {i + 1} ({label}) ===\n" + section_src + "\n\n"
                ),
                expected_namespaces=["transformers.sampling"],
                expected_kind="invariants",
            )
        )

    return chunks
