"""Observed-runtime-data extraction for inference engine implementations.

Extracts the authoritative post-construction state of native engine types
(effective params) and per-request runtime metrics. Shared by transformers.py,
vllm.py, and tensorrt.py to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

# ---------------------------------------------------------------------------
# Effective-params extraction (observed-config source)
# ---------------------------------------------------------------------------


def extract_observed_params(
    native_obj: Any,
    *,
    private_field_allowlist: set[str] | None = None,
) -> dict[str, Any]:
    """Dump a constructed native type's post-``__post_init__`` state.

    Used by the observed-config hashing pipeline (sweep-dedup.md §3.2) - after each
    backend constructs its native type (``GenerationConfig``,
    ``SamplingParams``, ``LlmArgs``), the harness calls this to extract
    the authoritative effective parameters the library settled on.

    PoC-C finding (sweep-dedup.md §3.2 and §10 decision log): live
    libraries leak private state that poisons observed_config_hash if passed through
    unfiltered. Specific leaks observed:

    - ``transformers.GenerationConfig``: ``_commit_hash``,
      ``_from_model_config``
    - ``vllm.SamplingParams``: ``_all_stop_token_ids``,
      ``_bad_words_token_ids``, ``_eos_token_id``

    Default behaviour is to strip all ``_``-prefixed fields. Engines pass
    ``private_field_allowlist`` only when a specific private field is
    genuinely measurement-relevant (rare).

    Dispatch covers the three observed native-type shapes - Pydantic
    (``model_dump``), dataclass (``dataclasses.asdict``), and
    ``__slots__``-based (vLLM msgspec-adjacent). Fallback is ``__dict__``.
    """
    allow = private_field_allowlist or set()
    raw = _dump_raw(native_obj)
    return {k: v for k, v in raw.items() if not k.startswith("_") or k in allow}


def library_version(module_name: str) -> str:
    """Return ``__version__`` of an imported library or ``"unknown"`` on failure.

    Shared helper for the observed-config capture path - all three engines publish the
    library version alongside their effective-params dump so researchers can
    disambiguate identical observed_config_hash values across library versions.
    """
    try:
        import importlib

        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unknown"


def assemble_observed_params(
    engine_params: dict[str, Any],
    sampling_params: dict[str, Any],
    module_name: str,
) -> dict[str, Any]:
    """Assemble the standard observed-params return dict for the sidecar.

    All three engines share the same return shape: ``{"engine": ...,
    "sampling": ..., "library_version": ...}``. Engine-specific extraction
    logic (BnB detection, vllm_config, llm.args) stays in each caller;
    this helper assembles the final dict and fills in the library version.

    Args:
        engine_params: Engine-specific observed params (engine section).
        sampling_params: Sampling-specific observed params.
        module_name: Library module name for :func:`library_version` lookup
            (e.g. ``"transformers"``, ``"vllm"``, ``"tensorrt_llm"``).
    """
    return {
        "engine": engine_params,
        "sampling": sampling_params,
        "library_version": library_version(module_name),
    }


def capture_two_part_observed(
    module_name: str,
    *,
    logger: logging.Logger,
    sampling_obj: Any = None,
    engine_obj: Any = None,
    engine_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Guarded two-part observed-params capture shared by the engine plugins.

    Every engine captures the same shape: a best-effort ``extract_observed_params``
    on the sampling native object and on the engine native object, then
    ``assemble_observed_params``. vLLM and TensorRT-LLM pass their native objects
    directly via ``engine_obj``; transformers resolves engine params specially (the
    nested BitsAndBytes key) and passes the built dict via ``engine_params``.

    Both extractions are best-effort: a failure is debug-logged under ``logger`` and
    leaves that section empty, never aborting the measurement.
    """
    sampling: dict[str, Any] = {}
    if sampling_obj is not None:
        try:
            sampling = extract_observed_params(sampling_obj)
        except Exception as exc:  # pragma: no cover - best-effort capture
            logger.debug("%s observed sampling-params capture failed: %s", module_name, exc)

    if engine_params is None:
        engine_params = {}
        if engine_obj is not None:
            try:
                engine_params = extract_observed_params(engine_obj)
            except Exception as exc:  # pragma: no cover - best-effort capture
                logger.debug("%s observed engine-params capture failed: %s", module_name, exc)

    return assemble_observed_params(engine_params, sampling, module_name)


def _dump_raw(native_obj: Any) -> dict[str, Any]:
    """Return a ``dict`` of every attribute observable on ``native_obj``.

    Order of attempts mirrors the three shapes :func:`extract_observed_params`
    commits to supporting; each is an explicit ``hasattr`` rather than
    ``try/except`` because some libraries raise non-``AttributeError`` for
    ``model_dump`` failures. The inner ``except Exception`` is intentional:
    sidecar-write errors should never crash a measurement run.
    """
    if hasattr(native_obj, "model_dump"):
        # Pydantic (transformers GenerationConfig inherits from PushToHubMixin;
        # its model_dump accepts mode="python").
        try:
            dumped = native_obj.model_dump(mode="python")
        except Exception:
            try:
                dumped = native_obj.model_dump()
            except Exception:
                dumped = None
        if isinstance(dumped, dict):
            return dict(dumped)
    if dataclasses.is_dataclass(native_obj) and not isinstance(native_obj, type):
        return dataclasses.asdict(native_obj)
    if hasattr(native_obj, "__slots__"):
        out: dict[str, Any] = {}
        for slot in native_obj.__slots__:
            if hasattr(native_obj, slot):
                out[slot] = getattr(native_obj, slot)
        if out:
            return out
    if hasattr(native_obj, "__dict__"):
        return dict(native_obj.__dict__)
    return {}


# ---------------------------------------------------------------------------
# Per-request metrics extraction (shared by vLLM and TRT-LLM RequestOutputs)
# ---------------------------------------------------------------------------


def count_request_tokens(outputs: Any) -> tuple[int, int]:
    """Count input (prompt) and output (generated) tokens across RequestOutputs.

    Sums across ALL outputs per request (n>1 sampling or beam search produces
    multiple sequences). Attribute access is guarded so a RequestOutput missing
    ``prompt_token_ids`` / ``outputs`` contributes zero rather than raising.

    Args:
        outputs: Iterable of ``RequestOutput`` objects (vLLM or TRT-LLM).

    Returns:
        Tuple of (input_token_count, output_token_count).
    """
    input_token_count = sum(
        len(o.prompt_token_ids) for o in outputs if hasattr(o, "prompt_token_ids")
    )
    output_token_count = sum(
        len(out.token_ids)
        for o in outputs
        if hasattr(o, "outputs") and o.outputs
        for out in o.outputs
    )
    return input_token_count, output_token_count
