"""Observed-runtime-data extraction for inference engine implementations.

Extracts the authoritative post-construction state of native engine types
(effective params) and per-request runtime metrics. Shared by transformers.py,
vllm.py, and tensorrt.py to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

import dataclasses
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


def extract_request_metrics(outputs: Any) -> tuple[list[float], list[float]]:
    """Extract per-request E2E latency and TTFT (ms) from RequestOutputs.

    vLLM and TRT-LLM both surface a ``RequestOutput.metrics`` namespace with
    ``arrival_time`` / ``finished_time`` / ``first_token_time`` timestamps.

    Best-effort: ``o.metrics`` may be None, and individual timestamp attrs may be
    missing depending on the library version / config. Any sample that cannot be
    fully computed is skipped (rather than producing a partial/garbage value).

    Args:
        outputs: Iterable of ``RequestOutput`` objects.

    Returns:
        Tuple of (per_request_latencies_ms, ttft_ms). Either list may be empty
        when no request exposed usable timestamps.
    """
    latencies_ms: list[float] = []
    ttft_ms: list[float] = []
    for o in outputs:
        metrics = getattr(o, "metrics", None)
        if metrics is None:
            continue
        arrival = getattr(metrics, "arrival_time", None)
        finished = getattr(metrics, "finished_time", None)
        first_token = getattr(metrics, "first_token_time", None)
        if arrival is not None and finished is not None:
            latencies_ms.append((finished - arrival) * 1000.0)
        if arrival is not None and first_token is not None:
            ttft_ms.append((first_token - arrival) * 1000.0)
    return latencies_ms, ttft_ms


def extract_decode_itl(outputs: Any) -> list[float]:
    """Extract a decode-average inter-token latency (ms) per request from RequestOutputs.

    The engine does not expose a true per-token timestamp stream, so this derives
    one average ITL per request from the decode phase: the wall time between the
    first generated token and request completion, divided by the number of decode
    intervals (``n_out - 1``). This is a PROPORTIONAL_ESTIMATE, not a true
    per-token measurement - it assumes uniform spacing across decode steps.

    Best-effort: a request is skipped (no sample produced) unless
    ``first_token_time`` and ``finished_time`` are both present and the request
    produced more than one output token (``n_out > 1``). The ``SimpleNamespace``
    shape of ``o.metrics`` / ``o.outputs[*].token_ids`` makes this testable
    without a live engine.

    Args:
        outputs: Iterable of ``RequestOutput`` objects.

    Returns:
        List of per-request decode-average ITL samples in ms (possibly empty).
    """
    itl_ms: list[float] = []
    for o in outputs:
        metrics = getattr(o, "metrics", None)
        if metrics is None:
            continue
        first_token = getattr(metrics, "first_token_time", None)
        finished = getattr(metrics, "finished_time", None)
        if first_token is None or finished is None:
            continue
        request_outputs = getattr(o, "outputs", None)
        if not request_outputs:
            continue
        n_out = sum(len(getattr(out, "token_ids", ()) or ()) for out in request_outputs)
        if n_out <= 1:
            continue
        itl_ms.append((finished - first_token) * 1000.0 / (n_out - 1))
    return itl_ms
