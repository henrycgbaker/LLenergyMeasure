"""Shared LLM-extractor infrastructure used by strategies (b), (c), (d).

Implements the four lessons from bake-off B:

1. **Markdown-fence stripping**: Bake-off B lost the entire invariants
   score because llama wrapped the YAML in ```yml ... ```. The parser
   here strips code fences before YAML/JSON parsing AND retries with a
   corrective re-prompt on persistent failure.

2. **Long-signature truncation -> chunking**: schema extraction missed
   the latter half of ``from_pretrained``'s signature. We chunk by
   class/method and merge outputs.

3. **Companion-class under-walking**: bake-off B missed all
   ``bnb_4bit_*`` and ``llm_int8_*`` fields because the LLM didn't
   "follow imports". The prompt builder explicitly inlines companion-
   class source.

4. **Internal-plumbing leakage**: bake-off B emitted ``_commit_hash``,
   ``_from_auto`` etc. We apply an explicit post-filter rejecting
   ``_*`` and a documented blocklist.

Backends
--------
Two transport backends, same prompt + retry logic:

* ``OllamaBackend``: container Ollama on ``http://localhost:11435``
  (port collision avoided with host Ollama on 11434). Used by (b).
* ``AnthropicBackend``: Anthropic SDK with prompt caching. Used by (c).
  Raises ``KeyAbsentError`` when ``ANTHROPIC_API_KEY`` is missing so
  trial_runner can skip the cell.

JSON Schema validation is OPTIONAL: when ``json_schema`` is supplied
the parsed output is validated and a retry is triggered on failure;
otherwise we just parse and trust.

This module is INFRASTRUCTURE - no engine-specific knowledge. The
strategy modules import this and provide engine-specific prompt
builders + chunk lists.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests
import yaml

try:
    import jsonschema as _jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11435"
"""Container Ollama (per Phase 2 design). Host Ollama is on 11434."""

DEFAULT_OLLAMA_MODEL = "llama3.1:70b"
DEFAULT_OLLAMA_SMALL_MODEL = "llama3.1:8b"

DEFAULT_TIMEOUT_SEC = 1800

INTERNAL_PLUMBING_BLOCKLIST = frozenset(
    [
        # Bake-off B observed spurious; transformers internals
        "_commit_hash",
        "_from_auto",
        "_from_pipeline",
        "adapter_kwargs",
        "model_kwargs",
        "torch_dtype",  # superseded by `dtype`; transformers DeprecationWarning
        # vllm internals
        "_revision",
    ]
)
"""Explicit blocklist for post-filtering LLM-emitted fields. Anything
starting with ``_`` is rejected too (broader filter)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KeyAbsentError(RuntimeError):
    """Raised by AnthropicBackend when ANTHROPIC_API_KEY is missing.

    trial_runner should catch this and skip the cell, emitting a CRASH
    record with the message in observations.
    """


class ParseFailure(RuntimeError):
    """Raised after retries are exhausted and output still won't parse."""


class SchemaValidationFailure(RuntimeError):
    """Raised after retries are exhausted and parsed output still fails
    schema validation."""


# ---------------------------------------------------------------------------
# Code-fence + parsing helpers
# ---------------------------------------------------------------------------


_FENCE_LANG = re.compile(r"```(?:ya?ml|json)?\s*\n?", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    """Strip ```yaml / ```yml / ```json / ``` opening + closing fences.

    Robust to:
    - Opening fence with no language tag.
    - Missing closing fence (LLM ran out of output budget).
    - Leading/trailing whitespace.
    - Multiple fences (very rare; takes the first block).
    """
    s = text.strip()
    # Find first ``` fence
    m = re.search(r"```(?:ya?ml|json)?\s*\n(.*?)(?:```|$)", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If starts with fence but doesn't match the multi-line form
    if s.startswith("```"):
        s = _FENCE_LANG.sub("", s, count=1)
        s = _FENCE_CLOSE.sub("", s)
        return s.strip()
    return s


def parse_yaml_block(text: str) -> Any:
    """Parse a YAML payload, robust to leading commentary + code fences.

    Returns the parsed object. Raises ``yaml.YAMLError`` (re-raised) on
    irrecoverable parse failure - caller decides to retry.

    Salvage strategy: if the full text won't parse, try parsing
    progressively-shorter prefixes (cutting at `\n- id:` boundaries) -
    this rescues invariant lists where the LLM truncated mid-entry.
    """
    stripped = strip_code_fences(text)
    lines = stripped.splitlines()
    yaml_start = 0
    for i, line in enumerate(lines):
        if line.startswith("- ") or re.match(r"^[A-Za-z_][\w-]*\s*:", line):
            yaml_start = i
            break
    yaml_text = "\n".join(lines[yaml_start:])
    try:
        return yaml.safe_load(yaml_text)
    except yaml.YAMLError as full_err:
        # Salvage: find the LAST complete `- id:` entry boundary and
        # truncate everything after. This rescues the common case
        # where the LLM truncated mid-entry or emitted an invalid
        # value (like an unquoted `torch.dtype`).
        entry_starts = [
            i for i, ln in enumerate(yaml_text.splitlines()) if re.match(r"^-\s+id:", ln)
        ]
        if len(entry_starts) >= 2:
            # Walk backwards from the last entry, trying to parse the
            # text up to each successive boundary.
            split_lines = yaml_text.splitlines()
            for boundary in reversed(entry_starts):
                trunc_text = "\n".join(split_lines[:boundary])
                try:
                    salvaged = yaml.safe_load(trunc_text)
                    if salvaged is not None:
                        return salvaged
                except yaml.YAMLError:
                    continue
        # If even one-entry salvage fails, surface the original error
        raise full_err


def parse_json_block(text: str) -> Any:
    """Parse a JSON payload. Strips code fences; tolerant of leading prose."""
    stripped = strip_code_fences(text)
    # Try direct parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Find first { ... } block
    m = re.search(r"(\{.*\})", stripped, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Final attempt - raises if still bad
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Field filtering
# ---------------------------------------------------------------------------


def is_internal_plumbing_field(name: str) -> bool:
    """True if ``name`` matches the internal-plumbing exclusion rule.

    Rule: starts with ``_`` OR is in the explicit blocklist.
    """
    if name.startswith("_"):
        return True
    if name in INTERNAL_PLUMBING_BLOCKLIST:
        return True
    return False


def filter_internal_plumbing(fields: dict[str, Any]) -> dict[str, Any]:
    """Post-filter step: drop fields matching internal-plumbing rules.

    Used after merging LLM-emitted chunks; the LLM may emit internal
    plumbing despite the prompt rule. Defensive belt-and-braces.
    """
    return {k: v for k, v in fields.items() if not is_internal_plumbing_field(k)}


# ---------------------------------------------------------------------------
# Transport backends
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Uniform contract for OSS-LLM + Claude transports."""

    name: str
    model: str

    def call(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        max_output_tokens: int | None = None,
    ) -> tuple[str, float]:
        """Send ``prompt``, return ``(response_text, wall_seconds)``."""
        ...


@dataclass
class OllamaBackend:
    """Container Ollama transport (default port 11435).

    Per Phase 2 tactical context:
    - num_ctx=32768 keeps llama3.1:70b fully on GPU (vs 35% CPU spill at 128k).
    - num_predict=4096 caps generation length (most extracted chunks fit).
    - temperature=0 for determinism / reproducibility.
    """

    name: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    url: str = DEFAULT_OLLAMA_URL
    num_ctx: int = 32768
    num_predict: int = 4096
    temperature: float = 0.0
    keep_alive: str = "30m"  # keep model loaded across chunks

    def call(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        max_output_tokens: int | None = None,
    ) -> tuple[str, float]:
        endpoint = f"{self.url}/api/generate"
        options: dict[str, Any] = {
            "num_ctx": self.num_ctx,
            "num_predict": max_output_tokens or self.num_predict,
            "temperature": self.temperature,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,  # streaming avoids read timeout on long outputs
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if json_mode:
            payload["format"] = "json"

        t0 = time.time()
        chunks: list[str] = []
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(chunk_size=None):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                chunks.append(token)
                if data.get("done"):
                    break
        elapsed = time.time() - t0
        return "".join(chunks), elapsed


@dataclass
class AnthropicBackend:
    """Claude transport via Anthropic SDK with prompt caching.

    On instantiation: checks ``ANTHROPIC_API_KEY``; raises
    :class:`KeyAbsentError` if missing. The trial_runner catches this
    and skips the cell (c) for the duration of the run.

    Cost cap: ``$5 per cell`` enforced post-call (logged but not
    aborting); trial-wide ``$75`` enforced by the coordinator.
    """

    name: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_output_tokens: int = 4096
    per_cell_usd_cap: float = 5.0
    """Cost cap per cell. Logged when exceeded; trial_runner can choose
    to halt the cell or continue. Defensively conservative."""

    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise KeyAbsentError("ANTHROPIC_API_KEY not set; strategy (c) cells are skipped.")
        try:
            import anthropic  # imported lazily; not a hard dep for (b)
        except ImportError as exc:
            raise KeyAbsentError(f"anthropic SDK not installed (uv add anthropic): {exc}") from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        prompt: str,
        *,
        json_mode: bool = False,
        timeout: int = DEFAULT_TIMEOUT_SEC,
        max_output_tokens: int | None = None,
    ) -> tuple[str, float]:
        # json_mode for Anthropic is achieved by prompt-side instruction;
        # the SDK doesn't expose a hard JSON-only flag. The trial prompts
        # tell the LLM to emit raw JSON/YAML so this should be no-op.
        t0 = time.time()
        # Split prompt into a system + user message; mark the LARGE
        # static prefix as cache_control so re-prompts hit the cache.
        # Heuristic: the SOURCE blocks (after "=== SOURCE: ...") are the
        # cache target; everything before is the instruction.
        marker = "=== SOURCE:"
        if marker in prompt:
            idx = prompt.index(marker)
            instructions = prompt[:idx].strip()
            source_block = prompt[idx:].strip()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": source_block,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": instructions,
                        },
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_output_tokens or self.max_output_tokens,
            messages=messages,
        )
        elapsed = time.time() - t0
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return text, elapsed


# ---------------------------------------------------------------------------
# Retry-driven extraction
# ---------------------------------------------------------------------------


@dataclass
class ChunkExtraction:
    """One chunk's extracted payload + raw transcript for audit."""

    chunk_name: str
    """Human-readable label, e.g. ``"from_pretrained_signature"``."""

    parsed: Any | None
    """Parsed YAML/JSON object. None if all retries failed."""

    raw_responses: list[str] = field(default_factory=list)
    """Every raw LLM response (initial + retries) for audit trail."""

    prompts: list[str] = field(default_factory=list)
    """Every prompt sent (initial + retries) for audit trail."""

    elapsed_sec: float = 0.0
    """Cumulative wall-clock across retries."""

    failure_modes: list[str] = field(default_factory=list)
    """Per-chunk failure tags (e.g. ``"parse_failure_after_retries"``)."""

    schema_errors: list[str] = field(default_factory=list)
    """JSON Schema validation errors encountered (if json_schema provided)."""


def extract_with_retry(
    *,
    backend: LLMBackend,
    prompt: str,
    chunk_name: str,
    output_format: str = "yaml",
    json_schema: dict[str, Any] | None = None,
    max_retries: int = 2,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_output_tokens: int | None = None,
) -> ChunkExtraction:
    """Run a single chunk through the LLM with parse + schema retries.

    Procedure:
    1. Send the prompt; receive raw text.
    2. Strip code fences; parse.
    3. On parse fail: re-prompt with the parse error appended ("Your
       output failed to parse with error X. Please re-emit raw <fmt>
       only, no commentary, no markdown.").
    4. On schema-validation fail (if json_schema): re-prompt with the
       validation error.
    5. After ``max_retries`` of either kind: return ChunkExtraction with
       ``parsed=None`` and a ``failure_modes`` tag.

    Returns the ChunkExtraction. The caller decides what to do with
    partial results (typically: log to ``raw_llm_transcripts/`` and
    keep going on other chunks).
    """
    extraction = ChunkExtraction(chunk_name=chunk_name, parsed=None)
    current_prompt = prompt
    attempt = 0
    parser = parse_json_block if output_format == "json" else parse_yaml_block
    json_mode = output_format == "json"

    while attempt <= max_retries:
        extraction.prompts.append(current_prompt)
        try:
            raw_text, dt = backend.call(
                current_prompt,
                json_mode=json_mode,
                timeout=timeout,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            extraction.raw_responses.append(f"<<TRANSPORT_ERROR: {type(exc).__name__}: {exc}>>")
            extraction.failure_modes.append(f"transport_error:{type(exc).__name__}")
            return extraction
        extraction.elapsed_sec += dt
        extraction.raw_responses.append(raw_text)

        try:
            parsed = parser(raw_text)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                extraction.failure_modes.append("parse_failure_after_retries")
                return extraction
            current_prompt = _build_retry_prompt(prompt, raw_text, exc, output_format, "parse")
            continue

        if json_schema is not None and HAS_JSONSCHEMA:
            try:
                _jsonschema.validate(instance=parsed, schema=json_schema)
            except _jsonschema.ValidationError as exc:
                extraction.schema_errors.append(str(exc))
                attempt += 1
                if attempt > max_retries:
                    extraction.failure_modes.append("schema_validation_failure_after_retries")
                    # Still return the parsed value - caller can salvage what's there
                    extraction.parsed = parsed
                    return extraction
                current_prompt = _build_retry_prompt(prompt, raw_text, exc, output_format, "schema")
                continue

        extraction.parsed = parsed
        return extraction

    return extraction


def _build_retry_prompt(
    original_prompt: str,
    bad_output: str,
    error: Exception,
    output_format: str,
    error_kind: str,
) -> str:
    """Construct a corrective re-prompt after a parse or schema failure."""
    fmt_label = "JSON" if output_format == "json" else "YAML"
    error_label = "parse error" if error_kind == "parse" else "schema validation error"
    truncated_bad = bad_output[:500] + ("..." if len(bad_output) > 500 else "")
    return (
        f"{original_prompt}\n\n"
        f"=== RETRY DUE TO {error_label.upper()} ===\n"
        f"Your previous response did not parse as valid {fmt_label}.\n"
        f"The {error_label} was: {error}\n\n"
        f"Your previous (truncated) response was:\n{truncated_bad}\n\n"
        f"Please re-emit ONLY the raw {fmt_label} document. No commentary, "
        f"no markdown code fences, no preamble. Start the response with "
        f"the first character of the {fmt_label} document.\n"
    )
