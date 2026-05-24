"""Load discovered engine schemas produced by ``scripts.engine_producers``.

The discovered JSON files in ``discovered_schemas/`` are the canonical SSOT for
"what parameters CAN be configured per engine". They are produced by running
introspection inside each engine's Docker image and committed to the repo.

This loader reads them via ``importlib.resources`` so it works in both editable
installs and installed wheels. Repeated loads are cached per-engine. Major
version mismatches (envelope schema breaking changes) raise
``UnsupportedSchemaVersionError``.

Downstream consumers (doc generators, field-name alignment, CI drift guards)
should load through this module rather than reading the JSON files directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from typing import Any

from llenergymeasure.config.ssot import Engine

# Set of envelope major versions this loader can read. 2.0.0 introduced
# canonical JSON Schema 2020-12 per-field shapes and dropped the
# ``discovery_method`` envelope key; major 1 is retained for read-compat
# with older committed schemas (downstream parsing is shape-tolerant).
# Cells write the current major only.
SUPPORTED_MAJOR_VERSIONS: frozenset[int] = frozenset({1, 2})

# Engines known to ship a discovered schema. Test-patchable module attribute
# (tests monkeypatch this to inject fake engines); derived from Engine SSOT.
_KNOWN_ENGINES: tuple[str, ...] = tuple(Engine)

# Schema files are now co-located with each engine sub-package
_SCHEMA_FILENAME = "schema.discovered.json"


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a discovered schema's major version doesn't match this loader."""


@dataclass(frozen=True)
class DiscoveryLimitation:
    """A single limitation recorded by the discovery script.

    Fields that discovery could not recover (e.g. HF's None-default fields with
    no type annotations, or kwargs that don't appear in an inspected signature)
    are surfaced here rather than silently dropped.
    """

    section: str
    fields: list[str]
    reason: str


@dataclass(frozen=True)
class DiscoveredSchema:
    """A parsed discovered engine schema.

    ``engine_params`` and ``sampling_params`` are kept as raw dicts rather than
    a typed FieldDescriptor because per-engine richness varies: TRT-LLM fields
    carry ``description`` and ``deprecated`` from its Pydantic schema, while
    vLLM and Transformers fields only have ``type`` and ``default``. Consumers
    that need uniform shape should adapt at read time.

    For schemas at major version 2 the per-field ``type`` is a canonical
    JSON Schema 2020-12 value (primitive name, type-array including
    ``"null"``, or ``anyOf`` branch). Major-1 schemas (read-compat only)
    carry the legacy Python-string compact form; consumers that classify
    types should detect either shape.
    """

    schema_version: str
    engine: str
    engine_version: str
    engine_commit_sha: str | None
    image_ref: str
    base_image_ref: str
    discovered_at: datetime
    discovery_limitations: list[DiscoveryLimitation] = field(default_factory=list)
    engine_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    sampling_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # JSON Schema 2020-12 ``$defs`` block. Populated by producers that walk
    # nested classes (Pydantic via model_json_schema, msgspec via
    # msgspec.json.schema, stdlib dataclasses via the #671 walker
    # enhancement). Empty when the engine producer doesn't yet emit $defs.
    defs: dict[str, dict[str, Any]] = field(default_factory=dict)


class SchemaLoader:
    """Load and cache discovered engine schemas.

    Uses a per-instance dict cache (rather than ``functools.lru_cache``) so
    multiple SchemaLoader instances don't share state - convenient for tests
    and for isolating reloads after a schema refresh.
    """

    def __init__(self) -> None:
        self._cache: dict[str, DiscoveredSchema] = {}

    def load_schema(self, engine: str) -> DiscoveredSchema:
        """Load the discovered schema for ``engine``.

        Raises:
            ValueError: ``engine`` is not a known engine name.
            FileNotFoundError: No discovered JSON exists for ``engine``.
            UnsupportedSchemaVersionError: Discovered schema major version
                is not in ``SUPPORTED_MAJOR_VERSIONS``.
            json.JSONDecodeError: Discovered file is not valid JSON.
        """
        if engine not in _KNOWN_ENGINES:
            raise ValueError(f"Unknown engine {engine!r}. Known engines: {list(_KNOWN_ENGINES)}.")

        cached = self._cache.get(engine)
        if cached is not None:
            return cached

        engine_package = f"llenergymeasure.engines.{engine}"
        try:
            raw_text = (resources.files(engine_package) / _SCHEMA_FILENAME).read_text()
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            # resources.files raises ModuleNotFoundError on a missing engine sub-package.
            raise FileNotFoundError(
                f"Discovered schema for engine {engine!r} not found. "
                f"Run `./scripts/refresh_discovered_schemas.sh {engine}` to generate it."
            ) from exc

        parsed = _parse_envelope(engine=engine, raw_text=raw_text)
        self._cache[engine] = parsed
        return parsed

    def load_all_schemas(self) -> dict[str, DiscoveredSchema]:
        """Load all known engines' schemas.

        Does not skip missing files - every engine in ``_KNOWN_ENGINES`` must
        have a discovered schema. Callers that need tolerance should iterate and
        catch ``FileNotFoundError`` themselves.
        """
        return {engine: self.load_schema(engine) for engine in _KNOWN_ENGINES}

    def invalidate(self, engine: str | None = None) -> None:
        """Drop cached schema(s). Useful after a schema refresh in-process."""
        if engine is None:
            self._cache.clear()
        else:
            self._cache.pop(engine, None)


def _parse_envelope(*, engine: str, raw_text: str) -> DiscoveredSchema:
    data = json.loads(raw_text)

    schema_version = data["schema_version"]
    major = _major_version(schema_version)
    if major not in SUPPORTED_MAJOR_VERSIONS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_MAJOR_VERSIONS))
        raise UnsupportedSchemaVersionError(
            f"Discovered schema for {engine!r} has schema_version={schema_version!r} "
            f"(major={major}); this SchemaLoader supports majors {{{supported}}}. "
            f"Regenerate with a matching discovery script, or upgrade the loader."
        )

    limitations_raw = data.get("discovery_limitations", [])
    limitations = [
        DiscoveryLimitation(
            section=item.get("section", ""),
            fields=list(item.get("fields", [])),
            reason=item.get("reason", ""),
        )
        for item in limitations_raw
    ]

    return DiscoveredSchema(
        schema_version=schema_version,
        engine=data["engine"],
        engine_version=data["engine_version"],
        engine_commit_sha=data.get("engine_commit_sha"),
        image_ref=data["image_ref"],
        # ``base_image_ref`` may be missing OR explicitly null in the
        # discovered envelope (vllm and tensorrt no longer derive it from a
        # first-party Dockerfile). Treat both as "fall back to image_ref"
        # so the loader contract stays "always populated". Empty string is
        # NOT treated as null - surface it as a config error via the
        # subsequent ``image_ref`` indexing rather than silently masking.
        base_image_ref=(
            data["base_image_ref"] if data.get("base_image_ref") is not None else data["image_ref"]
        ),
        discovered_at=_parse_iso(data["discovered_at"]),
        discovery_limitations=limitations,
        engine_params=data.get("engine_params", {}),
        sampling_params=data.get("sampling_params", {}),
        defs=data.get("$defs", {}),
    )


def _major_version(version: str) -> int:
    """Parse major from a semver-ish string. ``"1.0.0"`` -> ``1``."""
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise UnsupportedSchemaVersionError(
            f"Unparseable schema_version {version!r}: expected semver like '1.0.0'."
        ) from exc


def _parse_iso(value: str) -> datetime:
    # Accept both "...+00:00" and "...Z" terminations.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)
