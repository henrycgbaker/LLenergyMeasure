"""Corpus validation of the keys a config declares under ``<engine>.engine_params``.

An ``engine_params`` block is ``extra="allow"``: a key the generated engine config
does not curate still parses, and forwards to the engine. That openness is
deliberate - it keeps un-curated engine fields reachable without regenerating the
config surface - but it also means a misspelt key parses. In a sweep the cost is
silent: an axis over ``vllm.engine_params.max_num_seq`` (for ``max_num_seqs``)
expands into several distinctly-hashed experiments that all carry the same
effective engine configuration, so the whole sweep runs and measures one point.

This module vets the declared keys against the engine-knowledge corpus. The
vocabulary is the FULL discovered surface for the engine, never a curated subset or
a hand-kept allowlist:

- ``llenergymeasure/engines/<engine>/schema.discovered.json`` enumerates the
  engine's own argument surface (for vLLM, every ``EngineArgs`` field at the pinned
  version). Nested blocks appear there as ``$ref`` entries into the envelope's
  ``$defs``, so nesting is validated to whatever depth those refs resolve to a
  definition carrying ``properties``; anything the corpus leaves opaque (an enum, a
  bare type, a block it could not model) admits every key below it.
- The generated ``config.generated.<engine>`` model fields union in, so a curated
  field the discovery pass could not reach stays admissible.
- :attr:`~llenergymeasure.config.ssot.EngineDescriptor.engine_params_extras`
  unions in the llem-owned passthrough keys the engine's own surface never had.

Rejection is only sound where the corpus enumerates a CLOSED surface. Two cases
fall back to admitting the key:

- The corpus records a var-kwargs limitation for ``engine_params``. Transformers
  discovers ``from_pretrained`` by signature, and that signature ends in
  ``**kwargs`` whose documented members live only in a class docstring the
  discovery pass cannot read, so any key may be legitimate there.
- No corpus ships for the engine on this install.

Both fall back to warn-and-admit rather than reject: a close typo still warns
(:class:`~llenergymeasure.config.warnings.ConfigValidationWarning`), and the
missing-corpus case additionally warns that the check could not run, so a
silently-unchecked surface is never mistaken for a validated one.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from llenergymeasure.config.ssot import ENGINES, Engine, engine_str
from llenergymeasure.config.warnings import ConfigValidationWarning

__all__ = [
    "CLOSE_MATCH_CUTOFF",
    "close_match_hint",
    "reset_corpus_cache",
    "validate_engine_params_keys",
]

CLOSE_MATCH_CUTOFF = 0.8
"""difflib cutoff for a did-you-mean suggestion on an unrecognised key. 0.8 keeps
the suggestion conservative (clear typos like ``dtypee`` -> ``dtype``) without
guessing at a name that merely shares a prefix."""

_REF_PREFIX = "#/$defs/"

_VAR_KWARGS_MARKER = "**"
"""Substring that marks a discovery limitation as a var-kwargs (open) surface: the
limitation names the parameter it could not expand, e.g.
``AutoModelForCausalLM.from_pretrained.**kwargs``."""


@dataclass(frozen=True)
class _Corpus:
    """The corpus-visible ``engine_params`` surface for one engine."""

    properties: Mapping[str, Any]
    """Top-level ``engine_params`` field name -> its discovered spec."""

    definitions: Mapping[str, Any]
    """The envelope's ``$defs``, which the specs' ``$ref``s point into."""

    closed: bool
    """True when the corpus enumerates the whole surface, so an unrecognised key
    cannot be a legitimate engine field."""


@lru_cache(maxsize=8)
def _corpus(engine: str) -> _Corpus | None:
    """The engine's corpus surface, or None when no corpus ships for it.

    Memoised per engine (the discovered schema is a committed artifact) so a sweep
    validating many configs derives each surface once; tests that mutate the
    on-disk corpus call :func:`reset_corpus_cache`.
    """
    from llenergymeasure.config.schema_loader import load_schema_cached

    schema = load_schema_cached(engine)
    if schema is None:
        return None
    open_surface = any(
        _VAR_KWARGS_MARKER in name
        for limitation in schema.discovery_limitations
        if limitation.section == "engine_params"
        for name in limitation.fields
    )
    return _Corpus(
        properties=schema.engine_params,
        definitions=schema.definitions,
        closed=not open_surface,
    )


def reset_corpus_cache() -> None:
    """Clear the memoised corpus surfaces; used by tests that mutate the corpus."""
    from llenergymeasure.config.schema_loader import reset_schema_cache

    _corpus.cache_clear()
    reset_schema_cache()


def validate_engine_params_keys(engine: Engine | str, engine_params: BaseModel) -> None:
    """Vet every key declared under ``<engine>.engine_params`` against the corpus.

    Descends into nested blocks (typed sub-models and untyped mappings alike) as
    far as the corpus makes their structure visible.

    Args:
        engine: The engine whose ``engine_params`` block this is.
        engine_params: The declared ``engine_params`` sub-model.

    Raises:
        ValueError: A declared key is absent from the corpus-visible surface and
            that surface is closed. The message names the key, its full dotted
            path (so the engine is named), and a did-you-mean suggestion when one
            is close.
    """
    name = engine_str(engine)
    corpus = _corpus(name)
    if corpus is None:
        warnings.warn(
            f"no engine-knowledge corpus is installed for engine {name!r}, so keys under "
            f"{name}.engine_params cannot be checked against the engine surface; they are "
            "forwarded to the engine as written. Regenerate the engine schema to restore "
            "the check.",
            ConfigValidationWarning,
            stacklevel=2,
        )
    _check_block(
        engine_params,
        engine=name,
        path=f"{name}.engine_params",
        properties=corpus.properties if corpus is not None else {},
        definitions=corpus.definitions if corpus is not None else {},
        closed=corpus is not None and corpus.closed,
        extra_names=frozenset(ENGINES[Engine(name)].engine_params_extras),
    )


def _check_block(
    block: Any,
    *,
    engine: str,
    path: str,
    properties: Mapping[str, Any],
    definitions: Mapping[str, Any],
    closed: bool,
    extra_names: frozenset[str] = frozenset(),
    seen_refs: frozenset[str] = frozenset(),
) -> None:
    """Vet the keys declared on one block, then recurse into its nested blocks."""
    declared = _declared_keys(block)
    if declared is None:
        return

    vocabulary = set(properties) | extra_names
    if isinstance(block, BaseModel):
        vocabulary |= set(type(block).model_fields)
    if not vocabulary:
        # The corpus makes no structure visible at this depth (an opaque
        # passthrough block), so there is nothing to check the keys against and
        # every key below is admissible.
        return

    for key, value in declared.items():
        if key not in vocabulary:
            _report(engine=engine, path=path, key=key, vocabulary=vocabulary, closed=closed)
            # Nothing under an unrecognised key is checkable.
            continue
        child_properties, child_refs = _resolve_child(
            properties.get(key), definitions=definitions, seen_refs=seen_refs
        )
        _check_block(
            value,
            engine=engine,
            path=f"{path}.{key}",
            properties=child_properties,
            definitions=definitions,
            closed=closed,
            seen_refs=child_refs,
        )


def _declared_keys(block: Any) -> Mapping[str, Any] | None:
    """The keys a user actually wrote on ``block``, or None when it holds no keys.

    On a Pydantic block that is ``model_fields_set`` - the explicitly-supplied
    fields plus the ``extra="allow"`` extras - so an unset field with a default
    never counts as declared. On a plain mapping (a nested block the generated
    model leaves ``Any``-typed, so it arrives as a dict) it is the mapping itself.
    Non-string keys are dropped: they cannot name an engine field.
    """
    if isinstance(block, BaseModel):
        return {name: getattr(block, name, None) for name in block.model_fields_set}
    if isinstance(block, Mapping):
        return {key: value for key, value in block.items() if isinstance(key, str)}
    return None


def _resolve_child(
    spec: Any, *, definitions: Mapping[str, Any], seen_refs: frozenset[str]
) -> tuple[Mapping[str, Any], frozenset[str]]:
    """The properties the corpus makes visible one level under ``spec``.

    A nested block is a ``$ref`` into ``$defs``, either directly or inside an
    ``anyOf`` union alongside ``null``. A definition carrying ``properties``
    describes a structure worth descending into; anything else is opaque and
    yields ``{}``, which admits every key below it. Union members are merged rather
    than intersected, so resolution never narrows what the corpus admits.
    ``seen_refs`` carries the definitions already entered on this path, which stops
    a self-referential definition from recursing forever.
    """
    merged: dict[str, Any] = {}
    refs = set(seen_refs)
    for name in _referenced_definitions(spec):
        if name in refs:
            continue
        refs.add(name)
        definition = definitions.get(name)
        if not isinstance(definition, Mapping):
            continue
        nested = definition.get("properties")
        if isinstance(nested, Mapping):
            merged.update(nested)
    return merged, frozenset(refs)


def _referenced_definitions(spec: Any) -> list[str]:
    """Definition names ``spec`` references, following ``anyOf`` / ``oneOf`` unions."""
    if not isinstance(spec, Mapping):
        return []
    names: list[str] = []
    ref = spec.get("$ref")
    if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
        names.append(ref[len(_REF_PREFIX) :])
    for union_key in ("anyOf", "oneOf"):
        members = spec.get(union_key)
        if isinstance(members, list):
            for member in members:
                names.extend(_referenced_definitions(member))
    return names


def close_match_hint(key: str, vocabulary: Iterable[str]) -> str | None:
    """The ``"; did you mean <field>?"`` suffix for ``key``, or None if nothing is close.

    The single renderer for the did-you-mean tail every unrecognised-key
    diagnostic ends with, so the wording and the :data:`CLOSE_MATCH_CUTOFF`
    threshold stay the same whichever validator produced the message. Callers
    append the returned suffix to their own "unknown field ... in ..." stem, and
    a None result is what "no suggestion" looks like - several of them only warn
    when there IS a close match.

    ``vocabulary`` is sorted before matching so the suggestion is deterministic
    when several names tie on similarity.
    """
    suggestion = difflib.get_close_matches(key, sorted(vocabulary), n=1, cutoff=CLOSE_MATCH_CUTOFF)
    return f"; did you mean {suggestion[0]}?" if suggestion else None


def _report(*, engine: str, path: str, key: str, vocabulary: set[str], closed: bool) -> None:
    """Reject an unrecognised key on a closed surface; warn on a typo otherwise."""
    hint = close_match_hint(key, vocabulary)
    if not closed:
        if hint:
            warnings.warn(
                f"unknown field {key!r} in {path}{hint}",
                ConfigValidationWarning,
                stacklevel=2,
            )
        return
    raise ValueError(
        f"unknown field {key!r} in {path}{hint or '.'} Not on the {engine} engine surface "
        "(the discovered schema for the pinned engine version, plus the curated "
        "engine_params fields), so it configures nothing: a sweep over it expands into "
        "distinct-looking experiments that all measure the same engine configuration."
    )
