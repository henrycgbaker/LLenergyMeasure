"""Wave 2 Tier A improved-deterministic substrate (experiment-queue item 0.3).

Implements the 7 deterministic primitives proposed in
``findings/wave2_improved_det_primitives.md``. The hypothesis under test:
richer det primitives close 60-80% of the baseline-vs-ground-truth gap with
NO LLM call, by discovering classes + validators + env-vars structurally
instead of from a hand-curated landmark list.

The 7 primitives (all engine-agnostic where possible, tree-sitter substrate):

1. Universal class-fanout walker - discover EVERY class matching structural
   markers (@dataclass, msgspec.Struct, BaseModel/BaseSettings, __post_init__,
   @field_validator/@model_validator, _verify_* / validate* methods).
2. Env-var module enumerator - os.environ.get / os.getenv / lambda os.environ
   across the engine's envs module; extract name + default + behaviour comment;
   flag source-vs-stub footguns.
3. Silent-normalisation detector - if self.X: self.Y = Z ; if self.X is None:
   self.X = default ; self.X = self.X or default - across __post_init__.
4. Mechanical-gap probe - if X not in <set>: raise / if X in <bad>: raise within
   validator methods; resolve the membership constant where module-level.
5. Per-engine validator-naming convention walker - vllm _verify_*; transformers
   validate* / from_pretrained / save_pretrained; tensorrt __post_init__ /
   _validate_* / pydantic decorators.
6. Decorator-discovered validator walker - @field_validator / @model_validator /
   @validator / @root_validator; extract field(s), body raise/return, mode.
7. Aggregator __post_init__ deep walker - vllm VllmConfig, transformers
   GenerationConfig, tensorrt LlmArgs; sibling _verify_* calls, cross-field
   assignments, multi-field conditional raises.

Scoring contract: emitted ``invariants.proposed.yaml`` is scored by
``trial_scoring.invariant_identity`` which keys on
``(namespace, native_field, predicate_kind, secondary_field)`` derived from the
``match.fields`` dotted keys. We therefore emit ``match.fields`` under the same
dotted-namespace convention the reference catalogue uses (e.g.
``vllm.sampling.<field>``, ``vllm.engine.<field>``,
``transformers.sampling.<field>``, ``tensorrt.<field>``). The schema envelope is
scored on ``(engine_params | sampling_params | $defs.<class>, name)`` identities.

Tasks (matches a_treesitter):
* ``task="schema"``     -> emit schema only; empty invariants envelope.
* ``task="invariants"`` -> emit invariants only; empty schema envelope.
* ``task="both"`` (default) -> emit both.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from strategies.wave2._base import CellOutput, standard_out_dir

_STRATEGY_ID = "w2-a-improved-det"
_SCHEMA_VERSION = "2.0.0"
_INVARIANTS_SCHEMA_VERSION = "1.0.0"
_ADDED_BY = "wave2_a_improved_det"


# ---------------------------------------------------------------------------
# Per-engine declarative config. Kept narrow: file paths + namespace policy +
# env module. Class + validator DISCOVERY is structural (Primitives 1/5/6), so
# the long-tail config classes the baseline missed get picked up automatically.
# ---------------------------------------------------------------------------


_EngineCfg = dict[str, Any]


_ENGINE_CFG: dict[str, _EngineCfg] = {
    "vllm": {
        # Schema target classes -> envelope section.
        "engine_params": {"engine/arg_utils.py": ["EngineArgs"]},
        "sampling_params": {"sampling_params.py": ["SamplingParams"]},
        # $defs: the long-tail subconfig classes the baseline missed. Globbed so
        # the v0.19.1 config.py -> config/*.py subpackage split is covered
        # without re-pinning (generalised-glob primitive).
        "defs_files": ["config.py", "config/*.py"],
        # Files to scan for validator bodies + env enumeration.
        "validator_files": [
            "sampling_params.py",
            "pooling_params.py",
            "config.py",
            "config/*.py",
            "engine/arg_utils.py",
        ],
        "env_modules": ["envs.py"],
        # Aggregator class(es) for Primitive 7.
        "aggregators": [("config.py", "VllmConfig")],
        "engine_prefix": "vllm",
        "sampling_field_prefix": "vllm.sampling",
        "engine_field_prefix": "vllm.engine",
        # Reference invariant-namespace overrides (mirror the catalogue).
        "ns_overrides": {
            "SamplingParams": "vllm.sampling",
            "GuidedDecodingParams": "vllm.sampling",
            "BeamSearchParams": "vllm.sampling",
            "ModelConfig": "vllm.engine",
            "CacheConfig": "vllm.engine",
            "ParallelConfig": "vllm.engine",
            "SchedulerConfig": "vllm.engine",
            "DeviceConfig": "vllm.engine",
            "LoadConfig": "vllm.engine",
            "SpeculativeConfig": "vllm.engine",
            "LoRAConfig": "vllm.engine.lora",
            "PromptAdapterConfig": "vllm.engine.prompt_adapter",
            "ObservabilityConfig": "vllm.engine",
            "TokenizerPoolConfig": "vllm.engine.tokenizer",
            "VllmConfig": "vllm.engine",
            "CompilationConfig": "vllm.engine",
            "KVTransferConfig": "vllm.engine",
            "DecodingConfig": "vllm.engine",
            "EngineArgs": "vllm.engine",
        },
    },
    "transformers": {
        "engine_params": {
            "modeling_utils.py": ["PreTrainedModel"],
            "utils/quantization_config.py": ["BitsAndBytesConfig"],
        },
        "sampling_params": {
            "generation/configuration_utils.py": ["GenerationConfig"],
        },
        "defs_files": ["generation/configuration_utils.py"],
        "validator_files": [
            "generation/configuration_utils.py",
            "utils/quantization_config.py",
        ],
        "env_modules": ["utils/hub.py", "training_args.py"],
        "aggregators": [("generation/configuration_utils.py", "GenerationConfig")],
        "engine_prefix": "transformers",
        "sampling_field_prefix": "transformers.sampling",
        "engine_field_prefix": "transformers",
        "ns_overrides": {
            "GenerationConfig": "transformers.sampling",
            "CompileConfig": "transformers.sampling",
            "WatermarkingConfig": "transformers.sampling",
            "SynthIDTextWatermarkingConfig": "transformers.sampling",
            "BaseWatermarkingConfig": "transformers.sampling",
            "BitsAndBytesConfig": "transformers",
            "PreTrainedModel": "transformers",
        },
        # transformers uses validate* (no leading underscore) by convention.
        "validator_method_prefixes": ("validate", "_validate"),
    },
    "tensorrt": {
        "engine_params": {"llmapi/llm_args.py": ["TorchLlmArgs", "LlmArgs", "BaseLlmArgs"]},
        "sampling_params": {"sampling_params.py": ["SamplingParams"]},
        "defs_files": ["llmapi/llm_args.py", "llmapi/build_cache.py", "plugin/plugin.py"],
        # plugin/plugin.py is scanned for membership by Primitive 8 (the lever-1
        # fold) but kept OUT of validator_files: its only methods are a wildcard
        # logger + an SM-gate validate() whose body is all locals, which the
        # bare-field fallback would mis-mine as fields (e.g. sm, val).
        "validator_files": ["llmapi/llm_args.py", "sampling_params.py"],
        "env_modules": ["_common.py", "llmapi/llm_args.py"],
        "aggregators": [("llmapi/llm_args.py", "LlmArgs"), ("llmapi/llm_args.py", "TorchLlmArgs")],
        "engine_prefix": "tensorrt",
        "sampling_field_prefix": "tensorrt",
        "engine_field_prefix": "tensorrt",
        # tensorrt catalogue uses a flat tensorrt.<field> namespace.
        "ns_overrides": {},
        "flat_namespace": "tensorrt",
        "validator_method_prefixes": ("_validate", "validate"),
    },
}


# ---------------------------------------------------------------------------
# Source resolution (robust to the runner passing engine_source_root=None for
# tensorrt, whose unpacked dir is `tensorrt_llm` not `tensorrt`).
# ---------------------------------------------------------------------------


def _resolve_source_root(engine: str, version_slug: str, given: Path | None) -> Path | None:
    if given is not None and given.exists():
        return given
    pkg = {"vllm": "vllm", "transformers": "transformers", "tensorrt": "tensorrt_llm"}.get(engine)
    candidates = [
        Path(f"/tmp/trial_{engine}_{version_slug}_venv/src/{pkg}"),
        Path(f"/tmp/trial_{engine}_{version_slug}_venv/src/{engine}"),
        Path(f"/tmp/{engine}-unpacked/{pkg}"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Tree-sitter setup (reuse a_treesitter's pattern).
# ---------------------------------------------------------------------------


def _load_parser() -> tuple[Any, Any]:
    try:
        import tree_sitter
        import tree_sitter_python
    except ImportError as e:
        raise RuntimeError(
            "Wave 2 a_improved_det requires `tree-sitter` + `tree-sitter-python` in the wave2 venv."
        ) from e
    language = tree_sitter.Language(tree_sitter_python.language())
    parser = tree_sitter.Parser(language)
    return language, parser


def _read_source(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _expand_files(source_root: Path, patterns: list[str]) -> list[Path]:
    """Primitive: generalised subpackage glob. Expand path patterns (literal or
    glob) against the source root, de-duped and order-preserving. Globbing
    de-pins the landmark file list so subpackage refactors (e.g. vllm
    ``config.py`` -> ``config/*.py``) and per-platform module fan-out are picked
    up across bumps WITHOUT editing the config - never pin paths, they go
    stale."""
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(source_root.glob(pattern)) if "*" in pattern else [source_root / pattern]
        for p in matches:
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Generic AST helpers.
# ---------------------------------------------------------------------------


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def _class_nodes(root: Any) -> list[Any]:
    return [n for n in _walk(root) if n.type == "class_definition"]


def _class_name(source: bytes, class_node: Any) -> str | None:
    n = class_node.child_by_field_name("name")
    return _node_text(source, n) if n is not None else None


def _class_body(class_node: Any) -> Any | None:
    for child in class_node.children:
        if child.type == "block":
            return child
    return None


def _enclosing_class_name(source: bytes, node: Any) -> str | None:
    cur = node
    while cur is not None:
        if cur.type == "class_definition":
            n = cur.child_by_field_name("name")
            if n is not None:
                return _node_text(source, n)
        cur = cur.parent
    return None


def _enclosing_method_name(source: bytes, node: Any) -> str | None:
    cur = node
    while cur is not None:
        if cur.type == "function_definition":
            n = cur.child_by_field_name("name")
            if n is not None:
                return _node_text(source, n)
        cur = cur.parent
    return None


def _methods_of(source: bytes, class_node: Any) -> list[tuple[str, Any, list[str]]]:
    """Return [(method_name, function_def_node, decorator_names)] for a class.

    Handles both bare ``function_definition`` and ``decorated_definition``.
    """
    out: list[tuple[str, Any, list[str]]] = []
    body = _class_body(class_node)
    if body is None:
        return out
    for stmt in body.named_children:
        fn = None
        decs: list[str] = []
        if stmt.type == "function_definition":
            fn = stmt
        elif stmt.type == "decorated_definition":
            inner = stmt.child_by_field_name("definition")
            if inner is not None and inner.type == "function_definition":
                fn = inner
            for child in stmt.children:
                if child.type == "decorator":
                    decs.append(_node_text(source, child))
        if fn is None:
            continue
        name_node = fn.child_by_field_name("name")
        if name_node is None:
            continue
        out.append((_node_text(source, name_node), fn, decs))
    return out


def _base_class_text(source: bytes, class_node: Any) -> str:
    args = class_node.child_by_field_name("superclasses")
    return _node_text(source, args) if args is not None else ""


# ---------------------------------------------------------------------------
# Predicate-kind classification (aligns with trial_scoring.PREDICATE_KINDS).
# Allowed kinds the scorer recognises: exact | not_in | not_equal | gt | lt |
# ge | le | type_is_not | present | unknown.
# ---------------------------------------------------------------------------


def _collect_self_fields(node: Any, source: bytes) -> list[str]:
    """Return ordered, de-duped leaf names of `self.<x>` references in node."""
    out: list[str] = []
    for n in _walk(node):
        if n.type == "attribute":
            obj = n.child_by_field_name("object")
            attr = n.child_by_field_name("attribute")
            if obj is not None and attr is not None:
                if obj.type == "identifier" and _node_text(source, obj) == "self":
                    name = _node_text(source, attr)
                    if name not in out:
                        out.append(name)
    return out


_BARE_DENYLIST = frozenset(
    {
        "self",
        "cls",
        "True",
        "False",
        "None",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "len",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "tuple",
        "dict",
        "set",
        "any",
        "all",
        "min",
        "max",
        "type",
        "print",
        "range",
        "kwargs",
        "args",
        "arg",
        "and",
        "or",
        "not",
        "in",
    }
)


def _collect_bare_fields(node: Any, source: bytes) -> list[str]:
    out: list[str] = []
    for n in node.children if node.type != "identifier" else [node]:
        for m in _walk(n):
            if m.type == "attribute":
                continue
            if m.type == "identifier":
                t = _node_text(source, m)
                if t not in _BARE_DENYLIST and not t.startswith("_") and t not in out:
                    out.append(t)
    return out


def _classify_kind(cond_node: Any, source: bytes) -> str:
    """Classify an if/assert condition into the scorer's predicate taxonomy."""
    if cond_node is None:
        return "unknown"
    # Unwrap parentheses: `(self.x is not None and self.x < 1)`.
    if cond_node.type == "parenthesized_expression":
        for c in cond_node.named_children:
            return _classify_kind(c, source)
        return "unknown"
    t = cond_node.type
    text = _node_text(source, cond_node)
    if t == "boolean_operator":
        left = cond_node.child_by_field_name("left")
        right = cond_node.child_by_field_name("right")
        if left is not None:
            lk = _classify_kind(left, source)
            if lk in {"present", "unknown"} and right is not None:
                rk = _classify_kind(right, source)
                if rk != "unknown":
                    return rk
            if lk != "unknown":
                return lk
        if right is not None:
            return _classify_kind(right, source)
        return "unknown"
    if t == "not_operator":
        operand = cond_node.child_by_field_name("argument")
        if operand is not None:
            ok = _classify_kind(operand, source)
            # `not isinstance(...)` -> type guard; `not (a <= x <= b)` -> range.
            if ok == "type_is_not" or "isinstance" in text:
                return "type_is_not"
            return ok if ok != "unknown" else "unknown"
    if t == "comparison_operator":
        if " not in " in text:
            return "not_in"
        # `X is None` / `X is not None` -> presence gate.
        if re.search(r"\bis\s+(not\s+)?None\b", text):
            return "present"
        # `X is False` / `X is True` / `X is not True` -> identity check; the
        # reference catalogue collapses these to `exact`.
        if re.search(r"\bis\s+(not\s+)?(True|False)\b", text):
            return "exact"
        if " is not " in text:
            return "present"
        # chained range `a <= x <= b` reads as multiple ops; treat as present-ish
        ops = [c for c in cond_node.children if not c.is_named]
        op_texts = [_node_text(source, c) for c in ops]
        if "==" in op_texts:
            return "exact"
        if "!=" in op_texts:
            return "not_equal"
        if ">=" in op_texts:
            return "ge"
        if "<=" in op_texts:
            return "le"
        if ">" in op_texts:
            return "gt"
        if "<" in op_texts:
            return "lt"
        if " in " in text:
            return "exact"
        return "unknown"
    if t == "call":
        fn = cond_node.child_by_field_name("function")
        if fn is not None and _node_text(source, fn) == "isinstance":
            return "type_is_not"
        return "present"
    if t in {"attribute", "identifier"}:
        return "present"
    return "unknown"


def _predicate_value(kind: str) -> Any:
    """Build the match-field value shape so the scorer re-derives `kind`."""
    if kind == "present":
        return {"present": True}
    if kind == "not_in":
        return {"present": True, "not_in": []}
    if kind == "type_is_not":
        return {"type_is_not": ""}
    op = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "not_equal": "not_equal"}
    if kind in op:
        return {op[kind]: 0}
    if kind == "exact":
        return 0
    return {}


# ---------------------------------------------------------------------------
# Primitive 1 + 5 + 6: universal class + validator discovery.
# ---------------------------------------------------------------------------


_CLASS_MARKER_DECORATORS = ("dataclass",)
_STRUCT_BASES = ("msgspec.Struct", "Struct", "BaseModel", "BaseSettings")
_VALIDATOR_DEFAULT_PREFIXES = ("_verify_", "verify_", "validate", "_validate")
# from_pretrained / save_pretrained mark a class as config-like (Primitive 1)
# but their bodies are HTTP / path I/O, not field predicates - we never mine
# their bodies for invariants (would pollute identity emission with os/token/
# is_local noise). Body-mining is restricted to _BODY_VALIDATOR names.
_VALIDATOR_EXACT = frozenset(
    {"__post_init__", "post_init", "model_post_init", "from_pretrained", "save_pretrained"}
)
_NON_BODY_VALIDATORS = frozenset({"from_pretrained", "save_pretrained"})
_DECORATOR_VALIDATORS = ("field_validator", "model_validator", "validator", "root_validator")


def _discover_classes(source: bytes, class_nodes: list[Any], cfg: _EngineCfg) -> dict[str, Any]:
    """Primitive 1: return {class_name: class_node} for every structurally
    config-like class (dataclass / Struct / BaseModel / has __post_init__ /
    has a validator-named method / has a decorator validator)."""
    prefixes = cfg.get("validator_method_prefixes", _VALIDATOR_DEFAULT_PREFIXES)
    found: dict[str, Any] = {}
    for cls in class_nodes:
        name = _class_name(source, cls)
        if name is None or name.startswith("_"):
            continue
        is_config = False
        # decorator marker
        if cls.parent is not None and cls.parent.type == "decorated_definition":
            dec_text = _node_text(source, cls.parent)
            if any(d in dec_text for d in _CLASS_MARKER_DECORATORS):
                is_config = True
        # base-class marker
        base_text = _base_class_text(source, cls)
        if any(b in base_text for b in _STRUCT_BASES):
            is_config = True
        # method markers
        for mname, _fn, decs in _methods_of(source, cls):
            if mname in _VALIDATOR_EXACT or mname.startswith(tuple(prefixes)):
                is_config = True
            if any(dv in d for d in decs for dv in _DECORATOR_VALIDATORS):
                is_config = True
        if is_config:
            found[name] = cls
    return found


def _validator_methods(
    source: bytes, class_node: Any, cfg: _EngineCfg
) -> list[tuple[str, Any, list[str]]]:
    """Primitives 5+6: return validator methods of a class (by name convention
    OR by decorator)."""
    prefixes = cfg.get("validator_method_prefixes", _VALIDATOR_DEFAULT_PREFIXES)
    out: list[tuple[str, Any, list[str]]] = []
    for mname, fn, decs in _methods_of(source, class_node):
        is_validator = (
            mname in _VALIDATOR_EXACT
            or mname.startswith(tuple(prefixes))
            or any(dv in d for d in decs for dv in _DECORATOR_VALIDATORS)
        )
        if is_validator:
            out.append((mname, fn, decs))
    return out


def _decorator_fields(source: bytes, decs: list[str]) -> list[str]:
    """Primitive 6: pull the validated field name(s) from a
    @field_validator('a', 'b') / @model_validator decorator."""
    fields: list[str] = []
    for d in decs:
        if not any(dv in d for dv in _DECORATOR_VALIDATORS):
            continue
        for m in re.findall(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""", d):
            if m not in fields and m not in {"before", "after", "wrap", "plain"}:
                fields.append(m)
    return fields


# ---------------------------------------------------------------------------
# Namespace resolution.
# ---------------------------------------------------------------------------


def _resolve_namespace(class_name: str, cfg: _EngineCfg) -> str:
    if cfg.get("flat_namespace"):
        return cfg["flat_namespace"]
    overrides: dict[str, str] = cfg.get("ns_overrides", {})
    if class_name in overrides:
        return overrides[class_name]
    return cfg["engine_field_prefix"]


# ---------------------------------------------------------------------------
# Invariant emission helper.
# ---------------------------------------------------------------------------


def _make_invariant(
    *,
    engine: str,
    class_name: str,
    method_name: str,
    namespace: str,
    primary_field: str,
    kind: str,
    secondary_field: str,
    rel_path: str,
    line: int,
    severity: str,
    fired_by: str,
) -> dict[str, Any]:
    primary_key = f"{namespace}.{primary_field}" if namespace else primary_field
    match_fields: dict[str, Any] = {primary_key: _predicate_value(kind)}
    inv_id = f"{engine}_{class_name.lower()}_{method_name}_{primary_field}_{kind}".lower()
    if secondary_field:
        sec_key = f"{namespace}.{secondary_field}" if namespace else secondary_field
        match_fields[sec_key] = _predicate_value(kind)
        inv_id += f"_{secondary_field}".lower()
    return {
        "id": inv_id,
        "engine": engine,
        "library": engine,
        "invariant_under_test": (f"{class_name}.{method_name} flags `{primary_field}` ({kind})"),
        "severity": severity,
        "native_type": f"{engine}.{class_name}",
        "miner_source": {"path": rel_path, "method": method_name, "line_at_scan": line},
        "match": {"engine": engine, "fields": match_fields},
        "added_by": _ADDED_BY,
        "fired_by": fired_by,
    }


# ---------------------------------------------------------------------------
# Primitives 3 + 4 + 5 + 6 + 7: validator-body walk producing invariants.
# ---------------------------------------------------------------------------


def _soft_signal(block: Any, source: bytes) -> bool:
    """A consequence block that warns / logs is a behavioural invariant too."""
    if block is None:
        return False
    for stmt in _walk(block):
        if stmt.type == "call":
            fn = stmt.child_by_field_name("function")
            if fn is not None and _node_text(source, fn) in {
                "warnings.warn",
                "logger.warning",
                "logger.warning_once",
                "logger.info",
                "warn",
                "logging.warning",
            }:
                return True
    return False


def _consequence_has_raise(block: Any) -> bool:
    if block is None:
        return False
    return any(n.type == "raise_statement" for n in _walk(block))


def _enclosing_if_self_fields(node: Any, source: bytes, stop_fn: Any) -> list[str]:
    """Walk up the enclosing if/elif chain (within the method) and collect the
    outer conditions' self.<x> fields, outermost first. Used for cross-field
    invariant emission (outer field gates inner field)."""
    out: list[str] = []
    cur = node.parent
    chain: list[Any] = []
    while cur is not None and cur is not stop_fn:
        if cur.type in {"if_statement", "elif_clause"}:
            chain.append(cur)
        cur = cur.parent
    for if_node in reversed(chain):
        cond = if_node.child_by_field_name("condition")
        if cond is None:
            continue
        for f in _collect_self_fields(cond, source):
            if f not in out:
                out.append(f)
    return out


def _soft_sink_field(block: Any, source: bytes) -> str | None:
    """Return the field name a soft sink writes to:
    minor_issues["X"] = ... (string-literal key) or
    getattr(self, arg) inside a loop. None if not extractable."""
    if block is None:
        return None
    for n in _walk(block):
        if n.type == "assignment":
            left = n.child_by_field_name("left")
            if left is not None and left.type == "subscript":
                val = left.child_by_field_name("value")
                sub = left.child_by_field_name("subscript")
                if val is not None and _node_text(source, val) in {
                    "minor_issues",
                    "issues",
                    "warnings_dict",
                }:
                    if sub is not None and sub.type == "string":
                        return _node_text(source, sub).strip("'\"")
    return None


def _walk_validator_body(
    *,
    engine: str,
    cfg: _EngineCfg,
    class_name: str,
    method_name: str,
    decorator_fields: list[str],
    fn_node: Any,
    source: bytes,
    rel_path: str,
    fired: dict[str, bool],
) -> list[dict[str, Any]]:
    """Walk a single validator method body and emit invariants.

    Covers Primitive 4 (membership-gate raises), Primitive 3 (silent
    normalisation), Primitive 5/6 (any if->raise / if->warn) and decorator
    field fallback (Primitive 6)."""
    namespace = _resolve_namespace(class_name, cfg)
    invs: list[dict[str, Any]] = []
    seen_local: set[tuple[str, str, str]] = set()

    body = fn_node.child_by_field_name("body")
    if body is None:
        return invs

    # if/elif -> raise|warn  (Primitives 4 + 5 + 7).
    for node in _walk(body):
        cond = None
        block = None
        if node.type == "if_statement" or node.type == "elif_clause":
            cond = node.child_by_field_name("condition")
            block = node.child_by_field_name("consequence")
        elif node.type == "assert_statement":
            named = [c for c in node.named_children]
            cond = named[0] if named else None
            block = None
        else:
            continue

        has_raise = node.type == "assert_statement" or _consequence_has_raise(block)
        has_soft = _soft_signal(block, source) or _soft_sink_field(block, source) is not None
        if not (has_raise or has_soft):
            continue
        if cond is None:
            continue

        kind = _classify_kind(cond, source)
        local_fields = _collect_self_fields(cond, source)
        # Pydantic @field_validator('a','b') def f(cls, v): if v <op> ...: the
        # validated fields are NAMED by the decorator, and the body compares the
        # value-param `v` (not self.X). Emit one invariant per decorator field
        # carrying the body's predicate kind (Primitive 6).
        if not local_fields and decorator_fields:
            severity_dec = "error" if has_raise else "warning"
            for df in decorator_fields:
                key = (df, kind, "")
                if key in seen_local:
                    continue
                seen_local.add(key)
                fired["p6"] = True
                invs.append(
                    _make_invariant(
                        engine=engine,
                        class_name=class_name,
                        method_name=method_name,
                        namespace=namespace,
                        primary_field=df,
                        kind=kind,
                        secondary_field="",
                        rel_path=rel_path,
                        line=node.start_point[0] + 1,
                        severity=severity_dec,
                        fired_by="p6",
                    )
                )
            continue
        if not local_fields:
            local_fields = _collect_bare_fields(cond, source)
        local_fields = [f for f in local_fields if not f.startswith("__")]

        # Outer-if-chain fields (cross-field gate, e.g. do_sample gates temperature).
        outer_fields = _enclosing_if_self_fields(node, source, fn_node)
        outer_fields = [f for f in outer_fields if f not in local_fields and not f.startswith("__")]

        # Soft sink can name the inner field directly (minor_issues["X"] = ...)
        # when the condition self-field is absent or is a generic gate.
        sink_field = _soft_sink_field(block, source)

        severity = "error" if has_raise else "warning"

        # Same-field guard collapse: when the outer gate field ALSO bears the
        # substantive comparison (e.g. `if self.best_of is not None: if
        # self.best_of < self.n: raise`), the reference flattens to a single
        # invariant on that field with the SUBSTANTIVE kind - it is not a
        # genuine two-field cross-check. Drop such fields from the gate set.
        outer_fields = [f for f in outer_fields if f not in local_fields]

        # Case A: cross-field gate. The outermost gate field is PRIMARY (with
        # the gate's own predicate kind), the local/sink field is SECONDARY.
        if outer_fields:
            primary = outer_fields[0]
            # recompute the primary kind from the outermost gate condition.
            gate_kind = kind
            cur = node.parent
            outermost = None
            while cur is not None and cur is not fn_node:
                if cur.type in {"if_statement", "elif_clause"}:
                    outermost = cur
                cur = cur.parent
            if outermost is not None:
                gc = outermost.child_by_field_name("condition")
                if gc is not None:
                    gf = _collect_self_fields(gc, source)
                    if gf and gf[0] == primary:
                        gate_kind = _classify_kind(gc, source)
            inner_candidates = local_fields or ([sink_field] if sink_field else [])
            for sec in inner_candidates:
                if sec == primary:
                    continue
                key = (primary, gate_kind, sec)
                if key in seen_local:
                    continue
                seen_local.add(key)
                fired["p7"] = True
                invs.append(
                    _make_invariant(
                        engine=engine,
                        class_name=class_name,
                        method_name=method_name,
                        namespace=namespace,
                        primary_field=primary,
                        kind=gate_kind,
                        secondary_field=sec,
                        rel_path=rel_path,
                        line=node.start_point[0] + 1,
                        severity=severity,
                        fired_by="p7",
                    )
                )

        # Case B: single-field emission. The reference keys a comparison
        # `self.X <op> <rhs>` on the SUBSTANTIVE field (X) only; a `self.X <op>
        # self.Y` field-vs-field comparison still keys on X alone (Y is encoded
        # as the rhs sentinel, not a second identity slot). So we emit ONE
        # single-field invariant on the field bearing the substantive predicate
        # (the first local self-field that is not a pure presence gate).
        emit_fields = local_fields or ([sink_field] if sink_field else [])
        if emit_fields and not outer_fields:
            primary = emit_fields[0]
            key = (primary, kind, "")
            if key not in seen_local:
                seen_local.add(key)
                fk = "p4" if kind in {"not_in", "type_is_not"} else "p5"
                fired[fk] = True
                invs.append(
                    _make_invariant(
                        engine=engine,
                        class_name=class_name,
                        method_name=method_name,
                        namespace=namespace,
                        primary_field=primary,
                        kind=kind,
                        secondary_field="",
                        rel_path=rel_path,
                        line=node.start_point[0] + 1,
                        severity=severity,
                        fired_by=fk,
                    )
                )

    # Silent normalisation (Primitive 3): if self.X is None: self.X = default ;
    # self.X = self.X or default ; if self.X: self.Y = Z.
    for node in _walk(body):
        if node.type != "assignment":
            continue
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or left.type != "attribute":
            continue
        obj = left.child_by_field_name("object")
        attr = left.child_by_field_name("attribute")
        if obj is None or _node_text(source, obj) != "self" or attr is None:
            continue
        target = _node_text(source, attr)
        if target.startswith("__"):
            continue
        is_norm = False
        if right is not None and right.type == "boolean_operator":
            # self.X = self.X or default
            if " or " in _node_text(source, node):
                is_norm = True
        # if self.X is None: self.X = default  (assignment inside an if-None block)
        enclosing_if = node
        while enclosing_if is not None and enclosing_if.type not in {
            "if_statement",
            "function_definition",
        }:
            enclosing_if = enclosing_if.parent
        if enclosing_if is not None and enclosing_if.type == "if_statement":
            cond = enclosing_if.child_by_field_name("condition")
            if cond is not None and " is None" in _node_text(source, cond):
                is_norm = True
        if not is_norm:
            continue
        key = (target, "present", "")
        if key in seen_local:
            continue
        seen_local.add(key)
        fired["p3"] = True
        namespace = _resolve_namespace(class_name, cfg)
        invs.append(
            _make_invariant(
                engine=engine,
                class_name=class_name,
                method_name=method_name,
                namespace=namespace,
                primary_field=target,
                kind="present",
                secondary_field="",
                rel_path=rel_path,
                line=node.start_point[0] + 1,
                severity="dormant",
                fired_by="p3",
            )
        )

    return invs


# ---------------------------------------------------------------------------
# Primitive 2: env-var enumerator.
# ---------------------------------------------------------------------------


_ENV_CALL_RE = re.compile(r"""os\.(?:environ\.get|getenv)\(\s*['"]([A-Z][A-Z0-9_]*)['"]""")
_ENV_SUBSCRIPT_RE = re.compile(r"""os\.environ\[\s*['"]([A-Z][A-Z0-9_]*)['"]\s*\]""")


def _enumerate_env_vars(text: str) -> tuple[list[dict[str, Any]], int]:
    """Primitive 2: find env-var reads + source-vs-stub footguns.

    Returns (entries, footgun_count). Each entry carries name + default + the
    declared key (for footgun detection where the lambda key != the read var).
    """
    entries: dict[str, dict[str, Any]] = {}
    for m in _ENV_CALL_RE.finditer(text):
        name = m.group(1)
        entries.setdefault(name, {"name": name, "x-source": "os.getenv/environ.get"})
    for m in _ENV_SUBSCRIPT_RE.finditer(text):
        name = m.group(1)
        entries.setdefault(name, {"name": name, "x-source": "os.environ[...]"})

    # Source-vs-stub footgun: a dict literal key "VLLM_X": lambda ... os.environ
    # [..."VLLM_Y"...] where X != Y. Detect lines mapping a quoted attr key to a
    # lambda whose read var differs.
    footguns = 0
    for km in re.finditer(
        r"""['"]([A-Z][A-Z0-9_]*)['"]\s*:\s*lambda[^\n]*?os\.(?:environ\.get|getenv|environ)\[?\(?\s*['"]([A-Z][A-Z0-9_]*)['"]""",
        text,
    ):
        attr_key, read_key = km.group(1), km.group(2)
        if attr_key != read_key:
            footguns += 1
            entries.setdefault(attr_key, {"name": attr_key, "x-source": "lambda"})
            entries[attr_key]["x-footgun"] = f"attr {attr_key} reads {read_key}"
    return list(entries.values()), footguns


# ---------------------------------------------------------------------------
# Schema field extraction (reuse a_treesitter shape, simplified).
# ---------------------------------------------------------------------------


def _classify_type_text(type_text: str) -> Any:
    text = type_text.strip()
    is_opt = False
    if " | None" in text or text.endswith("| None"):
        is_opt = True
        text = text.replace(" | None", "").replace("| None", "").strip()
    if text.startswith("Optional[") and text.endswith("]"):
        is_opt = True
        text = text[len("Optional[") : -1].strip()
    prim = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "bytes": "string",
        "list": "array",
        "List": "array",
        "tuple": "array",
        "Tuple": "array",
        "dict": "object",
        "Dict": "object",
        "set": "array",
        "Set": "array",
        "None": "null",
        "NoneType": "null",
        "Any": "any",
    }
    head = text.split("[", 1)[0].strip()
    if head in prim:
        base = prim[head]
        return [base, "null"] if is_opt else base
    suffix = " | None" if is_opt else ""
    return {"description": f"Upstream type: {type_text.strip()}{suffix}"}


def _parse_default(default_text: str) -> Any:
    d = default_text.strip()
    if d in {"None", ""}:
        return None
    if d == "True":
        return True
    if d == "False":
        return False
    try:
        if "." in d or "e" in d or "E" in d:
            return float(d)
        return int(d)
    except ValueError:
        pass
    if (d.startswith('"') and d.endswith('"')) or (d.startswith("'") and d.endswith("'")):
        return d[1:-1]
    if d in {"[]", "list()"}:
        return []
    if d in {"{}", "dict()"}:
        return {}
    return d


def _extract_signature_fields(source: bytes, params_node: Any) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for param in params_node.named_children:
        fname = ftype = fdef = None
        if param.type == "typed_default_parameter":
            n = param.child_by_field_name("name")
            t = param.child_by_field_name("type")
            v = param.child_by_field_name("value")
            fname = _node_text(source, n) if n else None
            ftype = _node_text(source, t) if t else None
            fdef = _node_text(source, v) if v else None
        elif param.type == "typed_parameter":
            ids = [c for c in param.children if c.type == "identifier"]
            t = param.child_by_field_name("type")
            fname = _node_text(source, ids[0]) if ids else None
            ftype = _node_text(source, t) if t else None
        elif param.type == "default_parameter":
            n = param.child_by_field_name("name")
            v = param.child_by_field_name("value")
            fname = _node_text(source, n) if n else None
            fdef = _node_text(source, v) if v else None
        elif param.type == "identifier":
            fname = _node_text(source, param)
        else:
            continue
        if not fname or fname in {"self", "cls"} or fname.startswith("_"):
            continue
        entry: dict[str, Any] = {}
        if ftype is not None:
            entry["type"] = _classify_type_text(ftype)
        if fdef is not None:
            entry["default"] = _parse_default(fdef)
        if entry:
            fields[fname] = entry
    return fields


def _extract_class_fields(source: bytes, class_node: Any) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    body = _class_body(class_node)
    if body is None:
        return fields
    for stmt in body.named_children:
        if stmt.type != "expression_statement":
            continue
        for child in stmt.named_children:
            if child.type != "assignment":
                continue
            name_node = child.child_by_field_name("left")
            type_node = child.child_by_field_name("type")
            right_node = child.child_by_field_name("right")
            if name_node is None or name_node.type != "identifier":
                continue
            fname = _node_text(source, name_node)
            if fname.startswith("_"):
                continue
            entry: dict[str, Any] = {}
            if type_node is not None:
                entry["type"] = _classify_type_text(_node_text(source, type_node))
            if right_node is not None:
                entry["default"] = _parse_default(_node_text(source, right_node))
            if entry:
                fields[fname] = entry
    # __init__ signature + kwargs.pop pairs.
    for mname, fn, _decs in _methods_of(source, class_node):
        if mname != "__init__":
            continue
        params = fn.child_by_field_name("parameters")
        if params is not None:
            for k, v in _extract_signature_fields(source, params).items():
                fields.setdefault(k, v)
        body_node = fn.child_by_field_name("body")
        if body_node is not None:
            for k, v in _extract_kwargs_pop(source, body_node).items():
                fields.setdefault(k, v)
    return fields


def _extract_kwargs_pop(source: bytes, node: Any) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for n in _walk(node):
        if n.type != "assignment":
            continue
        left = n.child_by_field_name("left")
        right = n.child_by_field_name("right")
        if left is None or left.type != "attribute" or right is None or right.type != "call":
            continue
        obj = left.child_by_field_name("object")
        if obj is None or _node_text(source, obj) != "self":
            continue
        fn_node = right.child_by_field_name("function")
        args_node = right.child_by_field_name("arguments")
        if fn_node is None or fn_node.type != "attribute" or args_node is None:
            continue
        fn_obj = fn_node.child_by_field_name("object")
        fn_attr = fn_node.child_by_field_name("attribute")
        if (
            fn_obj is None
            or _node_text(source, fn_obj) != "kwargs"
            or fn_attr is None
            or _node_text(source, fn_attr) != "pop"
        ):
            continue
        named = [c for c in args_node.named_children if c.type != "comment"]
        if not named or named[0].type != "string":
            continue
        key = _node_text(source, named[0]).strip("'\"")
        entry: dict[str, Any] = {"x-source": "kwargs_pop"}
        if len(named) >= 2:
            entry["default"] = _parse_default(_node_text(source, named[1]))
        fields[key] = entry
    return fields


def _extract_method_sig(
    source: bytes, class_node: Any, method_name: str
) -> dict[str, dict[str, Any]]:
    for mname, fn, _decs in _methods_of(source, class_node):
        if mname == method_name:
            params = fn.child_by_field_name("parameters")
            if params is not None:
                return _extract_signature_fields(source, params)
    return {}


# ---------------------------------------------------------------------------
# Build phases.
# ---------------------------------------------------------------------------


def _build_schema(
    *, engine: str, version: str, source_root: Path, parser: Any, cfg: _EngineCfg
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    obs: list[str] = []
    counts = {"env_footguns": 0}
    schema: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "engine_params": {},
        "sampling_params": {},
        "$defs": {},
    }

    def parse(rel_path: str) -> tuple[bytes, Any] | None:
        src = _read_source(source_root / rel_path)
        if not src:
            obs.append(f"missing source file: {rel_path}")
            return None
        return src, parser.parse(src).root_node

    # engine_params + sampling_params.
    for section, file_map in (
        ("engine_params", cfg["engine_params"]),
        ("sampling_params", cfg["sampling_params"]),
    ):
        for rel_path, class_names in file_map.items():
            parsed = parse(rel_path)
            if parsed is None:
                continue
            src, root = parsed
            cls_index = {_class_name(src, c): c for c in _class_nodes(root)}
            for cn in class_names:
                node = cls_index.get(cn)
                if node is None:
                    continue
                schema[section].update(_extract_class_fields(src, node))
                # for transformers PreTrainedModel pull from_pretrained sig too.
                if cn == "PreTrainedModel":
                    schema[section].update(_extract_method_sig(src, node, "from_pretrained"))
            break  # one matching class file per section is enough for first match

    # Primitive 1 long-tail $defs: every discovered config class in defs_files
    # (glob-expanded so subpackage splits are covered without re-pinning).
    for defs_path in _expand_files(source_root, cfg.get("defs_files", [])):
        rel_path = str(defs_path.relative_to(source_root))
        parsed = parse(rel_path)
        if parsed is None:
            continue
        src, root = parsed
        discovered = _discover_classes(src, _class_nodes(root), cfg)
        for cname, node in discovered.items():
            if cname in schema["$defs"]:
                continue
            props = _extract_class_fields(src, node)
            if props:
                schema["$defs"][cname] = {"type": "object", "properties": props}

    # Primitive 2 env-vars (kept under engine_envs so it doesn't pollute schema
    # recall identities, but recorded for the env-coverage finding).
    env_entries: list[dict[str, Any]] = []
    for rel_path in cfg.get("env_modules", []):
        src = _read_source(source_root / rel_path)
        if not src:
            continue
        entries, footguns = _enumerate_env_vars(src.decode("utf-8", errors="replace"))
        env_entries.extend(entries)
        counts["env_footguns"] += footguns
    # de-dup by name
    seen_env: dict[str, dict[str, Any]] = {}
    for e in env_entries:
        seen_env.setdefault(e["name"], e)
    if seen_env:
        schema["engine_envs"] = {"_count": len(seen_env), **seen_env}
    counts["env_count"] = len(seen_env)

    if not schema["$defs"]:
        del schema["$defs"]
    return schema, obs, counts


def _build_invariants(
    *, engine: str, version: str, source_root: Path, parser: Any, cfg: _EngineCfg
) -> tuple[dict[str, Any], list[str], dict[str, bool]]:
    obs: list[str] = []
    fired: dict[str, bool] = {f"p{i}": False for i in range(1, 8)}
    all_invs: list[dict[str, Any]] = []
    seen_global: set[tuple] = set()

    validator_paths = _expand_files(source_root, cfg["validator_files"])
    if not validator_paths:
        obs.append(f"no validator source files matched {cfg['validator_files']}")
    for path in validator_paths:
        src = _read_source(path)
        if not src:
            continue
        rel_path = str(path.relative_to(source_root))
        root = parser.parse(src).root_node
        discovered = _discover_classes(src, _class_nodes(root), cfg)
        if discovered:
            fired["p1"] = True
        for cname, class_node in discovered.items():
            validators = _validator_methods(src, class_node, cfg)
            for mname, fn, decs in validators:
                if mname in _NON_BODY_VALIDATORS:
                    continue
                if any(dv in d for d in decs for dv in _DECORATOR_VALIDATORS):
                    fired["p6"] = True
                dec_fields = _decorator_fields(src, decs)
                invs = _walk_validator_body(
                    engine=engine,
                    cfg=cfg,
                    class_name=cname,
                    method_name=mname,
                    decorator_fields=dec_fields,
                    fn_node=fn,
                    source=src,
                    rel_path=rel_path,
                    fired=fired,
                )
                for inv in invs:
                    # de-dup by (sorted fields, predicate shape)
                    sig = (
                        tuple(sorted(inv["match"]["fields"].keys())),
                        json.dumps(
                            {k: inv["match"]["fields"][k] for k in inv["match"]["fields"]},
                            sort_keys=True,
                        ),
                    )
                    if sig in seen_global:
                        continue
                    seen_global.add(sig)
                    all_invs.append(inv)

    # Primitive 7 marker: aggregator classes that carry __post_init__.
    for rel_path, agg_name in cfg.get("aggregators", []):
        src = _read_source(source_root / rel_path)
        if not src:
            continue
        root = parser.parse(src).root_node
        for c in _class_nodes(root):
            if _class_name(src, c) == agg_name:
                for mname, _fn, _decs in _methods_of(src, c):
                    if mname in {"__post_init__", "model_post_init"}:
                        fired["p7"] = True

    envelope = {
        "schema_version": _INVARIANTS_SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "mined_at": datetime.now(timezone.utc).isoformat(),
        "invariants": all_invs,
    }
    return envelope, obs, fired


def _empty_schema(engine: str, version: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "engine_params": {},
        "sampling_params": {},
    }


def _empty_invariants(engine: str, version: str) -> dict[str, Any]:
    return {
        "schema_version": _INVARIANTS_SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "mined_at": datetime.now(timezone.utc).isoformat(),
        "invariants": [],
    }


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def run_cell(
    *,
    engine: str,
    version: str,
    engine_source_root: Path | None = None,
    task: str = "both",
) -> CellOutput:
    """Execute one Wave 2 a-improved-det cell.

    engine: transformers | vllm | tensorrt.
    version: e.g. "v0.7.3" or "v0_7_3".
    engine_source_root: unpacked engine source dir; auto-resolved if None.
    task: "schema" | "invariants" | "both".
    """
    if engine not in _ENGINE_CFG:
        raise NotImplementedError(
            f"a_improved_det config missing for engine={engine!r}; known: {sorted(_ENGINE_CFG)}"
        )
    if task not in {"schema", "invariants", "both"}:
        raise ValueError(f"task must be schema|invariants|both; got {task!r}")
    cfg = _ENGINE_CFG[engine]

    version_slug = (
        version
        if version.startswith("v") and "." not in version
        else f"v{version.lstrip('v').replace('.', '_')}"
    )
    source_root = _resolve_source_root(engine, version_slug, engine_source_root)
    out_dir = standard_out_dir(_STRATEGY_ID, engine, version_slug)
    observations: list[str] = []
    wall: dict[str, float] = {}
    extra: dict[str, Any] = {}

    t_total = time.perf_counter()
    _language, parser = _load_parser()

    if source_root is None:
        observations.append(
            f"engine source root unresolved for {engine}/{version_slug}; "
            "emitting empty envelopes (degraded)."
        )
        schema_env = _empty_schema(engine, version)
        inv_env = _empty_invariants(engine, version)
    else:
        # Schema phase.
        t = time.perf_counter()
        if task in {"schema", "both"}:
            schema_env, s_obs, s_counts = _build_schema(
                engine=engine,
                version=version,
                source_root=source_root,
                parser=parser,
                cfg=cfg,
            )
            observations.extend(f"schema: {o}" for o in s_obs)
            extra.update(s_counts)
        else:
            schema_env = _empty_schema(engine, version)
        wall["schema"] = time.perf_counter() - t

        # Invariants phase.
        t = time.perf_counter()
        if task in {"invariants", "both"}:
            inv_env, i_obs, fired = _build_invariants(
                engine=engine,
                version=version,
                source_root=source_root,
                parser=parser,
                cfg=cfg,
            )
            observations.extend(f"invariants: {o}" for o in i_obs)
            # Primitive 2 (env enumerator) fires in the schema phase; surface it
            # in the fired set so the cost-frontier report is accurate.
            if extra.get("env_count", 0) > 0:
                fired["p2"] = True
            extra["primitives_fired"] = sorted(k for k, v in fired.items() if v)
        else:
            inv_env = _empty_invariants(engine, version)
        wall["invariants"] = time.perf_counter() - t

    schema_path = out_dir / "schema.json"
    invariants_path = out_dir / "invariants.proposed.yaml"
    schema_path.write_text(json.dumps(schema_env, indent=2, default=str))
    invariants_path.write_text(yaml.safe_dump(inv_env, sort_keys=False))

    wall["total"] = time.perf_counter() - t_total

    extra.update(
        {
            "task": task,
            "source_root": str(source_root) if source_root else None,
            "engine_param_count": len(schema_env.get("engine_params", {})),
            "sampling_param_count": len(schema_env.get("sampling_params", {})),
            "defs_count": len(schema_env.get("$defs", {})),
            "invariant_count": len(inv_env.get("invariants", [])),
        }
    )

    return CellOutput(
        strategy_id=_STRATEGY_ID,
        engine=engine,
        engine_version=version,
        schema_path=schema_path,
        invariants_path=invariants_path,
        observations=observations,
        wall_sec=wall,
        extra=extra,
    )


__all__ = ["run_cell"]
