"""Wave 2 Tier A universal tree-sitter substrate (WAVE2_PROTOCOL section 3 Tier A).

Hypothesis: a universal tree-sitter query walker replaces the per-engine
handwritten AST walkers from Wave 1 (~1200 LoC each) with one query set,
trading some recall for an order of magnitude less code per engine.

Two tasks, scored independently:

* Task 1 -- schema discovery. Class definitions + fields + types + defaults.
  Reference: ``engine_versions/<engine>/v<v>/outputs/schema.discovered.json``.
  Prior: tree-sitter is well-suited here. Class/typed-assignment/parameter
  shapes are pure syntax.
* Task 2 -- invariant mining. Validator predicates extracted from method
  bodies (e.g. ``if num_beams > num_return_sequences: raise``).
  Reference: ``engine_versions/<engine>/v<v>/outputs/invariants.validated.yaml``.
  Prior: tree-sitter sees syntax, NOT semantics. Whenever a predicate
  references ``self.foo.bar`` -- the common case in real engines -- the
  walker can't resolve ``bar`` to a known sampling/engine field without
  semantic information. The probe measures how high the wall is.

The ``task`` kwarg lets the scorer hit each task independently:

* ``task="schema"``    -> emit only schema.json; empty invariants envelope.
* ``task="invariants"`` -> emit only invariants.proposed.yaml; empty schema.
* ``task="both"`` (default) -> emit both.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from strategies.wave2._base import CellOutput, standard_out_dir

# ---------------------------------------------------------------------------
# Engine target tables (which classes drive engine_params / sampling_params /
# $defs vs which classes are scanned for validator bodies). Sourced from
# engine_versions/<engine>/<v>/outputs/schema.discovered.json conventions.
# ---------------------------------------------------------------------------


_EngineTargets = dict[str, Any]


# Source-relative file paths to scan, plus class-name maps. Kept narrow on
# purpose: the trial measures the recall the universal walker achieves with
# a small, declarative target list -- NOT how cleverly the walker discovers
# targets. Discovery-by-AST is a separate axis.
_ENGINE_TARGETS: dict[str, _EngineTargets] = {
    "transformers": {
        # engine_params: PreTrainedModel.from_pretrained signature
        # parameters + BitsAndBytesConfig.__init__ untyped defaults. The
        # reference also pulls fields from kwargs docstring blocks (Sphinx
        # ":param X:" lines) which tree-sitter doesn't see -- that's the
        # measured gap.
        "engine_params_classes": {
            "modeling_utils.py": ["PreTrainedModel"],
            "utils/quantization_config.py": ["BitsAndBytesConfig"],
        },
        # sampling_params: GenerationConfig surface. Fields live in
        # `self.<x> = kwargs.pop("x", default)` pairs inside __init__.
        "sampling_params_classes": {
            "generation/configuration_utils.py": ["GenerationConfig"],
        },
        # $defs: companion config classes referenced from the sampling class.
        "defs_classes": {
            "generation/configuration_utils.py": ["CompileConfig"],
        },
        # Validator-body files: where invariant predicates live.
        "validator_files": [
            "generation/configuration_utils.py",
            "utils/quantization_config.py",
        ],
        # Engine prefix for invariant identity (matches reference namespace
        # convention: "transformers.sampling.<field>").
        "invariant_engine_prefix": "transformers",
        "sampling_field_prefix": "transformers.sampling",
        "engine_field_prefix": "transformers.engine",
        # For from_pretrained-style methods: scan the named method's
        # parameters list (not just __init__).
        "engine_signature_methods": {
            "modeling_utils.py": [("PreTrainedModel", "from_pretrained")],
        },
        # Class-name -> invariant-namespace overrides. transformers
        # reference uses `transformers.sampling.<field>` for GenerationConfig
        # and `transformers.<field>` (empty post-prefix) for the bnb fields:
        # the reference identity tuple is ("transformers", "load_in_4bit", ...).
        "invariant_class_namespace_overrides": {
            "GenerationConfig": "transformers.sampling",
            "BitsAndBytesConfig": "transformers",
            "CompileConfig": "transformers.sampling",
            "WatermarkingConfig": "transformers.sampling",
            "SynthIDTextWatermarkingConfig": "transformers.sampling",
            "BaseWatermarkingConfig": "transformers.sampling",
        },
    },
    "vllm": {
        "engine_params_classes": {
            "engine/arg_utils.py": ["EngineArgs"],
        },
        "sampling_params_classes": {
            "sampling_params.py": ["SamplingParams"],
        },
        "defs_classes": {},
        "validator_files": [
            "sampling_params.py",
            "config.py",
            "engine/arg_utils.py",
        ],
        "invariant_engine_prefix": "vllm",
        "sampling_field_prefix": "vllm.sampling",
        "engine_field_prefix": "vllm.engine",
        "engine_signature_methods": {},
        # Class-name -> namespace overrides for invariant identity (so
        # ModelConfig / CacheConfig / ParallelConfig predicates land under
        # the reference's `vllm.engine.<field>` namespace, not under each
        # class's own prefix).
        "invariant_class_namespace_overrides": {
            "ModelConfig": "vllm.engine",
            "CacheConfig": "vllm.engine",
            "ParallelConfig": "vllm.engine",
            "SchedulerConfig": "vllm.engine",
            "DeviceConfig": "vllm.engine",
            "LoadConfig": "vllm.engine",
            "LoRAConfig": "vllm.engine.lora",
            "PromptAdapterConfig": "vllm.engine.prompt_adapter",
            "ObservabilityConfig": "vllm.engine",
            "TokenizerPoolConfig": "vllm.engine.tokenizer",
            "VllmConfig": "vllm.engine",
            "CompilationConfig": "vllm.engine",
            "KVTransferConfig": "vllm.engine",
            "EngineArgs": "vllm.engine",
        },
    },
}


_STRATEGY_ID = "w2-a-treesitter"
_SCHEMA_VERSION = "2.0.0"
_INVARIANTS_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Tree-sitter setup
# ---------------------------------------------------------------------------


def _load_parser() -> tuple[Any, Any]:
    """Return (language, parser). Fails loud if the deps are missing."""
    try:
        import tree_sitter
        import tree_sitter_python
    except ImportError as e:
        raise RuntimeError(
            "Wave 2 a_treesitter requires `tree-sitter` + `tree-sitter-python` in the wave2 venv."
        ) from e
    language = tree_sitter.Language(tree_sitter_python.language())
    parser = tree_sitter.Parser(language)
    return language, parser


# ---------------------------------------------------------------------------
# Query strings -- the iteration surface. Keep them clustered + named so the
# probe doc can refer to specific queries by id.
# ---------------------------------------------------------------------------


# Schema queries (Task 1).
_Q_CLASS_DEF = "(class_definition name: (identifier) @class_name body: (block) @class_body)"

# Typed dataclass-style field assignment inside a class body.
# Matches:   foo: int = 3   /   bar: str = "x"   /   baz: list[int] = []
_Q_TYPED_FIELD = """
(class_definition
  body: (block
    (expression_statement
      (assignment
        left: (identifier) @field_name
        type: (type) @field_type
        right: (_) @field_default))))
"""

# Untyped dataclass-style field assignment (less common, but transformers'
# GenerationConfig __init__ writes `self.foo = kwargs.pop(...)`).
_Q_INIT_PARAM = """
(class_definition
  body: (block
    (function_definition
      name: (identifier) @method_name
      parameters: (parameters) @params)))
"""

# Subscript-typed default parameter inside any function's parameter list.
# Captures `foo: int | None = None` and `bar: list[str] = []` -- the common
# parameter-as-field pattern in Pydantic v2-style classes and EngineArgs.
_Q_TYPED_DEFAULT = """
(typed_default_parameter
  name: (identifier) @param_name
  type: (type) @param_type
  value: (_) @param_default)
"""

# Bare typed param (no default) -- e.g. `foo: int` in a signature.
_Q_TYPED_PARAM = """
(typed_parameter
  (identifier) @param_name
  type: (type) @param_type)
"""

# Invariant queries (Task 2).

# Round 1 baseline: any if -> raise.
_Q_IF_RAISE = """
(if_statement
  condition: (_) @cond
  consequence: (block
    (raise_statement
      (call function: (_) @exc) @raise)))
"""

# Round 2: comparison-operator conditions specifically (narrower, higher precision).
_Q_IF_CMP_RAISE = """
(if_statement
  condition: (comparison_operator) @cond
  consequence: (block (raise_statement) @raise))
"""

# Round 2: validator-decorator detection.
_Q_VALIDATOR_DECORATOR = """
(decorated_definition
  (decorator
    (call
      function: (identifier) @decorator_name
      (#match? @decorator_name "^(field_validator|validator|model_validator)$")))
  definition: (function_definition
    name: (identifier) @validator_name
    body: (block) @validator_body))
"""

# Round 3: assertion patterns -- `assert <cond>, "msg"` inside validators.
_Q_ASSERT = """
(assert_statement
  (_) @cond
  (_)? @msg) @assertion
"""

# Round 3: tuple-of-fields predicate (very common in Pydantic root validators).
# E.g. `if self.do_sample and self.temperature is not None:`
_Q_BOOL_OP_RAISE = """
(if_statement
  condition: (boolean_operator) @cond
  consequence: (block
    (raise_statement) @raise))
"""

# Round 4: `not isinstance(...)` raise -- type guard pattern.
_Q_TYPE_GUARD = """
(if_statement
  condition: (unary_operator
    operand: (call
      function: (identifier) @check
      (#eq? @check "isinstance")
      arguments: (argument_list (_) @target (_) @type_ref)))
  consequence: (block (raise_statement) @raise))
"""


# ---------------------------------------------------------------------------
# Schema extraction (Task 1)
# ---------------------------------------------------------------------------


def _read_source(path: Path) -> bytes:
    """Read file as bytes; return empty bytes if missing."""
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _node_text(source: bytes, node: Any) -> str:
    """Decode the source bytes spanned by ``node`` to UTF-8 text."""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _classify_type_text(type_text: str) -> Any:
    """Best-effort JSON-Schema-ish type mapping from a Python type expression."""
    text = type_text.strip()
    # Strip trailing | None to detect Optional shape.
    is_optional = False
    if " | None" in text or text.endswith("| None"):
        is_optional = True
        text = text.replace(" | None", "").replace("| None", "").strip()
    if text.startswith("Optional[") and text.endswith("]"):
        is_optional = True
        text = text[len("Optional[") : -1].strip()

    primitive_map = {
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
    # list[...] / dict[...] / etc. -- map by leading head.
    head = text.split("[", 1)[0].strip()
    if head in primitive_map:
        base = primitive_map[head]
        if is_optional:
            return [base, "null"]
        return base
    # Unknown -> object with hint.
    if is_optional:
        return {"description": f"Upstream type: {type_text.strip()} | None"}
    return {"description": f"Upstream type: {type_text.strip()}"}


def _parse_default(default_text: str) -> Any:
    """Best-effort literal eval for default values. Falls back to raw text."""
    default_text = default_text.strip()
    if default_text in {"None", ""}:
        return None
    if default_text == "True":
        return True
    if default_text == "False":
        return False
    # Try numeric literal.
    try:
        if "." in default_text or "e" in default_text or "E" in default_text:
            return float(default_text)
        return int(default_text)
    except ValueError:
        pass
    # Strip string quotes.
    if (default_text.startswith('"') and default_text.endswith('"')) or (
        default_text.startswith("'") and default_text.endswith("'")
    ):
        return default_text[1:-1]
    # Empty list/dict shorthand.
    if default_text in {"[]", "list()"}:
        return []
    if default_text in {"{}", "dict()"}:
        return {}
    # Give up: return text as-is.
    return default_text


def _find_class_node(language: Any, root: Any, class_name: str) -> Any | None:
    """Return the class_definition node whose name matches, or None."""
    query = language.query(_Q_CLASS_DEF)
    captures = query.captures(root)
    name_nodes = captures.get("class_name", [])
    for node in name_nodes:
        # Walk up to the class_definition node.
        parent = node.parent
        if parent is None:
            continue
        if node.text.decode("utf-8", errors="replace") == class_name:
            return parent
    return None


def _extract_signature_fields(source: bytes, params_node: Any) -> dict[str, dict[str, Any]]:
    """Pull every parameter from a function signature.

    Captures typed_default_parameter / typed_parameter / default_parameter
    (untyped with default) / identifier (bare). Skips self / cls /
    *args / **kwargs splats.
    """
    fields: dict[str, dict[str, Any]] = {}
    for param in params_node.named_children:
        field_name = None
        type_text = None
        default_text = None
        if param.type == "typed_default_parameter":
            name_node = param.child_by_field_name("name")
            type_node = param.child_by_field_name("type")
            value_node = param.child_by_field_name("value")
            if name_node is not None:
                field_name = _node_text(source, name_node)
            if type_node is not None:
                type_text = _node_text(source, type_node)
            if value_node is not None:
                default_text = _node_text(source, value_node)
        elif param.type == "typed_parameter":
            id_children = [c for c in param.children if c.type == "identifier"]
            type_node = param.child_by_field_name("type")
            if id_children:
                field_name = _node_text(source, id_children[0])
            if type_node is not None:
                type_text = _node_text(source, type_node)
        elif param.type == "default_parameter":
            # Untyped parameter with default: `foo=1`. Captures
            # BitsAndBytesConfig.__init__ style.
            name_node = param.child_by_field_name("name")
            value_node = param.child_by_field_name("value")
            if name_node is not None:
                field_name = _node_text(source, name_node)
            if value_node is not None:
                default_text = _node_text(source, value_node)
        elif param.type == "identifier":
            # Bare parameter with no type, no default.
            field_name = _node_text(source, param)
        elif param.type in {"list_splat_pattern", "dictionary_splat_pattern"}:
            continue
        else:
            continue
        if not field_name or field_name in {"self", "cls"}:
            continue
        if field_name.startswith("_"):
            continue
        entry: dict[str, Any] = {}
        if type_text is not None:
            entry["type"] = _classify_type_text(type_text)
        if default_text is not None:
            entry["default"] = _parse_default(default_text)
        if entry:
            fields[field_name] = entry
    return fields


def _extract_kwargs_pop_fields(source: bytes, init_node: Any) -> dict[str, dict[str, Any]]:
    """Pull `self.foo = kwargs.pop("foo", default)` pairs from an __init__.

    Critical for transformers.GenerationConfig: ~70 sampling fields live in
    this pattern and not as typed annotations.
    """
    fields: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if (
                left is not None
                and left.type == "attribute"
                and right is not None
                and right.type == "call"
            ):
                obj_node = left.child_by_field_name("object")
                attr_node = left.child_by_field_name("attribute")
                fn_node = right.child_by_field_name("function")
                args_node = right.child_by_field_name("arguments")
                if (
                    obj_node is not None
                    and obj_node.type == "identifier"
                    and _node_text(source, obj_node) == "self"
                    and attr_node is not None
                    and fn_node is not None
                    and fn_node.type == "attribute"
                    and args_node is not None
                ):
                    fn_obj = fn_node.child_by_field_name("object")
                    fn_attr = fn_node.child_by_field_name("attribute")
                    if (
                        fn_obj is not None
                        and fn_obj.type == "identifier"
                        and _node_text(source, fn_obj) == "kwargs"
                        and fn_attr is not None
                        and _node_text(source, fn_attr) == "pop"
                    ):
                        # Field name is the string-literal arg; default is the second arg.
                        named_args = [c for c in args_node.named_children if c.type != "comment"]
                        if named_args:
                            key_node = named_args[0]
                            if key_node.type == "string":
                                key_text = _node_text(source, key_node).strip("'\"")
                                entry: dict[str, Any] = {}
                                if len(named_args) >= 2:
                                    entry["default"] = _parse_default(
                                        _node_text(source, named_args[1])
                                    )
                                entry["x-source"] = "kwargs_pop"
                                fields[key_text] = entry
        for child in node.children:
            walk(child)

    walk(init_node)
    return fields


def _extract_class_fields(
    language: Any, source: bytes, class_node: Any
) -> dict[str, dict[str, Any]]:
    """Pull typed-field assignments + __init__ typed parameters + kwargs.pop
    pairs from a class.

    Field order preserved by walking the AST top-down. Pop pattern fields
    are added LAST so an explicit typed annotation wins over the kwargs.pop
    fallback.
    """
    fields: dict[str, dict[str, Any]] = {}

    # Find the body block child.
    body_node = None
    for child in class_node.children:
        if child.type == "block":
            body_node = child
            break
    if body_node is None:
        return fields

    # Direct typed assignments at class scope.
    for stmt in body_node.named_children:
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
            field_name = _node_text(source, name_node)
            if field_name.startswith("_") and field_name != "_from_model_config":
                continue
            entry: dict[str, Any] = {}
            if type_node is not None:
                type_text = _node_text(source, type_node)
                entry["type"] = _classify_type_text(type_text)
            if right_node is not None:
                entry["default"] = _parse_default(_node_text(source, right_node))
            if entry:
                fields[field_name] = entry

    # __init__ signature + kwargs.pop walk.
    init_node = None
    for stmt in body_node.named_children:
        if stmt.type == "function_definition":
            method_name_node = stmt.child_by_field_name("name")
            if method_name_node is not None and _node_text(source, method_name_node) == "__init__":
                init_node = stmt
                break
    if init_node is not None:
        params_node = init_node.child_by_field_name("parameters")
        if params_node is not None:
            sig_fields = _extract_signature_fields(source, params_node)
            for k, v in sig_fields.items():
                fields.setdefault(k, v)
        # kwargs.pop pairs inside the init body.
        pop_fields = _extract_kwargs_pop_fields(source, init_node)
        for k, v in pop_fields.items():
            fields.setdefault(k, v)

    return fields


def _extract_method_signature_fields(
    language: Any, source: bytes, class_node: Any, method_name: str
) -> dict[str, dict[str, Any]]:
    """Pull signature parameters from a NAMED method of a class.

    Used for transformers PreTrainedModel.from_pretrained: parameters that
    appear in from_pretrained's signature become engine_params.
    """
    body_node = None
    for child in class_node.children:
        if child.type == "block":
            body_node = child
            break
    if body_node is None:
        return {}
    method_node = None
    for stmt in body_node.named_children:
        if stmt.type == "function_definition":
            n = stmt.child_by_field_name("name")
            if n is not None and _node_text(source, n) == method_name:
                method_node = stmt
                break
        # decorated_definition also wraps function_definition.
        if stmt.type == "decorated_definition":
            inner = stmt.child_by_field_name("definition")
            if inner is not None and inner.type == "function_definition":
                n = inner.child_by_field_name("name")
                if n is not None and _node_text(source, n) == method_name:
                    method_node = inner
                    break
    if method_node is None:
        return {}
    params_node = method_node.child_by_field_name("parameters")
    if params_node is None:
        return {}
    return _extract_signature_fields(source, params_node)


def _build_schema(
    *,
    engine: str,
    version: str,
    source_root: Path,
    parser: Any,
    language: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Build the schema envelope by parsing each target class out of source."""
    targets = _ENGINE_TARGETS[engine]
    observations: list[str] = []

    schema: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "engine_params": {},
        "sampling_params": {},
        "$defs": {},
    }

    for section_key, section_classes in (
        ("engine_params", targets["engine_params_classes"]),
        ("sampling_params", targets["sampling_params_classes"]),
        ("$defs", targets["defs_classes"]),
    ):
        for rel_path, class_names in section_classes.items():
            full_path = source_root / rel_path
            source = _read_source(full_path)
            if not source:
                observations.append(f"missing source file: {rel_path}")
                continue
            tree = parser.parse(source)
            for class_name in class_names:
                class_node = _find_class_node(language, tree.root_node, class_name)
                if class_node is None:
                    observations.append(f"class {class_name} not found in {rel_path}")
                    continue
                fields = _extract_class_fields(language, source, class_node)
                if section_key == "$defs":
                    schema["$defs"][class_name] = {
                        "type": "object",
                        "properties": fields,
                    }
                else:
                    schema[section_key].update(fields)

    # Signature methods (e.g. transformers PreTrainedModel.from_pretrained).
    for rel_path, class_method_pairs in targets.get("engine_signature_methods", {}).items():
        full_path = source_root / rel_path
        source = _read_source(full_path)
        if not source:
            observations.append(f"missing signature source: {rel_path}")
            continue
        tree = parser.parse(source)
        for class_name, method_name in class_method_pairs:
            class_node = _find_class_node(language, tree.root_node, class_name)
            if class_node is None:
                observations.append(f"signature class {class_name} not found in {rel_path}")
                continue
            sig_fields = _extract_method_signature_fields(language, source, class_node, method_name)
            for k, v in sig_fields.items():
                schema["engine_params"].setdefault(k, v)

    if not schema["$defs"]:
        del schema["$defs"]

    return schema, observations


# ---------------------------------------------------------------------------
# Invariant extraction (Task 2)
# ---------------------------------------------------------------------------


def _enclosing_method_name(language: Any, source: bytes, node: Any) -> str | None:
    """Walk up the tree to the nearest enclosing function_definition; return its name."""
    cur = node
    while cur is not None:
        if cur.type == "function_definition":
            name_node = cur.child_by_field_name("name")
            if name_node is not None:
                return _node_text(source, name_node)
        cur = cur.parent
    return None


def _enclosing_class_name(language: Any, source: bytes, node: Any) -> str | None:
    """Walk up to the nearest enclosing class_definition; return its name."""
    cur = node
    while cur is not None:
        if cur.type == "class_definition":
            name_node = cur.child_by_field_name("name")
            if name_node is not None:
                return _node_text(source, name_node)
        cur = cur.parent
    return None


def _collect_attribute_refs(node: Any, source: bytes) -> list[str]:
    """Find all `self.<x>` (or chained `self.x.y`) attribute references inside node.

    Returns the leaf identifier name, e.g. `self.num_beams` -> "num_beams".
    For chained `self.foo.bar`, returns "foo" (the first hop off self) because
    that's the field-name candidate; "bar" is semantically a property OF foo.
    """
    out: list[str] = []
    self_attrs: list[Any] = []

    def walk(n: Any) -> None:
        if n.type == "attribute":
            obj_node = n.child_by_field_name("object")
            attr_node = n.child_by_field_name("attribute")
            if obj_node is not None and attr_node is not None:
                if obj_node.type == "identifier" and _node_text(source, obj_node) == "self":
                    self_attrs.append(attr_node)
        for child in n.children:
            walk(child)

    walk(node)
    for a in self_attrs:
        out.append(_node_text(source, a))
    return out


# Reserved Python identifiers / common keywords + builtins that show up as
# bare identifier references inside predicates but are NEVER field
# candidates. Used by _collect_bare_identifier_refs as a denylist so the
# parameter-reference scan doesn't pollute identity emission.
_BARE_REF_DENYLIST: frozenset[str] = frozenset(
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
    }
)


def _collect_bare_identifier_refs(node: Any, source: bytes) -> list[str]:
    """Return bare identifier names referenced in a predicate node.

    Used to detect parameter-name predicates like `if load_in_4bit and
    load_in_8bit: raise` where the field name is a function parameter,
    not a self attribute. We filter out builtins / reserved keywords via
    `_BARE_REF_DENYLIST` so the noise floor stays low.
    """
    out: list[str] = []
    seen: set[str] = set()

    def walk(n: Any) -> None:
        # Don't recurse into attribute children (we'd grab the `self`).
        if n.type == "identifier":
            text = _node_text(source, n)
            if text not in _BARE_REF_DENYLIST and not text.startswith("_") and text not in seen:
                seen.add(text)
                out.append(text)
            return
        if n.type == "attribute":
            return  # attribute walks owned by _collect_attribute_refs.
        for child in n.children:
            walk(child)

    walk(node)
    return out


def _classify_predicate_kind_from_text(cond_text: str) -> str:
    """Heuristic classifier mapping a condition text to the canonical
    predicate-kind taxonomy used by trial_scoring.invariant_identity.

    Recognised kinds (from PREDICATE_KINDS):
      exact | not_in | not_equal | gt | lt | ge | le | type_is_not | present | unknown.
    """
    t = cond_text.strip()
    if " not in " in t:
        return "not_in"
    if "!=" in t:
        return "not_equal"
    if ">=" in t:
        return "ge"
    if "<=" in t:
        return "le"
    # > but not >= ; same for < .
    if ">" in t and ">=" not in t:
        return "gt"
    if "<" in t and "<=" not in t:
        return "lt"
    if "isinstance" in t and ("not " in t or t.startswith("not ")):
        return "type_is_not"
    if " is None" in t or " is not None" in t:
        return "present"
    if "==" in t:
        return "exact"
    return "unknown"


def _find_all_if_statements(node: Any) -> list[Any]:
    """Recursive walk; return every if_statement node."""
    out: list[Any] = []
    if node.type == "if_statement":
        out.append(node)
    for child in node.children:
        out.extend(_find_all_if_statements(child))
    return out


def _if_has_raise(node: Any) -> bool:
    """True iff the if_statement's consequence block directly contains a raise."""
    block = node.child_by_field_name("consequence")
    if block is None:
        return False
    for child in block.named_children:
        if child.type == "raise_statement":
            return True
    return False


def _if_consequence_block(node: Any) -> Any | None:
    """Return the block node attached as the if_statement's consequence."""
    return node.child_by_field_name("consequence")


def _if_has_soft_signal(node: Any, source: bytes) -> bool:
    """True iff the if's consequence pushes to a known "soft-raise" sink.

    Patterns recognised:
        minor_issues["foo"] = ...
        warnings.warn(...)
        logger.warning(...)
    These are morally invariants the library announces at runtime.
    """
    block = _if_consequence_block(node)
    if block is None:
        return False
    for stmt in block.named_children:
        # Assignment to a subscript on a known soft-signal name.
        if stmt.type == "expression_statement":
            for child in stmt.named_children:
                if child.type == "assignment":
                    left = child.child_by_field_name("left")
                    if left is not None and left.type == "subscript":
                        obj = left.child_by_field_name("value")
                        if obj is not None and obj.type == "identifier":
                            name = _node_text(source, obj)
                            if name in {"minor_issues", "warnings_dict", "issues"}:
                                return True
        # Call to a soft-signal function.
        if stmt.type == "expression_statement":
            for child in stmt.named_children:
                if child.type == "call":
                    fn = child.child_by_field_name("function")
                    if fn is None:
                        continue
                    text = _node_text(source, fn)
                    if text in {
                        "warnings.warn",
                        "logger.warning",
                        "logger.warning_once",
                        "logger.info",
                        "warn",
                    }:
                        return True
    return False


def _classify_predicate_kind_from_node(cond_node: Any, source: bytes) -> str:
    """Tighter predicate-kind classifier that uses the AST shape, not just text.

    Falls back to the text-based heuristic for forms not yet AST-classified.
    """
    # boolean_operator: combinator like `and` / `or`. The reference's
    # convention is to use the SUBSTANTIVE (numeric/value) predicate as the
    # kind; presence checks (is None / is not None) are gating, not the
    # actual invariant. So: if the LEFT side is a presence check, defer
    # to the right.
    if cond_node.type == "boolean_operator":
        left = cond_node.child_by_field_name("left")
        right = cond_node.child_by_field_name("right")
        if left is not None:
            left_kind = _classify_predicate_kind_from_node(left, source)
            if left_kind == "present" and right is not None:
                return _classify_predicate_kind_from_node(right, source)
            return left_kind
    if cond_node.type == "comparison_operator":
        # Scan the operator slot. tree-sitter exposes operators as either
        # anonymous children (==, >, <, !=, ...) or named keyword-pair
        # children (is, is_not, in, not in).
        cond_text = _node_text(source, cond_node)
        # `is not` -- captured as a multi-token; check by text.
        if " is not " in cond_text:
            return "present"
        if " is None" in cond_text:
            return "present"
        if " is " in cond_text and " is not " not in cond_text:
            # `X is False` / `X is True` -> identity check; collapse to exact.
            return "exact"
        if " not in " in cond_text:
            return "not_in"
        for child in cond_node.children:
            op_text = _node_text(source, child)
            if op_text == "==":
                return "exact"
            if op_text == "!=":
                return "not_equal"
            if op_text == ">":
                return "gt"
            if op_text == "<":
                return "lt"
            if op_text == ">=":
                return "ge"
            if op_text == "<=":
                return "le"
            if op_text == "in":
                return "exact"
        return _classify_predicate_kind_from_text(_node_text(source, cond_node))
    if cond_node.type == "not_operator":
        operand = cond_node.child_by_field_name("argument")
        if operand is not None and operand.type == "call":
            fn = operand.child_by_field_name("function")
            if fn is not None and _node_text(source, fn) == "isinstance":
                return "type_is_not"
    if cond_node.type == "call":
        fn = cond_node.child_by_field_name("function")
        if fn is not None and _node_text(source, fn) == "isinstance":
            return "type_is_not"
    if cond_node.type == "attribute" or cond_node.type == "identifier":
        # Bare truthy predicate. Treat as "present".
        return "present"
    return _classify_predicate_kind_from_text(_node_text(source, cond_node))


def _enclosing_if_chain(node: Any) -> list[Any]:
    """Return the chain of enclosing if_statement nodes, outermost first.

    Used to detect cross-field invariants where the outer condition gates
    an inner predicate.
    """
    out: list[Any] = []
    cur = node.parent
    while cur is not None:
        if cur.type == "if_statement":
            out.append(cur)
        cur = cur.parent
    return list(reversed(out))


def _extract_invariants_from_file(
    *,
    engine: str,
    rel_path: str,
    source: bytes,
    language: Any,
    parser: Any,
    engine_targets: _EngineTargets,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the invariant queries against ``source`` and emit invariant records.

    Strategy:
    1. Run _Q_IF_RAISE; for each (cond, raise) capture pair, walk back to the
       enclosing class + method.
    2. From the condition node's tree, collect `self.<attr>` references; the
       FIRST such attribute is the candidate native field.
    3. Classify the predicate kind from condition text.
    4. Emit one record per (class, method, field, kind) tuple.
    5. Also run _Q_VALIDATOR_DECORATOR; for each decorated validator, treat
       the function name's matched-field-list as the candidate native fields
       (if extractable from the decorator args; else skip).
    """
    observations: list[str] = []
    invariants: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    tree = parser.parse(source)

    sampling_classes = set()
    for names in engine_targets["sampling_params_classes"].values():
        sampling_classes.update(names)
    engine_classes = set()
    for names in engine_targets["engine_params_classes"].values():
        engine_classes.update(names)

    # Allowlist of class names whose validator bodies should be scanned.
    # Restricts emission to the classes the reference's invariant catalogue
    # actually targets -- skips e.g. transformers' 20+ quantization config
    # classes (AqlmConfig / AwqConfig / GPTQConfig / ...) which the
    # reference does not enumerate.
    allowlist: set[str] = set()
    for cls in sampling_classes:
        allowlist.add(cls)
    for cls in engine_classes:
        allowlist.add(cls)
    for cls in engine_targets.get("defs_classes", {}).get(rel_path, []):
        allowlist.add(cls)
    # Override map keys also count as in-scope.
    overrides_keys = set(engine_targets.get("invariant_class_namespace_overrides", {}).keys())
    allowlist.update(overrides_keys)

    sampling_prefix = engine_targets["sampling_field_prefix"]
    engine_prefix = engine_targets["engine_field_prefix"]
    namespace_overrides: dict[str, str] = engine_targets.get(
        "invariant_class_namespace_overrides", {}
    )

    def _resolve_namespace(class_name: str) -> str:
        """Resolve the canonical invariant namespace for a class.

        Order: class-name override -> sampling/engine class list -> generic
        engine.<class>. The override map handles real engines where the
        reference catalogue uses a unified namespace (e.g. all of vllm's
        ModelConfig / CacheConfig / ParallelConfig fields land under
        `vllm.engine`, not under each individual class).
        """
        if class_name in namespace_overrides:
            return namespace_overrides[class_name]
        if class_name in sampling_classes:
            return sampling_prefix
        if class_name in engine_classes:
            return engine_prefix
        return f"{engine_targets['invariant_engine_prefix']}.{class_name}"

    # Pass 0: detect if-elif-elif-else-raise type guards. The pattern is
    # `if X is None: ... elif isinstance(X, T1): ... elif isinstance(X, T2):
    # ... else: raise`. Emit one type_is_not invariant per target field.
    for if_node in _find_all_if_statements(tree.root_node):
        # Only top-of-chain if_statements; don't double-count elif sub-chains.
        if if_node.parent is not None and if_node.parent.type == "elif_clause":
            continue
        else_node = None
        # alternatives can chain; find any else_clause.
        for child in if_node.children:
            if child.type == "else_clause":
                else_node = child
                break
            if child.type == "elif_clause":
                # else may live inside the chain; keep searching.
                continue
        if else_node is None:
            continue
        else_body = else_node.child_by_field_name("body")
        if else_body is None:
            continue
        has_else_raise = any(c.type == "raise_statement" for c in else_body.named_children)
        if not has_else_raise:
            continue
        # Collect isinstance target field from any elif call.
        target_field: str | None = None
        for child in if_node.children:
            if child.type != "elif_clause":
                continue
            cond = child.child_by_field_name("condition")
            if cond is None or cond.type != "call":
                continue
            fn = cond.child_by_field_name("function")
            if fn is None or _node_text(source, fn) != "isinstance":
                continue
            args = cond.child_by_field_name("arguments")
            if args is None:
                continue
            named = [c for c in args.named_children if c.type != "comment"]
            if not named:
                continue
            tgt = named[0]
            target_field = _node_text(source, tgt)
            break
        # Fallback: pull target from the head `if X is None:` cond.
        if target_field is None:
            head_cond = if_node.child_by_field_name("condition")
            if head_cond is not None and head_cond.type == "comparison_operator":
                for c in head_cond.children:
                    if c.is_named and c.type == "identifier":
                        target_field = _node_text(source, c)
                        break
                    if c.is_named and c.type == "attribute":
                        attrs = _collect_attribute_refs(c, source)
                        if attrs:
                            target_field = attrs[0]
                            break
        if not target_field:
            continue
        # Strip `self.` if present.
        if target_field.startswith("self."):
            target_field = target_field.split(".", 1)[1]
        class_name = _enclosing_class_name(language, source, if_node)
        if class_name is None or class_name not in allowlist:
            continue
        namespace = _resolve_namespace(class_name)
        key = (namespace, target_field, "type_is_not", "")
        if key in seen:
            continue
        seen.add(key)
        method_name = _enclosing_method_name(language, source, if_node) or "__init__"
        line = if_node.start_point[0] + 1
        primary_key = f"{namespace}.{target_field}" if namespace else target_field
        invariants.append(
            {
                "id": f"{engine}_{class_name.lower()}_{method_name}_{target_field}_type_is_not".lower(),
                "engine": engine,
                "library": engine,
                "invariant_under_test": (
                    f"{class_name}.{method_name} type-guards `{target_field}` via elif-chain"
                ),
                "severity": "error",
                "native_type": f"{engine}.{class_name}",
                "miner_source": {
                    "path": rel_path,
                    "method": method_name,
                    "line_at_scan": line,
                },
                "match": {
                    "engine": engine,
                    "fields": {primary_key: {"present": True, "type_is_not": ""}},
                },
                "added_by": "wave2_a_treesitter",
            }
        )

    # Pass 1: every if_statement; classify whether it carries an invariant
    # signal (direct raise OR soft sink like `minor_issues[...] = ...`).
    # For each signal, compute primary/secondary fields by combining the
    # enclosing-if chain's self.* references.
    for if_node in _find_all_if_statements(tree.root_node):
        cond_node = if_node.child_by_field_name("condition")
        if cond_node is None:
            continue
        has_raise = _if_has_raise(if_node)
        has_soft = _if_has_soft_signal(if_node, source)
        if not (has_raise or has_soft):
            continue

        class_name = _enclosing_class_name(language, source, cond_node)
        method_name = _enclosing_method_name(language, source, cond_node)
        if class_name is None or method_name is None:
            continue
        if class_name not in allowlist:
            continue

        namespace = _resolve_namespace(class_name)
        cond_text = _node_text(source, cond_node)
        kind = _classify_predicate_kind_from_node(cond_node, source)

        # Field discovery: walk the enclosing-if chain to gather self.*
        # references. The OUTERMOST if's first self.<X> is treated as the
        # primary cross-field signal (matching reference convention for
        # cases like "do_sample gates temperature"). When there is no
        # enclosing if (innermost-only raise), use this if's own attrs.
        chain = _enclosing_if_chain(cond_node)
        all_attrs: list[str] = []
        for outer_if in chain:
            outer_cond = outer_if.child_by_field_name("condition")
            if outer_cond is None:
                continue
            for a in _collect_attribute_refs(outer_cond, source):
                if a not in all_attrs:
                    all_attrs.append(a)
        local_attrs = _collect_attribute_refs(cond_node, source)
        # Also pull self.X references from the consequence block when the
        # condition has none -- catches `for arg in (...): if hasattr(self,
        # arg): raise` where the field-name lives in a loop variable.
        block = _if_consequence_block(if_node)
        if not local_attrs and block is not None:
            local_attrs = _collect_attribute_refs(block, source)
        # Fallback: scan bare identifier references in the condition for
        # __init__-parameter predicates like `if load_in_4bit and
        # load_in_8bit:`. Restricted to "raise"-direct cases (not soft
        # signals) because soft-sink branches accumulate too many spurious
        # parameter mentions otherwise.
        if not local_attrs and has_raise:
            bare = _collect_bare_identifier_refs(cond_node, source)
            local_attrs = bare
        for a in local_attrs:
            if a not in all_attrs:
                all_attrs.append(a)
        if not all_attrs:
            continue

        # Emit ONE record per (primary, secondary) pair: when the chain
        # produces an outer field AND a local field, both pairings get
        # emitted (covers cases like `do_sample` outer -> {temperature,
        # top_p, ...} local).
        outer_attrs = []
        for outer_if in chain:
            outer_cond = outer_if.child_by_field_name("condition")
            if outer_cond is None:
                continue
            outer_attrs.extend(
                a for a in _collect_attribute_refs(outer_cond, source) if a not in outer_attrs
            )

        pairs: list[tuple[str, str, str]] = []
        if outer_attrs and local_attrs:
            outer_field = outer_attrs[0]
            outer_kind = _classify_predicate_kind_from_node(
                chain[0].child_by_field_name("condition"), source
            )
            for local_field in local_attrs:
                if local_field == outer_field:
                    continue
                # Reference convention: outer field is PRIMARY (do_sample),
                # local field is SECONDARY (temperature).
                pairs.append((outer_field, outer_kind, local_field))
            # Also keep the local-only version (single-field invariant).
            for local_field in local_attrs:
                pairs.append((local_field, kind, ""))
        else:
            for f in local_attrs:
                pairs.append((f, kind, ""))

        # Exception type for severity heuristic.
        exc_name = ""
        if has_raise and block is not None:
            for child in block.named_children:
                if child.type == "raise_statement":
                    for sub in child.named_children:
                        if sub.type == "call":
                            fn_node = sub.child_by_field_name("function")
                            if fn_node is not None:
                                exc_name = _node_text(source, fn_node)
                            break
                    break
        severity = "error" if has_raise else "dormant"

        for primary_field, primary_kind, secondary_field in pairs:
            # Drop the noise tier: predicate-kind "unknown" means the
            # classifier could not fit the cond into the taxonomy. These
            # are universally low-value and inflate the cell-side count.
            if primary_kind == "unknown":
                continue
            # Skip dunder fields like __class__.
            if primary_field.startswith("__") or secondary_field.startswith("__"):
                continue
            key = (namespace, primary_field, primary_kind, secondary_field)
            if key in seen:
                continue
            seen.add(key)

            line = cond_node.start_point[0] + 1
            inv_id = f"{engine}_{class_name.lower()}_{method_name}_{primary_field}_{primary_kind}".lower()
            if secondary_field:
                inv_id += f"_{secondary_field}"
            match_fields: dict[str, Any] = {}
            primary_key = f"{namespace}.{primary_field}" if namespace else primary_field
            match_fields[primary_key] = _predicate_value_for(primary_kind, cond_text, primary_field)
            if secondary_field:
                secondary_key = f"{namespace}.{secondary_field}" if namespace else secondary_field
                match_fields[secondary_key] = _predicate_value_for(
                    primary_kind, cond_text, secondary_field
                )

            invariants.append(
                {
                    "id": inv_id,
                    "engine": engine,
                    "library": engine,
                    "invariant_under_test": (
                        f"{class_name}.{method_name} flags `{primary_field}` ({primary_kind})"
                    ),
                    "severity": severity,
                    "native_type": f"{engine}.{class_name}",
                    "miner_source": {
                        "path": rel_path,
                        "method": method_name,
                        "line_at_scan": line,
                    },
                    "match": {
                        "engine": engine,
                        "fields": match_fields,
                    },
                    "added_by": "wave2_a_treesitter",
                }
            )

    # Validator-decorator query: catches pydantic-style validators where the
    # raised condition lives elsewhere but the @field_validator decorator
    # names the field directly. Less relevant for transformers (uses bare
    # if/raise) but valuable for engines using Pydantic-style validators.
    query_dec = language.query(_Q_VALIDATOR_DECORATOR)
    dec_matches = query_dec.matches(tree.root_node)
    for match_id, caps in dec_matches:
        dec_name_nodes = caps.get("decorator_name", [])
        vname_nodes = caps.get("validator_name", [])
        if not dec_name_nodes or not vname_nodes:
            continue
        class_name = _enclosing_class_name(language, source, vname_nodes[0])
        if class_name is None or class_name not in allowlist:
            continue
        # Walk up to find the decorator's call args -- the validator field
        # name is typically the first positional string literal.
        dec_node = dec_name_nodes[0].parent  # call node
        if dec_node is None or dec_node.type != "call":
            continue
        args_node = dec_node.child_by_field_name("arguments")
        if args_node is None:
            continue
        # First string-literal arg is the field name.
        target_field = None
        for arg in args_node.named_children:
            if arg.type == "string":
                target_field = _node_text(source, arg).strip("'\"")
                break
        if not target_field:
            continue
        namespace = _resolve_namespace(class_name)

        key = (namespace, target_field, "present", "")
        if key in seen:
            continue
        seen.add(key)
        line = vname_nodes[0].start_point[0] + 1
        validator_method = _node_text(source, vname_nodes[0])
        invariants.append(
            {
                "id": f"{engine}_{class_name.lower()}_{validator_method}_{target_field}_present".lower(),
                "engine": engine,
                "library": engine,
                "invariant_under_test": (
                    f"{class_name}.{validator_method} validates `{target_field}` (pydantic decorator)"
                ),
                "severity": "error",
                "native_type": f"{engine}.{class_name}",
                "miner_source": {
                    "path": rel_path,
                    "method": validator_method,
                    "line_at_scan": line,
                },
                "match": {
                    "engine": engine,
                    "fields": {f"{namespace}.{target_field}": {"present": True}},
                },
                "added_by": "wave2_a_treesitter",
            }
        )

    return invariants, observations


def _predicate_value_for(kind: str, cond_text: str, field: str) -> Any:
    """Build the match-field value matching the predicate kind.

    For kinds with a concrete RHS literal (gt/lt/ge/le/exact/not_equal),
    attempt to recover the literal from the cond text. Otherwise emit a
    structural marker (the scorer ignores the value -- only the kind +
    presence matter for identity).
    """
    if kind == "present":
        return {"present": True}
    if kind == "not_in":
        return {"present": True, "not_in": []}
    if kind == "type_is_not":
        return {"type_is_not": ""}
    # For numeric/exact kinds, we use a single sentinel literal so the value
    # round-trips, but identity comes from the kind alone.
    op_map = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "not_equal": "not_equal"}
    if kind in op_map and kind != "exact":
        return {op_map[kind]: 0}
    if kind == "exact":
        return 0
    return {}


def _build_invariants(
    *,
    engine: str,
    version: str,
    source_root: Path,
    parser: Any,
    language: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Build the invariants envelope by parsing every validator-file."""
    targets = _ENGINE_TARGETS[engine]
    observations: list[str] = []
    all_invariants: list[dict[str, Any]] = []

    for rel_path in targets["validator_files"]:
        full_path = source_root / rel_path
        source = _read_source(full_path)
        if not source:
            observations.append(f"missing validator source: {rel_path}")
            continue
        invariants, file_obs = _extract_invariants_from_file(
            engine=engine,
            rel_path=rel_path,
            source=source,
            language=language,
            parser=parser,
            engine_targets=targets,
        )
        all_invariants.extend(invariants)
        observations.extend(file_obs)

    envelope = {
        "schema_version": _INVARIANTS_SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "mined_at": datetime.now(timezone.utc).isoformat(),
        "invariants": all_invariants,
    }
    return envelope, observations


# ---------------------------------------------------------------------------
# Empty-envelope helpers (for single-task runs)
# ---------------------------------------------------------------------------


def _empty_schema_envelope(engine: str, version: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "engine_params": {},
        "sampling_params": {},
    }


def _empty_invariants_envelope(engine: str, version: str) -> dict[str, Any]:
    return {
        "schema_version": _INVARIANTS_SCHEMA_VERSION,
        "engine": engine,
        "engine_version": version.lstrip("v").replace("_", "."),
        "mined_at": datetime.now(timezone.utc).isoformat(),
        "invariants": [],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cell(
    *,
    engine: str,
    version: str,
    engine_source_root: Path,
    task: str = "both",
) -> CellOutput:
    """Execute one Wave 2 a-treesitter cell.

    Parameters
    ----------
    engine : str
        Engine name (transformers / vllm / tensorrt-llm).
    version : str
        Engine version (e.g. ``"v4.57.3"`` or ``"v4_57_3"``).
    engine_source_root : Path
        Filesystem root of the unpacked engine source tree (the directory
        whose children are ``generation/``, ``config.py``, etc.).
    task : str
        ``"schema"`` -> emit only schema; empty invariants envelope.
        ``"invariants"`` -> emit only invariants; empty schema envelope.
        ``"both"`` (default) -> emit both.
    """
    if engine not in _ENGINE_TARGETS:
        raise NotImplementedError(
            f"a_treesitter target table missing for engine={engine!r}; "
            f"add an entry to _ENGINE_TARGETS."
        )
    if task not in {"schema", "invariants", "both"}:
        raise ValueError(f"task must be 'schema' | 'invariants' | 'both'; got {task!r}")

    version_slug = (
        version
        if version.startswith("v") and "." not in version
        else (f"v{version.lstrip('v').replace('.', '_')}")
    )
    out_dir = standard_out_dir(_STRATEGY_ID, engine, version_slug)
    observations: list[str] = []
    wall: dict[str, float] = {}

    t_total = time.perf_counter()
    language, parser = _load_parser()

    # Schema phase.
    t_schema = time.perf_counter()
    if task in {"schema", "both"}:
        schema_envelope, schema_obs = _build_schema(
            engine=engine,
            version=version,
            source_root=engine_source_root,
            parser=parser,
            language=language,
        )
        observations.extend(f"schema: {o}" for o in schema_obs)
    else:
        schema_envelope = _empty_schema_envelope(engine, version)
    wall["schema"] = time.perf_counter() - t_schema

    # Invariants phase.
    t_inv = time.perf_counter()
    if task in {"invariants", "both"}:
        invariants_envelope, inv_obs = _build_invariants(
            engine=engine,
            version=version,
            source_root=engine_source_root,
            parser=parser,
            language=language,
        )
        observations.extend(f"invariants: {o}" for o in inv_obs)
    else:
        invariants_envelope = _empty_invariants_envelope(engine, version)
    wall["invariants"] = time.perf_counter() - t_inv

    schema_path = out_dir / "schema.json"
    invariants_path = out_dir / "invariants.proposed.yaml"
    schema_path.write_text(json.dumps(schema_envelope, indent=2, default=str))
    invariants_path.write_text(yaml.safe_dump(invariants_envelope, sort_keys=False))

    wall["total"] = time.perf_counter() - t_total

    return CellOutput(
        strategy_id=_STRATEGY_ID,
        engine=engine,
        engine_version=version,
        schema_path=schema_path,
        invariants_path=invariants_path,
        observations=observations,
        wall_sec=wall,
        extra={
            "task": task,
            "engine_param_count": len(schema_envelope.get("engine_params", {})),
            "sampling_param_count": len(schema_envelope.get("sampling_params", {})),
            "defs_count": len(schema_envelope.get("$defs", {})),
            "invariant_count": len(invariants_envelope.get("invariants", [])),
        },
    )


__all__ = ["run_cell"]
