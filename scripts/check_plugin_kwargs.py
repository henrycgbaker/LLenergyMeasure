#!/usr/bin/env python3
"""Cross-check hand-typed plugin constructor-kwarg names against the mined schema.

Each engine plugin has a thin translation layer that builds the keyword-argument
dict passed to the engine's native model constructor (``vllm.LLM(**kwargs)``,
``tensorrt_llm.LLM(**kwargs)``, ``AutoModelForCausalLM.from_pretrained(model,
**kwargs)``). Most of that dict is schema-sourced: it comes from
``model_dump()`` of the generated typed config, so those key names are the mined
schema field names by construction. But the layer also HAND-TYPES a handful of
kwarg names as string literals - the ``{"model": ...}`` seed, ``kwargs["X"] =
...`` assignments, and ``kwargs.update({...})`` merges. A hand-typed name that
upstream renamed or removed is silent drift: the mined schema knows the correct
name, but nothing checked the glue code against it. This lint closes that gap.
The schema is the truth; the translation layer must answer to it.

The motivating case: the tensorrt plugin passed quantisation under the hand-typed
name ``quantization``, a name upstream removed in favour of ``quant_config``. The
mined ``TrtLlmArgs`` schema carried ``quant_config`` all along; this check would
have flagged ``quantization`` as absent from the schema surface.

WHAT IS CHECKED (per engine, scoped to the constructor-kwargs variable(s) inside
the plugin's kwargs-builder function(s)):

  1. dict-literal keys in the kwargs seed:      ``kwargs = {"model": ...}``
  2. string-literal subscript assignments:      ``kwargs["quantization"] = ...``
  3. string-literal keys in a literal update:   ``kwargs.update({"model": ...})``

Each extracted name must be a field in the engine's discovered ``engine_params``
surface, or carry a rationale in :data:`ALLOWLIST`.

WHAT IS NOT CHECKED (by design):

  - Dynamic keys from ``model_dump()`` loops and ``kwargs.update(<var>)`` with a
    non-literal argument. These are schema-sourced by construction, so they
    already answer to the schema and are not hand-typed drift risks.
  - Kwargs to NESTED sub-config constructors (``QuantConfig``, ``SchedulerConfig``,
    ``KvCacheConfig``, ``BitsAndBytesConfig``, ``SamplingParams``). Those live on
    their own local variables (``qc_kwargs``, ``sc_kwargs``, ``bnb_kwargs``, ...),
    not on the engine-constructor kwargs variable, so scoping to the constructor
    variable excludes them. The lint targets the top-level engine constructor
    surface, which is what the discovered ``engine_params`` describes.
  - Computed / f-string / variable-typed keys, and ``|=`` dict merges.

The transformers surface is a special case: its schema is discovered from
``inspect.signature(from_pretrained)``, which excludes ``**kwargs`` (the schema's
own ``discovery_limitations`` documents this). Every kwarg the transformers layer
hand-types is routed through ``from_pretrained``'s open ``**kwargs``, so none are
in the signature-based surface; each is enumerated in :data:`ALLOWLIST` with a
rationale. The check still fires if a NEW hand-typed transformers kwarg appears,
forcing a conscious decision.

The schema is read at the current pin: ``engine_versions/<engine>/current.yaml``
gives the pinned version, and the mined snapshot at
``engine_versions/<engine>/v<safe>/outputs/schema.discovered.json`` supplies the
field surface (the same resolution the schema-version check uses).

Usage:
    python scripts/check_plugin_kwargs.py [--engine ENGINE]

Exit codes:
    0 = every hand-typed kwarg is a schema field or allowlisted
    1 = a hand-typed kwarg is neither in the schema nor allowlisted
    2 = error (missing file, parse failure, a configured builder went missing)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine_versions import _outputs  # noqa: E402

REPO_ROOT = _PROJECT_ROOT

# The engines that ship a mined-schema workspace, from that workspace's own
# registry (engine_versions/_outputs.py); do not hardcode the list here. The
# llenergymeasure package is deliberately not imported so hosted CI can run
# this check on the runner's bare python3, like the sibling schema-version
# check (a test cross-checks this registry against the config-side Engine enum).
ENGINES: tuple[str, ...] = _outputs.ENGINES


@dataclass(frozen=True)
class PluginSpec:
    """Where an engine plugin hand-types its constructor kwargs.

    ``builders`` names the functions in ``plugin.py`` that write the engine
    constructor's kwargs dict (a method plus any same-module helper it threads
    the dict through). ``kwargs_vars`` names the local variables that hold that
    dict; scoping extraction to these names excludes kwargs built for nested
    sub-config constructors, which use their own variables.
    """

    builders: tuple[str, ...]
    kwargs_vars: frozenset[str]


# A configured builder that goes missing is a loud error, not a silent pass: the
# extractor would otherwise find nothing and the lint would go dormant after a
# rename. ``early_kwargs`` is tensorrt's engine-path early-return dict; it may be
# absent in a given plugin revision, so an absent kwargs variable is fine, but an
# absent builder function is not.
_PLUGIN_SPECS: dict[str, PluginSpec] = {
    "tensorrt": PluginSpec(
        builders=("_build_llm_kwargs", "_apply_default_build_cache"),
        kwargs_vars=frozenset({"kwargs", "early_kwargs"}),
    ),
    "vllm": PluginSpec(
        builders=("_build_llm_kwargs",),
        kwargs_vars=frozenset({"kwargs"}),
    ),
    "transformers": PluginSpec(
        builders=("_model_load_kwargs",),
        kwargs_vars=frozenset({"kwargs"}),
    ),
}

# Hand-typed kwargs that are legitimately outside the mined engine_params
# surface. Keyed by (engine, kwarg_name) with a one-line rationale. A name that
# looks like drift belongs in a plugin fix, never here.
#
# transformers: every entry is routed through from_pretrained's open **kwargs.
# The transformers schema is discovered from inspect.signature(from_pretrained),
# which the schema's own discovery_limitations record as excluding **kwargs, so
# none of these appear in engine_params despite being valid constructor kwargs.
ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "transformers",
        "torch_dtype",
    ): "from_pretrained **kwargs; HF dtype selector (BC alias of dtype)",
    ("transformers", "device_map"): "from_pretrained **kwargs; accelerate device placement",
    ("transformers", "tp_plan"): "from_pretrained **kwargs; tensor-parallel plan",
    (
        "transformers",
        "tp_size",
    ): "from_pretrained **kwargs; tensor-parallel degree, paired with tp_plan",
    (
        "transformers",
        "trust_remote_code",
    ): "from_pretrained **kwargs; permits custom modelling code",
    ("transformers", "attn_implementation"): "from_pretrained **kwargs; attention kernel selector",
    ("transformers", "quantization_config"): "from_pretrained **kwargs; BitsAndBytesConfig carrier",
    ("transformers", "max_memory"): "from_pretrained **kwargs; per-device memory cap for offload",
    (
        "transformers",
        "low_cpu_mem_usage",
    ): "from_pretrained **kwargs; sharded-load memory optimisation",
}


@dataclass(frozen=True)
class EngineReport:
    """Result of checking one engine's plugin against its schema."""

    engine: str
    version: str
    field_count: int
    checked: tuple[str, ...]  # every hand-typed name found, sorted
    violations: tuple[str, ...]  # not in schema and not allowlisted, sorted
    allowlisted: tuple[str, ...]  # not in schema but allowlisted, sorted


def _literal_str(node: ast.expr | None) -> str | None:
    """Return the value of a string-literal AST node, or None if it is not one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dict_literal_keys(node: ast.Dict) -> list[str]:
    """String-literal keys of a dict literal (``**spread`` keys are None; skipped)."""
    keys: list[str] = []
    for key in node.keys:
        literal = _literal_str(key)
        if literal is not None:
            keys.append(literal)
    return keys


def extract_literal_kwargs(source: str, spec: PluginSpec) -> tuple[dict[str, int], frozenset[str]]:
    """Extract hand-typed constructor-kwarg names from a plugin's source.

    Walks the functions named in ``spec.builders`` and, scoped to assignments and
    updates of the ``spec.kwargs_vars`` variables, collects string-literal kwarg
    names from the three emission patterns (dict-literal seed, subscript assign,
    literal ``.update``). Returns ``(name -> first line seen, missing builders)``.
    The missing-builders set lets the caller fail loudly on a rename rather than
    go dormant.
    """
    tree = ast.parse(source)
    builder_fns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in spec.builders
    ]
    missing = frozenset(spec.builders) - {fn.name for fn in builder_fns}

    names: dict[str, int] = {}

    def _record(name: str, lineno: int) -> None:
        names.setdefault(name, lineno)

    def _is_kwargs_var(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id in spec.kwargs_vars

    for fn in builder_fns:
        for node in ast.walk(fn):
            # Patterns 1 & 2: dict-literal seed and subscript assignment.
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if _is_kwargs_var(target) and isinstance(node.value, ast.Dict):
                        for key in _dict_literal_keys(node.value):
                            _record(key, node.lineno)
                    elif isinstance(target, ast.Subscript) and _is_kwargs_var(target.value):
                        literal = _literal_str(target.slice)
                        if literal is not None:
                            _record(literal, node.lineno)
            # Pattern 3: literal .update({...}).
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and _is_kwargs_var(node.func.value)
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                for key in _dict_literal_keys(node.args[0]):
                    _record(key, node.lineno)

    return names, missing


def _pinned_version(current_yaml: Path) -> str:
    """Return ``library.current_version`` from an engine current.yaml."""
    data = yaml.safe_load(current_yaml.read_text()) or {}
    library = data.get("library") or {}
    value = library.get("current_version")
    if value is None:
        raise KeyError(f"library.current_version not found in {current_yaml}")
    return str(value)


def check_engine(engine: str, repo_root: Path) -> EngineReport:
    """Check one engine's plugin translation layer against its mined schema.

    Raises FileNotFoundError / KeyError / SyntaxError for the caller to turn into
    an exit-2 error, and a RuntimeError if a configured builder went missing or
    the engine has no registered PluginSpec.
    """
    spec = _PLUGIN_SPECS.get(engine)
    if spec is None:
        raise RuntimeError(
            f"{engine}: no PluginSpec registered. A new engine joined the "
            "mined-schema workspace; add its kwargs-builder spec to _PLUGIN_SPECS."
        )

    version = _pinned_version(repo_root / "engine_versions" / engine / "current.yaml")
    schema_path = (
        repo_root
        / "engine_versions"
        / engine
        / _outputs.safe_version(version)
        / "outputs"
        / _outputs.SCHEMA_FILENAME
    )
    schema = json.loads(schema_path.read_text())
    fields = set(schema.get("engine_params") or {})

    plugin_path = repo_root / "src" / "llenergymeasure" / "engines" / engine / "plugin.py"
    names, missing = extract_literal_kwargs(plugin_path.read_text(), spec)
    if missing:
        raise RuntimeError(
            f"{engine}: configured kwargs-builder function(s) not found in "
            f"{plugin_path.name}: {', '.join(sorted(missing))}. The plugin was "
            "refactored; update _PLUGIN_SPECS so the lint does not go dormant."
        )

    violations: list[str] = []
    allowlisted: list[str] = []
    for name in names:
        if name in fields:
            continue
        if (engine, name) in ALLOWLIST:
            allowlisted.append(name)
        else:
            violations.append(name)

    return EngineReport(
        engine=engine,
        version=str(schema.get("engine_version", version)),
        field_count=len(fields),
        checked=tuple(sorted(names)),
        violations=tuple(sorted(violations)),
        allowlisted=tuple(sorted(allowlisted)),
    )


def render_report(report: EngineReport) -> str:
    """Render a deterministic per-engine report block."""
    status = "FAIL" if report.violations else "PASS"
    lines = [
        f"[{report.engine}] {status}  "
        f"(engine_params @ {report.version}, {report.field_count} fields; "
        f"{len(report.checked)} hand-typed kwargs checked)"
    ]
    if report.violations:
        lines.append("  hand-typed kwargs absent from the mined schema:")
        for name in report.violations:
            lines.append(
                f"    {name}  -> not in engine_params; fix the plugin to use the "
                "mined name, or allowlist it with a rationale if it is a genuine "
                "off-surface constructor arg"
            )
    if report.allowlisted:
        lines.append(f"  allowlisted off-surface kwargs: {', '.join(report.allowlisted)}")
    return "\n".join(lines)


def main(repo_root: Path | None = None, engines: tuple[str, ...] | None = None) -> int:
    root = repo_root or REPO_ROOT
    reports: list[EngineReport] = []
    for engine in engines or ENGINES:
        try:
            reports.append(check_engine(engine, root))
        except (FileNotFoundError, KeyError, SyntaxError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    for report in reports:
        print(render_report(report))

    if any(report.violations for report in reports):
        print(
            "\nplugin kwarg lint FAILED: a hand-typed constructor kwarg does not "
            "match the mined schema.",
            file=sys.stderr,
        )
        return 1

    print("\nAll hand-typed plugin kwargs match the mined engine schema.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default=None,
        help="Check only the named engine. Omit to check all engines.",
    )
    args = parser.parse_args()
    sys.exit(main(engines=(args.engine,) if args.engine else None))
