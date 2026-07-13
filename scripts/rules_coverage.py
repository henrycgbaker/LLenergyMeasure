#!/usr/bin/env python3
"""Rules coverage check: does the shipped corpus still cover the engine's validators?

Run once per engine-version bump, next to the knowledge-refresh conductor. The
conductor re-confirms rules the maintainer already knows; this check answers the
complementary question: has upstream grown a NEW config validator that no shipped
rule speaks to? Such a validator is a coverage gap the maintainer should decide on.

Mechanism (host-only, no engine imports, uniform across engines):

1. Load the shipped rules and collect the leaf name (last dotted segment) of every
   rule's match path: the config fields the corpus already reasons about.
2. AST-parse every ``*.py`` under the staged source tree for VALIDATOR SITES:
   pydantic ``field_validator`` / ``model_validator`` functions (matched by the
   decorator's last dotted name, so any import alias is caught) and ``__post_init__``
   methods. For each, collect the config fields it reads - public attributes of the
   receiver (``self`` / ``cls``) or first data arg, plus any name in a
   ``field_validator`` decorator. A site counts only if it BOTH raises and reads a
   field; normalisers and field-less guards are out of the model.
3. A site is COVERED when its fields intersect the corpus leaves. The grain is
   deliberately coarse - a shared name, not a proven predicate. Advisory, not a prover.
4. Report every UNCOVERED site (engine-relative path, line, class.function, fields),
   sorted for byte-stable output, with a summary count. Exit 0 by default;
   ``--fail-on-uncovered`` exits 1 when any gap remains, for later opt-in gating.

Sites in the pinned sources that are genuinely not engine-config material (HTTP
request models, per-request objects, runtime structs, vendored kernels) are listed
in :data:`IGNORED_SITES` so the uncovered list stays real gaps. This script never
writes the corpus; proposing new rules is the refresh conductor's job.

Run: python scripts/rules_coverage.py --engine vllm --source-root SRC/
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.config.engine_rules.loader import EngineRulesLoader  # noqa: E402

ENGINES = ("vllm", "transformers", "tensorrt")
DEFAULT_CORPUS_ROOT = _PROJECT_ROOT / "src" / "llenergymeasure" / "engines"

# Pydantic validator decorators, matched by their last dotted segment so any import
# alias (``pydantic.field_validator``, ``from pydantic import model_validator``) is
# caught without following imports.
_VALIDATOR_DECORATORS = frozenset({"field_validator", "model_validator"})
_POST_INIT = "__post_init__"

# Validator sites in the pinned sources that are NOT engine-config-rule material,
# triaged by hand so the uncovered list stays real gaps. Each entry is
# "engine relative/posix/path Class.function"; the trailing comment is the rationale.
# A genuinely rule-material gap belongs in a rule (via the conductor), never here.
_IGNORED_RAW = (
    # vllm: HTTP API request models, not engine-construction config.
    "vllm entrypoints/anthropic/protocol.py AnthropicTool.validate_input_schema",
    "vllm entrypoints/anthropic/protocol.py AnthropicToolChoice.validate_name_required_for_tool",
    "vllm entrypoints/anthropic/protocol.py AnthropicMessagesRequest.validate_model",
    "vllm entrypoints/anthropic/protocol.py AnthropicCountTokensRequest.validate_model",
    "vllm entrypoints/serve/disagg/protocol.py GenerateRequest.validate_token_ids",
    # vllm: internal RPC weight-transfer payload structs, not user config.
    "vllm distributed/weight_transfer/ipc_engine.py IPCTrainerSendWeightsArgs.__post_init__",
    "vllm distributed/weight_transfer/ipc_engine.py IPCWeightTransferUpdateInfo.__post_init__",
    "vllm distributed/weight_transfer/nccl_engine.py NCCLWeightTransferUpdateInfo.__post_init__",
    # vllm: per-request / runtime objects and a vendored kernel struct.
    "vllm lora/request.py LoRARequest.__post_init__",  # per-request adapter handle
    "vllm renderers/params.py TokenizeParams.__post_init__",  # tokenize request params
    "vllm model_executor/layers/attention/mla_attention.py MLACommonMetadata.__post_init__",  # runtime metadata
    "vllm model_executor/model_loader/tensorizer.py TensorizerConfig.__post_init__",  # loader plumbing
    "vllm third_party/triton_kernels/tensor.py Tensor.__post_init__",  # vendored kernel tensor
    # tensorrt: the trtllm-bench tool's own dataclasses, not engine llm args.
    "tensorrt bench/dataclasses/configuration.py DecodingConfig.validate_speculative_decoding",
    "tensorrt bench/dataclasses/configuration.py ExecutorWorldConfig.validate_world_size",
    "tensorrt bench/dataclasses/general.py InferenceRequest.verify_prompt_and_logits",
    # tensorrt: per-request / runtime input objects and a private wrapper.
    "tensorrt executor/request.py LoRARequest.__post_init__",  # per-request adapter handle
    "tensorrt executor/request.py PromptAdapterRequest.__post_init__",  # per-request adapter handle
    "tensorrt inputs/multimodal.py MultimodalInput.__post_init__",  # runtime input struct
    "tensorrt inputs/multimodal.py MultimodalRuntimeData.__post_init__",  # runtime input struct
    "tensorrt llmapi/llm_args.py _ModelWrapper.__post_init__",  # private internal wrapper
    # tensorrt, new at 1.2.1: dataset-prep tool args, not engine llm args.
    "tensorrt bench/dataset/prepare_dataset.py RootArgs.validate_tokenizer",
    "tensorrt bench/dataset/prepare_real_data.py DatasetConfig.check_prompt",
    # tensorrt, new at 1.2.1: HTTP API request model, not engine-construction config.
    "tensorrt serve/openai_protocol.py ChatCompletionRequest.check_cache_salt_support",
    # tensorrt, new at 1.2.1: per-request / runtime objects and loader plumbing.
    "tensorrt disaggregated_params.py DisaggregatedParams.__post_init__",  # per-request disagg params
    "tensorrt inputs/utils.py VideoData.__post_init__",  # runtime input struct
    "tensorrt _torch/attention_backend/sparse/rocket.py RocketVanillaAttentionMetadata.__post_init__",  # runtime metadata
    "tensorrt _torch/auto_deploy/utils/_config.py DynamicYamlMixInForSettings.validate_mode_and_yaml_default_not_both_provided",  # settings plumbing
    # tensorrt, new at 1.2.1: the _autodeploy backend's own args; llem does not
    # route backend=_autodeploy.
    "tensorrt _torch/auto_deploy/llm_args.py AutoDeployConfig.model_factory_exists",
)
# (engine, engine-relative posix path, "Class.function"), parsed from _IGNORED_RAW.
IGNORED_SITES: frozenset[tuple[str, ...]] = frozenset(tuple(e.split()) for e in _IGNORED_RAW)


@dataclass(frozen=True)
class ValidatorSite:
    """One material validator function found in the source tree."""

    path: str  # engine-relative posix path
    line: int  # 1-based line of the ``def``
    qualname: str  # "Class.function"
    fields: tuple[str, ...]  # sorted config field names the body reads


def _decorator_last_name(node: ast.expr) -> str | None:
    """Return the last dotted name of a decorator expression (``pydantic.x`` -> ``x``), or None."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_validator_site(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    # A pydantic validator or a __post_init__ hook.
    if fn.name == _POST_INIT:
        return True
    names = {_decorator_last_name(d) for d in fn.decorator_list}
    return bool(names & _VALIDATOR_DECORATORS)


def _field_validator_targets(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    # The field names in a @field_validator("a", "b") decorator.
    targets: set[str] = set()
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        if _decorator_last_name(dec) != "field_validator":
            continue
        for arg in dec.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                targets.add(arg.value)
    return targets


def _read_fields(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return public config fields ``fn`` reads via receiver / data-arg attributes.

    Receivers are the first two positional params (the receiver self/cls and first
    data arg). Method calls (``self.foo()``) are excluded - the attribute is a
    ``Call.func``, not a read. Private / dunder attributes are excluded: the corpus
    names only public fields, so a leading-underscore read only adds noise.
    """
    positional = fn.args.posonlyargs + fn.args.args
    receivers = {p.arg for p in positional[:2]}
    called = {
        id(node.func)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    fields: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and id(node) not in called
            and not node.attr.startswith("_")
            and isinstance(node.value, ast.Name)
            and node.value.id in receivers
        ):
            fields.add(node.attr)
    return fields


def _iter_class_sites(cls: ast.ClassDef, relpath: str) -> list[ValidatorSite]:
    # Material validator sites defined directly in cls's body.
    sites: list[ValidatorSite] = []
    for stmt in cls.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_validator_site(stmt):
            continue
        fields = _read_fields(stmt) | _field_validator_targets(stmt)
        raises = any(isinstance(node, ast.Raise) for node in ast.walk(stmt))
        if not (fields and raises):
            continue  # a coverage target must both raise and read a field
        sites.append(
            ValidatorSite(
                path=relpath,
                line=stmt.lineno,
                qualname=f"{cls.name}.{stmt.name}",
                fields=tuple(sorted(fields)),
            )
        )
    return sites


def scan_source(source_root: Path) -> list[ValidatorSite]:
    """Parse every ``*.py`` under ``source_root`` and collect material validator sites.

    Each validator method is enumerated once, at its own definition; scanning the
    whole tree covers in-tree base-class validators at their own location without an
    import graph. Files that fail to parse are skipped.
    """
    sites: list[ValidatorSite] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # upstream files may have bad escape sequences
        for py in sorted(source_root.rglob("*.py")):
            try:
                tree = ast.parse(py.read_text())
            except (SyntaxError, ValueError, UnicodeDecodeError):
                continue
            relpath = py.relative_to(source_root).as_posix()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    sites.extend(_iter_class_sites(node, relpath))
    return sites


def corpus_leaf_names(engine: str, corpus_root: Path) -> tuple[str, set[str]]:
    """Return (engine_version, set of leaf field names) for the shipped corpus."""
    rules = EngineRulesLoader(corpus_root).load_rules(engine)
    leaves = {path.rsplit(".", 1)[-1] for rule in rules.rules for path in rule.match_fields}
    return rules.engine_version, leaves


def classify(
    engine: str, sites: list[ValidatorSite], leaves: set[str]
) -> tuple[list[ValidatorSite], list[ValidatorSite], int]:
    """Split sites into (covered, uncovered_reported, ignored_count).

    Covered = fields intersect the corpus leaves; the rest are uncovered, minus any
    :data:`IGNORED_SITES` entry for this engine (counted separately).
    """
    covered: list[ValidatorSite] = []
    uncovered: list[ValidatorSite] = []
    ignored = 0
    ignored_keys = {(e[1], e[2]) for e in IGNORED_SITES if e[0] == engine}
    for site in sites:
        if set(site.fields) & leaves:
            covered.append(site)
        elif (site.path, site.qualname) in ignored_keys:
            ignored += 1
        else:
            uncovered.append(site)
    order = attrgetter("path", "line", "qualname")
    return sorted(covered, key=order), sorted(uncovered, key=order), ignored


def render_report(
    engine: str,
    engine_version: str,
    covered: list[ValidatorSite],
    uncovered: list[ValidatorSite],
    ignored: int,
) -> str:
    """Render a deterministic, path-relative coverage report."""
    total = len(covered) + len(uncovered) + ignored
    lines = [
        f"Rules coverage check: {engine} (corpus engine_version {engine_version})",
        f"validator sites: {total} | covered: {len(covered)} | "
        f"uncovered: {len(uncovered)} | ignored: {ignored}",
        "",
    ]
    if uncovered:
        lines.append("Uncovered validator sites (no shipped rule names a field they read):")
        for site in uncovered:
            lines.append(
                f"  {site.path}:{site.line}  {site.qualname}  reads: {', '.join(site.fields)}"
            )
    else:
        lines.append("All validator sites are covered by the shipped rules.")
    return "\n".join(lines) + "\n"


def run(engine: str, source_root: Path, corpus_root: Path) -> tuple[str, int]:
    """Compute the report and the count of uncovered sites."""
    engine_version, leaves = corpus_leaf_names(engine, corpus_root)
    sites = scan_source(source_root)
    covered, uncovered, ignored = classify(engine, sites, leaves)
    report = render_report(engine, engine_version, covered, uncovered, ignored)
    return report, len(uncovered)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, choices=ENGINES)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Staged engine package directory (the same tree the refresh conductor takes).",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Root holding <engine>/rules.yaml (default: the shipped corpus).",
    )
    parser.add_argument(
        "--fail-on-uncovered",
        action="store_true",
        help="Exit 1 when any uncovered site remains (default: advisory, exit 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.source_root.is_dir():
        print(f"source root not found: {args.source_root}", file=sys.stderr)
        return 2
    report, uncovered_count = run(args.engine, args.source_root, args.corpus_root)
    sys.stdout.write(report)
    if args.fail_on_uncovered and uncovered_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
