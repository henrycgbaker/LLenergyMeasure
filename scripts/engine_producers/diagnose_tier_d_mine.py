"""Tier-D pure-inference advisory mine (maintainer venue).

Drives the Tier-D proposer (:func:`llenergymeasure.api.diagnose.diagnose_tier_d`)
over the deterministic class-3 candidate set
(:func:`scripts.engine_producers.tier_d_candidates.enumerate_class3_candidates`)
and the construction-violation gate inside the engine cache image. Lives under
``scripts/`` (NOT the ``llem`` CLI) because the candidate enumeration and the
match-path validator both import the producer toolbox, which the CLI layer may
not reach; this is the same maintainer venue as ``build_corpus`` and the bump
recipes.

GPU etiquette is the operator's: pass ``--docker-flag '--gpus' --docker-flag
'device=1'`` (or 2/3, NEVER device 0) to pin the gate container; the vLLM gate
runs CPU-only (``--gpus none``). Only construction-CONFIRMED advisories are
written; the rich unconfirmed report (constructs_legal / illegal_raises /
illegal_is_sentinel) is printed so the maintainer can see what a more permissive
keep-policy would additionally retain.

    python3 scripts/engine_producers/diagnose_tier_d_mine.py \
        --engine vllm --image llenergymeasure:vllm-0.19.1 --new-version 0.19.1 \
        --source-root /tmp/vllm-src --surface-file vllm/config/__init__.py \
        --surface-file vllm/sampling_params.py --docker-flag --gpus --docker-flag none \
        --out /tmp/tier_d_vllm.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.api.diagnose import (  # noqa: E402
    TIER_D_RESPONSE_SCHEMA,
    ContainerGateRunner,
    DiagnoseModel,
    DiagnoseResult,
    GateRunner,
    OllamaDiagnoseModel,
)
from llenergymeasure.api.diagnose import diagnose_tier_d as _diagnose_tier_d  # noqa: E402
from scripts.engine_producers._section_classifier import assert_path_resolves  # noqa: E402
from scripts.engine_producers.tier_d_candidates import (  # noqa: E402
    enumerate_class3_candidates,
)

_ENGINES = ("vllm", "tensorrt", "transformers")


def _path_validator(engine: str):
    """A ``Callable[[str], None]`` that raises on a non-resolving match path."""

    def _check(path: str) -> None:
        assert_path_resolves(engine, path)

    return _check


def run_tier_d_mine(
    *,
    engine: str,
    new_version: str,
    source_root: Path,
    surface_files: list[str],
    model: DiagnoseModel,
    gate_runner: GateRunner,
) -> DiagnoseResult:
    """Compute candidates and run the Tier-D proposer/gate (no GPU in this layer).

    Both ``model`` and ``gate_runner`` are injected so the whole orchestration is
    unit-testable without ollama or docker; the CLI ``main`` supplies the live
    :class:`OllamaDiagnoseModel` + :class:`ContainerGateRunner`.
    """
    candidates, _counts = enumerate_class3_candidates(engine)
    candidate_dicts = [
        {"section": c.section, "leaf": c.leaf, "type": c.type, "default": c.default}
        for c in candidates
    ]
    if not candidate_dicts:
        raise SystemExit(f"no class-3 candidates for {engine}; nothing to mine")
    return _diagnose_tier_d(
        engine=engine,
        new_version=new_version,
        candidates=candidate_dicts,
        source_root=source_root,
        config_surface_files=surface_files,
        model=model,
        gate_runner=gate_runner,
        path_validator=_path_validator(engine),
    )


def _report(result: DiagnoseResult) -> None:
    print(
        f"[{result.engine} {result.engine_version}] tier-d: "
        f"{len(result.confirmed)} construction-confirmed advisory(ies), "
        f"{len(result.tier_d_unconstrained)} unconstrained (model-rejected), "
        f"{len(result.unconfirmed)} not-confirmed, "
        f"{len(result.dropped_malformed)} dropped (malformed)",
        file=sys.stderr,
    )
    # The permissive-policy view: of the not-confirmed, how many constructed the
    # legal value and are NOT a sentinel / coercion artefact (a more permissive
    # keep-policy would retain these as UNVERIFIED warn advisories).
    permissive_extra = [
        r
        for r in result.unconfirmed
        if isinstance(r.get("gate"), dict)
        and r["gate"].get("constructs_legal")
        and not r["gate"].get("illegal_is_sentinel")
        and not r["gate"].get("coercion_artifact")
    ]
    if permissive_extra:
        print(
            f"  [permissive policy would additionally keep {len(permissive_extra)} "
            "unverified advisory(ies): construct-clean, non-sentinel, non-coercion]",
            file=sys.stderr,
        )
        for r in permissive_extra:
            print(f"    unverified: {r['rule_id']}", file=sys.stderr)
    for r in result.tier_d_unconstrained:
        print(f"  unconstrained: {r['rule_id']} ({r['reason']})", file=sys.stderr)
    for note in result.source_truncated:
        print(f"  source window (truncated for budget): {note}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Tier-D pure-inference advisory mine.")
    parser.add_argument("--engine", required=True, choices=_ENGINES)
    parser.add_argument("--image", required=True, help="Engine cache image to run the gate inside.")
    parser.add_argument("--new-version", required=True, help="The current engine version.")
    parser.add_argument("--source-root", required=True, type=Path, help="Engine source tree root.")
    parser.add_argument(
        "--surface-file",
        action="append",
        required=True,
        dest="surface_files",
        help="Config-surface source file relative to --source-root (repeatable).",
    )
    parser.add_argument(
        "--docker-flag",
        action="append",
        default=[],
        dest="docker_flags",
        help="Extra docker run flag for the gate container, e.g. --gpus / device=1 (repeatable).",
    )
    parser.add_argument("--model", default="qwen2.5-coder:32b", help="Local ollama proposer model.")
    parser.add_argument(
        "--out", type=Path, help="Where to write confirmed advisories (else stdout)."
    )
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/llem-tier-d"))
    args = parser.parse_args(argv)

    model = OllamaDiagnoseModel(model=args.model, response_schema=TIER_D_RESPONSE_SCHEMA)
    gate_runner = ContainerGateRunner(
        image=args.image,
        repo=Path.cwd(),
        workdir=args.workdir,
        docker_flags=tuple(args.docker_flags),
    )
    result = run_tier_d_mine(
        engine=args.engine,
        new_version=args.new_version,
        source_root=args.source_root,
        surface_files=args.surface_files,
        model=model,
        gate_runner=gate_runner,
    )
    _report(result)
    if result.confirmed and result.proposed_yaml is not None:
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(result.proposed_yaml)
            print(
                f"Wrote {len(result.confirmed)} advisory(ies) (added_by: llm_advisory) to {args.out}.",
                file=sys.stderr,
            )
        else:
            print(result.proposed_yaml)
    else:
        print("No construction-confirmed advisories; nothing written.", file=sys.stderr)
    return 0


__all__ = ["main", "run_tier_d_mine"]


if __name__ == "__main__":
    raise SystemExit(main())
