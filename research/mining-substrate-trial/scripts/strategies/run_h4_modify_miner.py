"""H4 (LLM-modifies-miner) runner for Phase 3b.

Per ``research/mining-substrate-trial/findings/phase3b_hybrid_catalogue.md`` H4 + the gap inventory
in ``post_trial_a_gap_closure.md``. For each (engine, version) cell:

1. Construct an LLM prompt containing
   (a) the existing static-miner producer source,
   (b) a curated excerpt of the engine source showing the known gap
       locations (kept small to fit 32k context),
   (c) a sample of (a)'s current output (just IDs + severities,
       not full bodies), and
   (d) the gap-inventory entries describing what (a) misses + WHY
       structurally (normalisation pattern, local-var compare, if/elif
       /else, nested-config dispatch, type-blind probe synthesis,
       deprecation warning poisoning, etc.).

2. Ask llama3.1:70b (container Ollama, port 11435) to emit a STRUCTURED
   JSON response with:
   - ``diagnoses``: list of {gap_id, what_missed, why_structurally,
     proposed_fix_summary}.
   - ``patches``: list of {patch_id, gap_id, anchor_text, new_code,
     mode (``replace`` | ``insert_after`` | ``insert_before``),
     scope_note (mergeable / exploratory / needs_broader_refactor)}.

   We do NOT ask for unified diffs - 70B-q4 unreliable at exact line
   numbers. ``anchor_text`` + ``new_code`` is robust to whitespace
   drift.

3. Apply patches to an ISOLATED COPY of the producer at
   ``research/mining-substrate-trial/findings/hybrid_experiments/h4_modify_miner/<engine>/
   patched_producer/static_invariant_miner.py``. The canonical producer
   at ``engine_versions/<engine>/v*/producers/`` is NEVER touched.

4. Re-run the patched miner via subprocess (PYTHONPATH points at the
   isolated copy + the pre-built source venv tree). Capture output.

5. Diff: original (a) invariant count vs patched (a) invariant count,
   per-identity additions / changes, runtime-validation if possible.

6. Score patched output against the active reference using the existing
   ``trial_scoring.score_invariants`` machinery.

Contract: one entry point ``run_h4_for_engine(engine, out_root, ...)``
returns ``H4Result`` with all paths + counts. The CLI at __main__ runs
all three engines and writes per-engine ``h4_results.md``.

This script is the runner; the post-run ``h4_summary.md`` writeup lives
under ``research/mining-substrate-trial/findings/hybrid_experiments/h4_modify_miner/h4_summary.md``
and is produced by the agent after reviewing the per-engine outputs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Project root sits four parents above research/mining-substrate-trial/scripts/strategies/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_TRIAL_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_TRIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_TRIAL_SCRIPTS_DIR))

# Local imports.
from strategies.llm_extractor import (  # noqa: E402
    OllamaBackend,
    extract_with_retry,
)
from trial_scoring import (  # noqa: E402
    ScoringConfig,
    score_invariants,
)


# ---------------------------------------------------------------------------
# Engine registry: where the producer lives, where the venv source is, and
# the curated engine-source excerpts each LLM call needs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineConfig:
    engine: str
    """``transformers`` | ``vllm`` | ``tensorrt``."""

    version_slug: str
    """``v4_57_3`` | ``v0_7_3`` | ``v0_21_0``."""

    producer_rel: str
    """Producer path under ``engine_versions/<engine>/<version_slug>/``."""

    walker_fn: str
    """Function name to call: ``walk_transformers_static`` / ``walk_vllm_static`` / ``walk_tensorrt``."""

    venv_source_parent: str
    """sys.path entry for subprocess: ``/tmp/trial_<engine>_<version>_venv/src``."""

    gap_summary: str
    """Short prose describing the known gaps from ``post_trial_a_gap_closure.md``."""

    engine_excerpts: list[tuple[str, str]] = field(default_factory=list)
    """List of ``(label, source_text)`` curated engine-source snippets the
    LLM should see. Each ~50-150 lines of the engine source covering ONE
    gap location."""


# ---------------------------------------------------------------------------
# Patch + diagnosis dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Diagnosis:
    gap_id: str
    what_missed: str
    why_structurally: str
    proposed_fix_summary: str

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Diagnosis:
        return cls(
            gap_id=str(obj.get("gap_id", "<unknown>")),
            what_missed=str(obj.get("what_missed", "")),
            why_structurally=str(obj.get("why_structurally", "")),
            proposed_fix_summary=str(obj.get("proposed_fix_summary", "")),
        )


@dataclass
class Patch:
    patch_id: str
    gap_id: str
    anchor_text: str
    new_code: str
    mode: str  # ``replace`` | ``insert_after`` | ``insert_before``
    scope_note: str  # ``mergeable`` | ``exploratory`` | ``needs_broader_refactor``

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> Patch:
        return cls(
            patch_id=str(obj.get("patch_id", "<unknown>")),
            gap_id=str(obj.get("gap_id", "<unknown>")),
            anchor_text=str(obj.get("anchor_text", "")),
            new_code=str(obj.get("new_code", "")),
            mode=str(obj.get("mode", "replace")),
            scope_note=str(obj.get("scope_note", "exploratory")),
        )


@dataclass
class PatchApplicationResult:
    patch_id: str
    applied: bool
    reason: str
    location_line: int | None = None


@dataclass
class H4Result:
    engine: str
    version_slug: str
    diagnoses_count: int
    patches_proposed: int
    patches_applied: int
    patches_results: list[PatchApplicationResult]
    # canonical = the active-reference invariants.proposed.yaml under
    # engine_versions/<e>/v*/outputs/ - mined at some earlier moment
    # with a fully-installed engine.
    canonical_count: int
    # baseline = re-running the UNPATCHED walker today against the
    # current source tree + the trial env's monkey-patched landmark
    # resolver. This is the "what would (a) produce today" baseline.
    baseline_count: int
    baseline_run_ok: bool
    # patched = re-running the H4-PATCHED walker today, same context.
    patched_count: int
    patched_run_ok: bool
    # Recall comparisons.
    invariant_recall_patched_vs_canonical: float
    invariant_precision_patched_vs_canonical: float
    invariant_recall_patched_vs_baseline: float
    invariant_precision_patched_vs_baseline: float
    # Identity-level delta.
    new_identities_vs_canonical: list[list[str]]
    lost_identities_vs_canonical: list[list[str]]
    new_identities_vs_baseline: list[list[str]]
    lost_identities_vs_baseline: list[list[str]]
    miner_stderr_tail: str | None
    baseline_stderr_tail: str | None
    llm_wall_sec: float
    out_dir: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are an expert Python AST static-analysis engineer. You are reviewing
an AST-walker that mines validation invariants from an inference-engine's
configuration classes (e.g. vLLM's CacheConfig, ModelConfig).

Your job: read the AST-walker source, the engine-source excerpts (which
show the patterns the walker MUST detect), and the existing output (which
shows what the walker currently emits). Then:

1. Diagnose the GAPS: what validation patterns are present in the engine
   source but NOT emitted by the walker. For each gap, explain WHY
   structurally (e.g. "the walker only handles top-level `if X: raise`
   patterns, but this validator uses `if X: pass / elif Y: log / else:
   raise` which means the raise lives in an `orelse` branch the walker
   doesn't descend into").

2. Propose MINIMAL walker patches: for each gap, propose ONE concrete
   code change to the walker source. Each patch must be:
   - Localised (one function or one detector at a time, ideally <30 lines).
   - Robust to whitespace drift (use a multi-line `anchor_text` that's
     unique in the walker source).
   - Honest: if a gap needs broader refactoring (e.g. a new dispatch
     table, restructuring the walk), set scope_note to
     "needs_broader_refactor" and explain in proposed_fix_summary.

OUTPUT FORMAT: a single JSON object matching this schema (no markdown,
no commentary, no code fences, just the raw JSON):

{{
  "diagnoses": [
    {{
      "gap_id": "G1",
      "what_missed": "...",
      "why_structurally": "...",
      "proposed_fix_summary": "..."
    }}
  ],
  "patches": [
    {{
      "patch_id": "P1",
      "gap_id": "G1",
      "anchor_text": "<the EXACT existing text the patch attaches to; pick a unique multi-line snippet>",
      "new_code": "<the replacement / inserted code>",
      "mode": "replace",
      "scope_note": "mergeable"
    }}
  ]
}}

Constraints on patches:
- ``anchor_text`` MUST appear EXACTLY ONCE in the walker source. Pick
  something unique. Multi-line is fine; >= 3 lines preferred.
- For ``mode=replace``: new_code replaces anchor_text entirely.
- For ``mode=insert_after`` / ``insert_before``: new_code is inserted
  immediately after / before the anchor.
- Preserve the walker's existing import set (no new imports unless they
  come from the existing import block).
- Patches should be small (typically <40 lines each). Big refactors get
  ``scope_note: needs_broader_refactor`` and a textual sketch in
  ``proposed_fix_summary`` rather than a working patch.
- 1-5 patches per engine is realistic. Quality > quantity. If the gaps
  are unfixable with a localised patch, say so in diagnoses and emit
  zero patches.

=== WALKER SOURCE: {engine} v{version_slug} static_invariant_miner.py ===
{walker_source}

=== ENGINE SOURCE EXCERPTS ===
{engine_excerpts}

=== EXISTING (a) OUTPUT (invariant IDs + severities + match-field shape) ===
{output_summary}

=== KNOWN GAPS (from post_trial_a_gap_closure.md) ===
{gap_summary}

Now emit the JSON. Remember: NO markdown code fences, NO commentary, just
the raw JSON object starting with `{{`.
"""


def _build_output_summary(invariants_yaml_path: Path) -> str:
    """Extract just IDs + severities + first match-field key from (a) output.

    Full invariant bodies are too token-heavy; the LLM only needs to know
    what's already covered to identify gaps.
    """
    import yaml

    text = invariants_yaml_path.read_text()
    doc = yaml.safe_load(text) or {}
    invs = doc.get("invariants", []) or []
    out_lines: list[str] = []
    for i, inv in enumerate(invs):
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", f"<{i}>")
        sev = inv.get("severity", "?")
        mfields = (inv.get("match") or {}).get("fields") or {}
        first_field = next(iter(mfields.keys()), "<no-match>")
        out_lines.append(f"- {iid} [{sev}] -> {first_field}")
    return "\n".join(out_lines) if out_lines else "(no invariants)"


def _build_engine_excerpts_block(excerpts: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for label, src in excerpts:
        parts.append(f"--- {label} ---\n{src.rstrip()}\n")
    return "\n".join(parts) if parts else "(no excerpts provided)"


def build_prompt(engine_cfg: EngineConfig, walker_source: str, invariants_yaml_path: Path) -> str:
    return PROMPT_TEMPLATE.format(
        engine=engine_cfg.engine,
        version_slug=engine_cfg.version_slug,
        walker_source=walker_source,
        engine_excerpts=_build_engine_excerpts_block(engine_cfg.engine_excerpts),
        output_summary=_build_output_summary(invariants_yaml_path),
        gap_summary=engine_cfg.gap_summary,
    )


# ---------------------------------------------------------------------------
# Patch application: string-based, not unified-diff-based
# ---------------------------------------------------------------------------


def apply_patch(source_text: str, patch: Patch) -> tuple[str, PatchApplicationResult]:
    """Apply one patch to ``source_text``. Returns ``(new_text, result)``."""
    anchor = patch.anchor_text
    if not anchor.strip():
        return source_text, PatchApplicationResult(
            patch_id=patch.patch_id, applied=False, reason="empty anchor_text"
        )

    # Count occurrences; require exactly 1.
    count = source_text.count(anchor)
    if count == 0:
        return source_text, PatchApplicationResult(
            patch_id=patch.patch_id,
            applied=False,
            reason="anchor_text not found in walker source",
        )
    if count > 1:
        return source_text, PatchApplicationResult(
            patch_id=patch.patch_id,
            applied=False,
            reason=f"anchor_text appears {count} times; not unique",
        )

    # Find line number for diagnostics.
    pos = source_text.find(anchor)
    line_no = source_text.count("\n", 0, pos) + 1

    mode = patch.mode.strip().lower()
    if mode == "replace":
        new_text = source_text.replace(anchor, patch.new_code, 1)
    elif mode == "insert_after":
        new_text = source_text.replace(anchor, anchor + "\n" + patch.new_code, 1)
    elif mode == "insert_before":
        new_text = source_text.replace(anchor, patch.new_code + "\n" + anchor, 1)
    else:
        return source_text, PatchApplicationResult(
            patch_id=patch.patch_id,
            applied=False,
            reason=f"unknown mode: {patch.mode!r}",
            location_line=line_no,
        )

    return new_text, PatchApplicationResult(
        patch_id=patch.patch_id, applied=True, reason="applied", location_line=line_no
    )


# ---------------------------------------------------------------------------
# Patched-miner subprocess runner
# ---------------------------------------------------------------------------


def run_patched_miner(
    *,
    engine: str,
    walker_fn: str,
    patched_producer_dir: Path,
    venv_source_parent: str,
    output_yaml: Path,
    timeout_sec: int = 180,
) -> tuple[bool, str, str]:
    """Subprocess-invoke the patched producer. Returns (ok, stdout, stderr).

    The trial source-only venvs at ``/tmp/trial_<engine>_<v>_venv`` contain
    JUST the engine source tree - no transitive deps (msgspec, zmq, ...).
    The vllm walker hits ``_check_landmarks()`` -> ``import vllm`` which
    cascades into the dep wall on this env. Two coping strategies in this
    runner:

    1. **Light dep stubs**: before exec, register dummy ``msgspec``,
       ``zmq``, ``cloudpickle``, ``numba``, ``ninja``, ``gguf`` modules
       in ``sys.modules``. Just enough to satisfy import-time references;
       no runtime functionality. Won't work if vllm uses these at class
       body time (e.g. inherits from ``msgspec.Struct``), but covers the
       common case.

    2. **Already-mined baseline**: if the live walker crashes (typical
       for vllm at this version), the existing canonical
       ``engine_versions/<e>/v*/outputs/invariants.proposed.yaml`` IS the
       result of a previous successful walker run against a fully-
       installed engine. We can't reproduce it here, but the patch's
       LOGIC change can be inspected via static AST diff + diagnoses.
       Subprocess returncode 1 with ImportError chain in stderr signals
       this case; the H4 result still captures the patches + diagnoses.
    """
    runner_src = f"""
import sys, os, types, importlib, importlib.util

# Keep workspace's main checkout off sys.path; we only want the trial worktree.
sys.path = [p for p in sys.path if 'workspace/llenergymeasure/.venv' not in p]
sys.path.insert(0, os.environ['SOURCE_ROOT_PARENT'])
sys.path.insert(0, os.environ['PROJECT_ROOT'])

# --- Dep stubs for the source-only venv environment ---------------------
# vllm 0.7.3 imports msgspec, zmq, cloudpickle, etc. transitively at
# package-load time. We can't install these on the trial host. Provide
# minimal sys.modules stubs so the import chain doesn't crash. If a
# stub is insufficient (e.g. vllm class inherits msgspec.Struct), we
# return success=False and the H4 result records the failure mode.
def _make_pkg(n):
    m = types.ModuleType(n); m.__path__ = []
    return m
def _ensure(n, *, pkg=False, **attrs):
    m = sys.modules.get(n)
    if m is None:
        m = _make_pkg(n) if pkg else types.ModuleType(n)
        sys.modules[n] = m
    for k, v in attrs.items(): setattr(m, k, v)
    return m

# msgspec: vllm sequence + sampling use msgspec.Struct. Provide a
# tolerant base class.
class _MsgSpecStruct:
    def __init_subclass__(cls, **kwargs): pass
class _Meta:
    def __init__(self, **kwargs): pass
ms = _ensure('msgspec', pkg=True, Struct=_MsgSpecStruct, UNSET=None,
              field=lambda *a, **k: None, Meta=_Meta)
_ensure('msgspec.json', encode=lambda *a, **k: b'', decode=lambda *a, **k: None)

# zmq package + asyncio submodule. The asyncio submodule must be bound
# to the parent as an attribute, since vllm/utils.py does
# ``zmq.asyncio.Context`` at module body load. Stub the symbol set
# vllm imports via Union[zmq.Socket, zmq.asyncio.Socket].
class _ZmqOpaque:
    def __init__(self, *a, **k): pass
_z = _ensure('zmq', pkg=True, REQ=0, REP=1, PUSH=2, PULL=3, NOBLOCK=4,
             Context=_ZmqOpaque, Socket=_ZmqOpaque)
_z_async = _ensure('zmq.asyncio', Context=_ZmqOpaque, Socket=_ZmqOpaque)
_z.asyncio = _z_async
_ensure('zmq.constants')

# Misc.
_ensure('cloudpickle', dumps=lambda x: b'', loads=lambda x: None)
_ensure('numba', pkg=True, jit=lambda *a, **k: (lambda f: f))
_ensure('numba.core', pkg=True)
_ensure('numba.core.compiler')
_ensure('ninja')
_ensure('blake3', blake3=lambda data=None: type('B', (), {{'digest': lambda self: b'0'*32, 'hexdigest': lambda self: '0'*64}})())
# gguf intentionally NOT stubbed: transformers' import_utils probes for
# gguf via importlib.util.find_spec, and a stub without __spec__ fails
# that check with ValueError. Leaving gguf absent lets transformers
# detect "not installed" and continue.

# vllm itself: pre-register the package node so children can import.
# We don't dummy this - we let vllm/__init__.py run; if it crashes the
# stderr will tell us.
# ----------------------------------------------------------------------

MOD_NAME = 'h4_patched_producer'
spec = importlib.util.spec_from_file_location(
    MOD_NAME,
    os.environ['PATCHED_PRODUCER_PATH'],
)
miner = importlib.util.module_from_spec(spec)
sys.modules[MOD_NAME] = miner
spec.loader.exec_module(miner)

# vllm walker's _check_landmarks does ``import vllm`` which cascades
# into a deep transitive dep wall. Monkey-patch it to a file-based
# resolver that just verifies the source tree exists. This lets us
# exercise the walker logic on the active cell despite missing
# transitive deps in the trial environment.
ENGINE = os.environ.get('H4_ENGINE', '')
if ENGINE == 'vllm' and hasattr(miner, '_check_landmarks'):
    import ast as _ast
    from pathlib import Path as _Path
    def _file_landmarks():
        src = _Path(os.environ['SOURCE_ROOT_PARENT']) / 'vllm'
        if not src.is_dir():
            raise miner.MinerLandmarkMissingError('vllm package', detail=f'source tree absent at {{src}}')
        # Build abs_paths dict in the shape the walker expects.
        abs_paths = {{}}
        for target in miner._CLASS_TARGETS:
            sub = target.module_path.replace('vllm.', '', 1).replace('.', '/') + '.py'
            cand = src / sub
            if not cand.is_file():
                raise miner.MinerLandmarkMissingError(target.module_path, detail=f'file missing: {{cand}}')
            abs_paths[target.module_path] = str(cand)
            # Also verify class + method via AST so patched walker logic
            # gets the same landmark-presence guarantee as the live path.
            tree = _ast.parse(cand.read_text())
            cls_def = miner.find_class(tree, target.class_name)
            if cls_def is None:
                raise miner.MinerLandmarkMissingError(f'{{target.module_path}}.{{target.class_name}}', detail='ClassDef not in AST')
            m_def = miner.find_method(cls_def, target.method)
            if m_def is None:
                raise miner.MinerLandmarkMissingError(f'{{target.module_path}}.{{target.class_name}}.{{target.method}}', detail='FunctionDef not in AST')
        return '0.7.3', abs_paths
    miner._check_landmarks = _file_landmarks
    print('[h4 subprocess] vllm walker: live _check_landmarks replaced with file-based resolver', file=sys.stderr)

try:
    result = miner.{walker_fn}()
    if len(result) == 2:
        candidates, version = result
        rel_path = None
    else:
        candidates, version, rel_path = result
    if rel_path is None:
        text = miner.emit_yaml(candidates, engine_version=version)
    else:
        text = miner.emit_yaml(candidates, engine_version=version, rel_path=rel_path)
    with open(os.environ['OUT_PATH'], 'w') as f:
        f.write(text)
    print(f'wrote {{len(candidates)}} candidates to {{os.environ["OUT_PATH"]}}')
except Exception as exc:
    import traceback
    print(f'PATCHED_MINER_CRASH:{{type(exc).__name__}}:{{exc}}', file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise SystemExit(1)
"""
    env = {
        **__import__("os").environ,
        "H4_ENGINE": engine,
        "SOURCE_ROOT_PARENT": venv_source_parent,
        "PROJECT_ROOT": str(_PROJECT_ROOT),
        "PATCHED_PRODUCER_PATH": str(patched_producer_dir / "static_invariant_miner.py"),
        "OUT_PATH": str(output_yaml),
    }
    py = "/home/h.baker@hertie-school.lan/miniforge3/bin/python3.12"
    if not Path(py).exists():
        py = sys.executable
    try:
        proc = subprocess.run(
            [py, "-c", runner_src],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return False, "", f"TIMEOUT after {timeout_sec}s: {exc}"
    if proc.returncode != 0:
        return False, proc.stdout, proc.stderr
    return True, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def run_h4_for_engine(
    engine_cfg: EngineConfig,
    *,
    out_root: Path,
    backend: OllamaBackend,
    max_retries: int = 1,
) -> H4Result:
    """Run the full H4 pipeline for one engine cell."""
    engine_root = out_root / engine_cfg.engine
    raw_dir = engine_root / "raw_llm_outputs"
    patches_dir = engine_root / "proposed_patches"
    patched_prod_dir = engine_root / "patched_producer"
    patched_out_dir = engine_root / "patched_outputs"
    for d in [raw_dir, patches_dir, patched_prod_dir, patched_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load producer source + (a) output for this engine.
    producer_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / engine_cfg.engine
        / engine_cfg.version_slug
        / "producers"
        / "static_invariant_miner.py"
    )
    walker_source = producer_path.read_text()
    invariants_yaml = (
        _PROJECT_ROOT
        / "engine_versions"
        / engine_cfg.engine
        / engine_cfg.version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )

    # 2. Build prompt + call Ollama.
    prompt = build_prompt(engine_cfg, walker_source, invariants_yaml)
    (raw_dir / "prompt.txt").write_text(prompt)

    print(f"[H4 {engine_cfg.engine}] sending prompt ({len(prompt):,} chars) to Ollama...")
    extraction = extract_with_retry(
        backend=backend,
        prompt=prompt,
        chunk_name=f"h4_{engine_cfg.engine}",
        output_format="json",
        max_retries=max_retries,
        timeout=1800,
        max_output_tokens=4096,
    )
    # Save raw responses no matter what.
    (raw_dir / "raw_response.txt").write_text(
        "\n=== RESPONSE ===\n".join(extraction.raw_responses)
        if extraction.raw_responses
        else "<no responses>"
    )

    llm_wall = extraction.elapsed_sec
    diagnoses: list[Diagnosis] = []
    patches: list[Patch] = []
    if extraction.parsed is not None and isinstance(extraction.parsed, dict):
        for d in extraction.parsed.get("diagnoses", []) or []:
            if isinstance(d, dict):
                diagnoses.append(Diagnosis.from_dict(d))
        for p in extraction.parsed.get("patches", []) or []:
            if isinstance(p, dict):
                patches.append(Patch.from_dict(p))

    # Save parsed structured outputs.
    (raw_dir / "diagnoses.json").write_text(
        json.dumps([asdict(d) for d in diagnoses], indent=2)
    )
    (raw_dir / "patches.json").write_text(
        json.dumps([asdict(p) for p in patches], indent=2)
    )

    # 3. Apply patches to isolated copy.
    patched_text = walker_source
    patch_results: list[PatchApplicationResult] = []
    applied_count = 0
    for p in patches:
        # Save individual patch description for the audit trail.
        (patches_dir / f"{p.patch_id}.json").write_text(json.dumps(asdict(p), indent=2))
        # Skip needs_broader_refactor patches - they're sketches not code.
        if p.scope_note.strip().lower() == "needs_broader_refactor":
            patch_results.append(
                PatchApplicationResult(
                    patch_id=p.patch_id,
                    applied=False,
                    reason="scope_note=needs_broader_refactor; skipped",
                )
            )
            continue
        patched_text, result = apply_patch(patched_text, p)
        patch_results.append(result)
        if result.applied:
            applied_count += 1

    patched_producer_file = patched_prod_dir / "static_invariant_miner.py"
    patched_producer_file.write_text(patched_text)

    # 4a. Baseline run: unpatched walker re-run today against the current
    # source tree. Gives us "what (a) would emit today" as the delta floor.
    baseline_dir = engine_root / "baseline_unpatched_run"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_producer_dir = baseline_dir / "producer_copy"
    baseline_producer_dir.mkdir(parents=True, exist_ok=True)
    (baseline_producer_dir / "static_invariant_miner.py").write_text(walker_source)
    baseline_yaml = baseline_dir / "invariants.proposed.yaml"
    print(f"[H4 {engine_cfg.engine}] running BASELINE (unpatched) walker for delta floor...")
    base_ok, base_stdout, base_stderr = run_patched_miner(
        engine=engine_cfg.engine,
        walker_fn=engine_cfg.walker_fn,
        patched_producer_dir=baseline_producer_dir,
        venv_source_parent=engine_cfg.venv_source_parent,
        output_yaml=baseline_yaml,
        timeout_sec=240,
    )
    (baseline_dir / "subprocess_stdout.txt").write_text(base_stdout)
    (baseline_dir / "subprocess_stderr.txt").write_text(base_stderr)

    # 4b. Patched run.
    patched_yaml = patched_out_dir / "invariants.proposed.yaml"
    print(f"[H4 {engine_cfg.engine}] running patched miner...")
    ok, stdout, stderr = run_patched_miner(
        engine=engine_cfg.engine,
        walker_fn=engine_cfg.walker_fn,
        patched_producer_dir=patched_prod_dir,
        venv_source_parent=engine_cfg.venv_source_parent,
        output_yaml=patched_yaml,
        timeout_sec=240,
    )
    (patched_out_dir / "subprocess_stdout.txt").write_text(stdout)
    (patched_out_dir / "subprocess_stderr.txt").write_text(stderr)

    # 5. Score against canonical reference + against today-baseline.
    canonical_invs = _load_yaml(invariants_yaml)
    canonical_count = len((canonical_invs.get("invariants") or []))

    baseline_invs = (
        _load_yaml(baseline_yaml) if (base_ok and baseline_yaml.exists()) else {"invariants": []}
    )
    baseline_count = len((baseline_invs.get("invariants") or []))

    if ok and patched_yaml.exists():
        patched_invs = _load_yaml(patched_yaml)
        patched_count = len((patched_invs.get("invariants") or []))
    else:
        patched_invs = {"invariants": []}
        patched_count = 0

    cfg = ScoringConfig()
    (
        recall_vs_canon,
        precision_vs_canon,
        *_rest,
    ) = score_invariants(
        reference_invariants=canonical_invs, cell_invariants=patched_invs, config=cfg
    )
    (
        recall_vs_base,
        precision_vs_base,
        *_rest2,
    ) = score_invariants(
        reference_invariants=baseline_invs, cell_invariants=patched_invs, config=cfg
    )

    from trial_scoring import invariant_identities

    ids_canon = invariant_identities(canonical_invs)
    ids_base = invariant_identities(baseline_invs)
    ids_patch = invariant_identities(patched_invs)
    new_vs_canon = sorted([list(map(str, x)) for x in (ids_patch - ids_canon)])
    lost_vs_canon = sorted([list(map(str, x)) for x in (ids_canon - ids_patch)])
    new_vs_base = sorted([list(map(str, x)) for x in (ids_patch - ids_base)])
    lost_vs_base = sorted([list(map(str, x)) for x in (ids_base - ids_patch)])

    result = H4Result(
        engine=engine_cfg.engine,
        version_slug=engine_cfg.version_slug,
        diagnoses_count=len(diagnoses),
        patches_proposed=len(patches),
        patches_applied=applied_count,
        patches_results=patch_results,
        canonical_count=canonical_count,
        baseline_count=baseline_count,
        baseline_run_ok=base_ok,
        patched_count=patched_count,
        patched_run_ok=ok,
        invariant_recall_patched_vs_canonical=recall_vs_canon,
        invariant_precision_patched_vs_canonical=precision_vs_canon,
        invariant_recall_patched_vs_baseline=recall_vs_base,
        invariant_precision_patched_vs_baseline=precision_vs_base,
        new_identities_vs_canonical=new_vs_canon,
        lost_identities_vs_canonical=lost_vs_canon,
        new_identities_vs_baseline=new_vs_base,
        lost_identities_vs_baseline=lost_vs_base,
        miner_stderr_tail=(stderr[-2000:] if stderr else None),
        baseline_stderr_tail=(base_stderr[-2000:] if base_stderr else None),
        llm_wall_sec=llm_wall,
        out_dir=str(engine_root),
    )

    # 6. Persist summary JSON.
    (engine_root / "h4_summary.json").write_text(
        json.dumps(
            {**asdict(result), "patches_results": [asdict(r) for r in patch_results]},
            indent=2,
            default=str,
        )
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["transformers", "vllm", "tensorrt"],
        help="Which engine(s) to run H4 on.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_PROJECT_ROOT / "research/mining-substrate-trial/findings/hybrid_experiments/h4_modify_miner",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Per-cell parse-retry budget.",
    )
    args = parser.parse_args(argv)

    # Late import to avoid circulars when this module is imported.
    from strategies.h4_engine_configs import ENGINE_CONFIGS

    backend = OllamaBackend()
    print(f"[H4] backend={backend.name} model={backend.model} url={backend.url}")
    print(f"[H4] out_root={args.out_root}")

    results: list[H4Result] = []
    for engine in args.engines:
        cfg = ENGINE_CONFIGS.get(engine)
        if cfg is None:
            print(f"[H4] skipping unknown engine {engine!r}")
            continue
        t0 = time.time()
        try:
            r = run_h4_for_engine(cfg, out_root=args.out_root, backend=backend, max_retries=args.max_retries)
            results.append(r)
        except Exception as exc:
            import traceback

            print(f"[H4 {engine}] CRASH: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
        dt = time.time() - t0
        print(
            f"[H4 {engine}] DONE in {dt:.1f}s | "
            f"diagnoses={r.diagnoses_count} patches_proposed={r.patches_proposed} "
            f"applied={r.patches_applied} | "
            f"canon={r.canonical_count} baseline={r.baseline_count}(ok={r.baseline_run_ok}) "
            f"patched={r.patched_count}(ok={r.patched_run_ok}) | "
            f"recall_vs_baseline={r.invariant_recall_patched_vs_baseline:.3f} "
            f"new_vs_baseline={len(r.new_identities_vs_baseline)} "
            f"lost_vs_baseline={len(r.lost_identities_vs_baseline)}"
        )

    # Aggregate JSON dump.
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "h4_aggregate.json").write_text(
        json.dumps(
            {
                "ran_at": _dt.datetime.utcnow().isoformat() + "Z",
                "results": [
                    {
                        **asdict(r),
                        "patches_results": [asdict(x) for x in r.patches_results],
                    }
                    for r in results
                ],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
