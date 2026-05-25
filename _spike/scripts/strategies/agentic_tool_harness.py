"""Agentic tool-harness for H7 (Phase 3b hybrid experiment).

H7 tests whether closed-loop tool-mediated iteration changes the ceiling
established by single-shot patterns. Ollama has no native tool-use, so
this module builds a faux tool-call loop:

1. System prompt explains the task + available tools (with signatures
   + JSON-formatted call protocol).
2. Each turn:
   - LLM emits a JSON object: ``{"tool": "<name>", "args": {...}}``.
   - Harness validates, dispatches, returns the result as JSON.
   - Tool-call + result get appended to the conversation history.
3. Budget enforcement: at most 30 tool calls OR 30 min wall-clock per
   cell. Whichever fires first ends the loop. The LLM's most recent
   ``finalise`` call is the cell output (or empty if never finalised).

Tools exposed:

- ``read_file(path: str, line_start: int = 0, line_end: int = -1) -> str``
- ``list_validators(engine: str, class_name: str) -> list[str]``
- ``run_miner(engine: str, version_slug: str) -> dict``
- ``score_against(emitted_yaml: str, engine: str, version_slug: str) -> dict``
- ``finalise(invariants_yaml: str) -> None``

Sandboxing: ``read_file`` clamps to a set of permitted roots (engine
source tree + venv source + the trial worktree). Path traversal
attempts (``..``, abs paths outside the roots) are rejected.

This is INFRASTRUCTURE - reusable for future agentic patterns. The H7
runner lives in ``run_h7_agentic.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root sits three parents above _spike/scripts/strategies/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from _spike.scripts.strategies.llm_extractor import OllamaBackend  # noqa: E402
from _spike.scripts.trial_scoring import (  # noqa: E402
    ScoringConfig,
    score_invariants,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


MAX_READ_LINES = 200
"""Maximum lines per ``read_file`` call. Bounded to keep context manageable."""

DEFAULT_MAX_TOOL_CALLS = 30
"""Hard cap on tool calls per cell (per H7 spec)."""

DEFAULT_WALL_CLOCK_BUDGET_SEC = 30 * 60
"""Hard cap on wall-clock per cell (30 min, per H7 spec)."""

DEFAULT_MAX_OUTPUT_TOKENS = 4096
"""LLM per-turn output token cap. Tool-call JSON should be much shorter
than this; finalise's invariants_yaml can be larger."""

DEFAULT_HISTORY_CHAR_CAP = 80_000
"""Approximate char budget for conversation history. When exceeded, the
oldest tool results get truncated (kept-as-summary) to stay under ctx."""


# ---------------------------------------------------------------------------
# Tool result envelope
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """One tool invocation's outcome."""

    tool: str
    args: dict[str, Any]
    ok: bool
    result: Any = None
    """Tool-specific payload. For read_file: string text. For
    list_validators: list[str]. For run_miner / score_against: dict.
    For finalise: None."""

    error: str = ""
    elapsed_sec: float = 0.0
    """Wall-clock for the tool dispatch itself (excludes LLM time)."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class AgentTurn:
    """One turn of the agentic loop: LLM emission + tool dispatch."""

    turn_index: int
    llm_raw_response: str
    llm_elapsed_sec: float
    parsed_call: dict[str, Any] | None
    parse_error: str = ""
    tool_result: ToolResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "turn_index": self.turn_index,
            "llm_raw_response": self.llm_raw_response,
            "llm_elapsed_sec": self.llm_elapsed_sec,
            "parsed_call": self.parsed_call,
            "parse_error": self.parse_error,
        }
        d["tool_result"] = self.tool_result.to_dict() if self.tool_result else None
        return d


@dataclass
class LoopResult:
    """The full loop's outcome for one cell."""

    engine: str
    version_slug: str
    turns: list[AgentTurn] = field(default_factory=list)
    finalised: bool = False
    final_yaml: str | None = None
    """The last ``finalise.args.invariants_yaml`` the LLM submitted."""
    total_wall_sec: float = 0.0
    total_llm_sec: float = 0.0
    total_tool_calls: int = 0
    """Counts only successful tool-call attempts that parsed + dispatched."""
    stop_reason: str = ""
    """One of: ``finalised``, ``max_calls``, ``wall_clock``, ``crash``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "version_slug": self.version_slug,
            "finalised": self.finalised,
            "final_yaml": self.final_yaml,
            "total_wall_sec": self.total_wall_sec,
            "total_llm_sec": self.total_llm_sec,
            "total_tool_calls": self.total_tool_calls,
            "stop_reason": self.stop_reason,
            "turns": [t.to_dict() for t in self.turns],
        }


# ---------------------------------------------------------------------------
# Sandboxed path resolution
# ---------------------------------------------------------------------------


@dataclass
class ReadSandbox:
    """Permitted roots for ``read_file``. Paths are resolved against the
    first matching root; absolute paths outside the roots are rejected."""

    roots: list[Path] = field(default_factory=list)
    """Ordered list of permitted directory roots."""

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` against the sandbox.

        - If ``path`` is absolute and inside a root: accept it directly.
        - If relative: try each root in order; accept the first that
          contains a file at ``root / path``.
        - Reject ``..`` traversal and paths outside any root.

        Raises ``PermissionError`` on rejection; ``FileNotFoundError`` if
        the resolved path doesn't exist.
        """
        p = Path(path)
        if p.is_absolute():
            try:
                resolved = p.resolve()
            except OSError as exc:
                raise FileNotFoundError(f"cannot resolve {path!r}: {exc}") from exc
            for root in self.roots:
                try:
                    resolved.relative_to(root.resolve())
                    if resolved.exists():
                        return resolved
                    raise FileNotFoundError(f"path not found: {resolved}")
                except ValueError:
                    continue
            raise PermissionError(
                f"path {path!r} outside sandbox roots {[str(r) for r in self.roots]}"
            )
        # Relative path
        if ".." in p.parts:
            raise PermissionError(f"path traversal forbidden: {path!r}")
        for root in self.roots:
            candidate = (root / p).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"path {path!r} not found in any sandbox root ({[str(r) for r in self.roots]})"
        )


# ---------------------------------------------------------------------------
# Tool dispatchers
# ---------------------------------------------------------------------------


def tool_read_file(args: dict[str, Any], *, sandbox: ReadSandbox) -> str:
    """Read a file (bounded by line range, capped at MAX_READ_LINES)."""
    path_arg = args.get("path")
    if not isinstance(path_arg, str) or not path_arg.strip():
        raise ValueError("read_file: `path` (string) is required")
    line_start = int(args.get("line_start", 0) or 0)
    line_end_raw = args.get("line_end", -1)
    line_end = int(-1 if line_end_raw is None else line_end_raw)

    resolved = sandbox.resolve(path_arg)
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {resolved}")

    text = resolved.read_text(errors="replace")
    lines = text.splitlines()
    total = len(lines)
    if line_start < 0:
        line_start = max(0, total + line_start)
    if line_end == -1 or line_end > total:
        line_end = total
    # Cap span at MAX_READ_LINES
    span = line_end - line_start
    if span > MAX_READ_LINES:
        line_end = line_start + MAX_READ_LINES
        truncated_note = f"\n... [TRUNCATED: file has {total} lines; showed lines {line_start}-{line_end} of {total}; re-call with new range to see more]"
    elif line_end < total:
        truncated_note = f"\n... [showed lines {line_start}-{line_end} of {total}; re-call with new range to see more]"
    else:
        truncated_note = ""

    rendered: list[str] = []
    for i in range(line_start, line_end):
        rendered.append(f"{i + 1:5d}  {lines[i]}")
    return "\n".join(rendered) + truncated_note


def tool_list_validators(args: dict[str, Any], *, sandbox: ReadSandbox) -> list[str]:
    """Return method names matching ``_verify_*`` or ``__post_init__``
    for a class in the engine's source. Uses AST parsing; doesn't import.

    Searches across all .py files in the sandboxed engine source root.
    """
    engine = args.get("engine")
    class_name = args.get("class_name")
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError("list_validators: `engine` (string) is required")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("list_validators: `class_name` (string) is required")
    if not sandbox.roots:
        raise RuntimeError("list_validators: sandbox has no roots configured")

    # Scan ALL python files in the engine sandbox root(s). For perf we
    # rg-walk via os.walk but cap recursion at the first 5000 files.
    method_names: list[str] = []
    scanned_files = 0
    for root in sandbox.roots:
        if not root.exists() or not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            scanned_files += 1
            if scanned_files > 5000:
                break
            try:
                tree = ast.parse(py.read_text(errors="replace"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if node.name != class_name:
                    continue
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = member.name
                        if name == "__post_init__" or name.startswith("_verify_"):
                            method_names.append(name)
        if scanned_files > 5000:
            break
    # Dedupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for n in method_names:
        if n not in seen:
            unique.append(n)
            seen.add(n)
    return unique


def tool_run_miner(
    args: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Return summary of the engine's STATIC MINER output.

    For H7 we serve the pre-mined active-row artefact (the canonical
    invariants.proposed.yaml in ``engine_versions/<engine>/<version>/
    outputs/``) rather than re-invoking the walker subprocess. The walker
    is already deterministic; re-running it via the trial's subprocess
    harness has known dep-wall issues for vllm + tensorrt. Returning the
    artefact summary gives the LLM the same information without the
    subprocess flakiness.

    Returns: {count, namespaces (list), invariant_ids (list), source_path}.
    """
    import yaml as _yaml

    engine = args.get("engine")
    version_slug = args.get("version_slug")
    if not isinstance(engine, str) or not isinstance(version_slug, str):
        raise ValueError("run_miner: `engine` and `version_slug` (strings) required")
    invariants_path = (
        project_root
        / "engine_versions"
        / engine
        / version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    if not invariants_path.exists():
        raise FileNotFoundError(
            f"run_miner: no mined invariants at {invariants_path}; engine/version unknown?"
        )
    doc = _yaml.safe_load(invariants_path.read_text()) or {}
    invariants = doc.get("invariants") or []
    namespaces: set[str] = set()
    ids: list[str] = []
    for inv in invariants:
        if not isinstance(inv, dict):
            continue
        iid = inv.get("id", "")
        if iid:
            ids.append(iid)
        match = inv.get("match") or {}
        fields = match.get("fields") or {}
        if isinstance(fields, dict) and fields:
            first_key = next(iter(fields.keys()), "")
            if "." in first_key:
                namespaces.add(first_key.rsplit(".", 1)[0])
    return {
        "count": len(invariants),
        "namespaces": sorted(namespaces),
        "invariant_ids": ids,
        "source_path": str(invariants_path.relative_to(project_root)),
    }


def tool_score_against(
    args: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Score emitted YAML against the active reference.

    Returns dict matching CellScore-ish shape: recall, precision,
    severity_accuracy, intersection_count, ref_count, cell_count,
    failure_mode, recall_misses (count + sample), precision_spurious
    (count + sample).
    """
    import yaml as _yaml

    emitted_yaml = args.get("emitted_yaml")
    engine = args.get("engine")
    version_slug = args.get("version_slug")
    if not isinstance(emitted_yaml, str) or not emitted_yaml.strip():
        raise ValueError("score_against: `emitted_yaml` (non-empty string) required")
    if not isinstance(engine, str) or not isinstance(version_slug, str):
        raise ValueError("score_against: `engine` and `version_slug` (strings) required")

    try:
        emitted_doc = _yaml.safe_load(emitted_yaml) or {}
    except Exception as exc:
        raise ValueError(f"score_against: YAML parse failure: {exc}") from exc
    if not isinstance(emitted_doc, dict):
        raise ValueError(
            f"score_against: emitted YAML root is not a mapping ({type(emitted_doc).__name__})"
        )

    reference_path = (
        project_root
        / "engine_versions"
        / engine
        / version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    if not reference_path.exists():
        raise FileNotFoundError(f"score_against: no reference at {reference_path}")
    ref_doc = _yaml.safe_load(reference_path.read_text()) or {}

    cfg = ScoringConfig()
    (
        recall,
        precision,
        severity_accuracy,
        failure_mode,
        ref_count,
        cell_count,
        int_count,
        recall_misses,
        precision_spurious,
        severity_mismatches,
    ) = score_invariants(
        reference_invariants=ref_doc,
        cell_invariants=emitted_doc,
        config=cfg,
    )

    # Sample first 10 of each diff list so the LLM can self-correct
    # without getting flooded by a 60-entry diff dump.
    def _summarise_diffs(diffs: list, kind: str) -> list[dict]:
        out: list[dict] = []
        for d in diffs[:10]:
            out.append(
                {
                    "identity": list(d.identity),
                    "kind": d.kind,
                    "note": d.note,
                }
            )
        return out

    return {
        "recall": recall,
        "precision": precision,
        "severity_accuracy": severity_accuracy,
        "failure_mode": str(failure_mode.value if hasattr(failure_mode, "value") else failure_mode),
        "ref_count": ref_count,
        "cell_count": cell_count,
        "intersection_count": int_count,
        "recall_misses_total": len(recall_misses),
        "recall_misses_sample": _summarise_diffs(recall_misses, "recall_miss"),
        "precision_spurious_total": len(precision_spurious),
        "precision_spurious_sample": _summarise_diffs(precision_spurious, "precision_spurious"),
        "severity_mismatches_total": len(severity_mismatches),
    }


def tool_finalise(args: dict[str, Any], *, state: dict[str, Any]) -> None:
    """Mark the loop done + save the final invariants YAML in state."""
    invariants_yaml = args.get("invariants_yaml")
    if not isinstance(invariants_yaml, str) or not invariants_yaml.strip():
        raise ValueError("finalise: `invariants_yaml` (non-empty string) required")
    state["finalised"] = True
    state["final_yaml"] = invariants_yaml


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------


def parse_tool_call(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the first JSON object from the LLM response.

    Returns ``(parsed, error_msg)``. If parsing fails, ``parsed`` is None
    and ``error_msg`` is a human-readable description for the LLM-feedback
    retry prompt.

    Robust to:
    - Markdown code fences (```json ... ``` or ``` ... ```).
    - Leading commentary text.
    - Multiple JSON objects (takes the first).
    - Trailing prose after the JSON.
    """
    if not raw or not raw.strip():
        return None, "empty response from LLM"
    text = raw.strip()
    # Strip code fences if present
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the first balanced { ... } JSON object using a brace counter
    start = text.find("{")
    if start == -1:
        return None, "no JSON object found in response (no '{')"
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None, "unbalanced braces; JSON object incomplete"
    candidate = text[start:end]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failure: {exc.msg} at pos {exc.pos}"
    if not isinstance(parsed, dict):
        return None, "parsed value is not a JSON object"
    if "tool" not in parsed:
        return None, "JSON object missing required field `tool`"
    if not isinstance(parsed["tool"], str):
        return None, "field `tool` must be a string"
    args = parsed.get("args", {})
    if not isinstance(args, dict):
        return None, "field `args` must be an object (or absent)"
    return {"tool": parsed["tool"], "args": args}, ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert invariant-extraction engineer. Your task is to produce a
catalogue of validation invariants for inference engine `{engine}` version
`{version_slug}`.

An "invariant" is a structural rule the engine enforces at config validation
time (e.g. "GenerationConfig raises when num_beams < 1", "ModelConfig
normalises tokenizer_mode lowercase"). Each invariant maps to ONE field in
the engine's class hierarchy plus a predicate kind (exact / not_in /
not_equal / gt / lt / ge / le / type_is_not / present).

The reference catalogue lives at:
  engine_versions/{engine}/{version_slug}/outputs/invariants.proposed.yaml

You have access to TOOLS. Each turn, emit a SINGLE JSON object describing
ONE tool call. After the tool result comes back, decide what to do next.
Budget: at most {max_calls} tool calls OR {max_wall_min} minutes wall-clock,
whichever fires first.

== AVAILABLE TOOLS ==

1. read_file(path: str, line_start: int = 0, line_end: int = -1) -> str
   Read a slice of a file. Bounded by line range; max {max_read_lines} lines
   per call. Use line ranges to navigate large files iteratively.
   Permitted paths: the engine source tree at the venv root, and the
   engine_versions tree.

2. list_validators(engine: str, class_name: str) -> list[str]
   Return method names matching `_verify_*` or `__post_init__` for the
   class. AST-parsed; doesn't import.

3. run_miner(engine: str, version_slug: str) -> dict
   Returns summary of the existing static-miner output: count, namespaces,
   invariant_ids. Useful for seeing what (a) already covers so you can
   focus on gaps OR mirror its identity shape.

4. score_against(emitted_yaml: str, engine: str, version_slug: str) -> dict
   Score emitted YAML against the active reference. Returns recall,
   precision, intersection counts + sampled diffs. USE THIS MID-LOOP
   to validate your work before finalising.

5. finalise(invariants_yaml: str) -> None
   Submit the final invariants YAML. The loop ends. Your finalised YAML
   is the cell output.

== OUTPUT YAML SHAPE ==

The reference uses this top-level shape:

  schema_version: 1.0.0
  engine: {engine}
  engine_version: <version>
  invariants:
    - id: <unique_id>
      engine: {engine}
      library: {engine}
      invariant_under_test: <short description>
      severity: <error | warning | dormant>
      native_type: <{engine}.ClassName>
      miner_source:
        path: <file/path/in/source>
        method: <method_name>
        line_at_scan: <int>
      match:
        engine: {engine}
        fields:
          <namespace>.<field>: <expected_value_or_predicate_dict>
      kwargs_positive: {{<kwargs that trigger>}}
      kwargs_negative: {{<kwargs that don't trigger>}}
      expected_outcome: <error | warn | dormant_announced>

The `match.fields` identity is what scoring compares. A predicate dict
like `{{not_equal: <X>}}`, `{{not_in: [...]}}`, `{{">": 0}}`, etc. is
allowed in the field value position.

== JSON CALL PROTOCOL ==

Every response from you must be a single JSON object of the form:

  {{"tool": "<name>", "args": {{<argname>: <value>, ...}}}}

No prose. No markdown code fences. Just the JSON object.

If your previous response failed to parse, the harness will tell you why
in the next turn. Re-emit the call with the fix.

== STRATEGY HINTS ==

- Start by calling `run_miner` to see what (a) covers. Then `read_file`
  the engine source for classes you want to add invariants for, or call
  `list_validators` to discover where to look.
- Build your invariants list incrementally. Call `score_against` with a
  draft YAML to see where you are; iterate.
- When recall+precision look right, call `finalise` to submit.
- DO NOT call `read_file` on the reference invariants.proposed.yaml -
  that would be cheating. The reference is what we score AGAINST, not
  what you mine FROM.

Begin. Your first tool call:
""".strip()


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------


def render_tool_result_for_llm(turn: AgentTurn) -> str:
    """Render a tool result as text for inclusion in the next prompt."""
    if turn.tool_result is None:
        # Parse error case
        return f"<<TOOL_CALL_PARSE_ERROR turn={turn.turn_index}: {turn.parse_error}>>"
    r = turn.tool_result
    if not r.ok:
        return f"<<TOOL_RESULT turn={turn.turn_index} tool={r.tool} ok=false error={r.error!r}>>"
    # Render result as JSON for the LLM. Truncate long strings.
    payload: Any = r.result
    if isinstance(payload, str) and len(payload) > 30_000:
        payload = payload[:30_000] + f"\n... [truncated {len(payload) - 30_000} chars]"
    try:
        payload_json = json.dumps(payload, default=str)
    except Exception:
        payload_json = json.dumps(str(payload))
    return (
        f"<<TOOL_RESULT turn={turn.turn_index} tool={r.tool} ok=true "
        f"elapsed_sec={r.elapsed_sec:.3f}>>\n{payload_json}"
    )


def render_history(turns: list[AgentTurn], char_cap: int = DEFAULT_HISTORY_CHAR_CAP) -> str:
    """Render conversation history as text, truncating oldest entries if
    needed to fit within char_cap. Tool reads are the bulkiest; truncation
    summarises them rather than fully dropping."""
    # Render newest-to-oldest, then reverse so newest is at the bottom.
    cumulative = 0
    rendered_rev: list[str] = []
    for turn in reversed(turns):
        # LLM's call
        call_text = f"-- turn {turn.turn_index} --\nASSISTANT: {turn.llm_raw_response.strip()}\n"
        result_text = render_tool_result_for_llm(turn)
        block = call_text + result_text + "\n"
        cumulative += len(block)
        if cumulative > char_cap and rendered_rev:
            # Stop adding older turns; emit a summary placeholder
            rendered_rev.append("[... OLDER TURNS TRUNCATED FOR CONTEXT BUDGET ...]\n")
            break
        rendered_rev.append(block)
    rendered_rev.reverse()
    return "".join(rendered_rev)


# ---------------------------------------------------------------------------
# LLM transport contract
# ---------------------------------------------------------------------------


class LLMCallable:
    """Minimal protocol for an LLM transport. Used by ``AgenticLoop``.

    Must support ``call(prompt: str) -> (text: str, elapsed_sec: float)``.
    """

    def call(self, prompt: str) -> tuple[str, float]:
        raise NotImplementedError


@dataclass
class OllamaLLM(LLMCallable):
    """Adapter around OllamaBackend for the agentic loop."""

    backend: OllamaBackend
    timeout: int = 600
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def call(self, prompt: str) -> tuple[str, float]:
        # No json_mode here - the LLM needs the freedom to emit either a
        # bare JSON object OR an object inside a code fence. parse_tool_call
        # handles both shapes robustly.
        return self.backend.call(
            prompt,
            json_mode=False,
            timeout=self.timeout,
            max_output_tokens=self.max_output_tokens,
        )


@dataclass
class StubLLM(LLMCallable):
    """Deterministic LLM for tests: returns canned responses in order."""

    responses: list[str] = field(default_factory=list)
    _idx: int = 0

    def call(self, prompt: str) -> tuple[str, float]:
        if self._idx >= len(self.responses):
            # Default tail behaviour: finalise with empty invariants
            return '{"tool": "finalise", "args": {"invariants_yaml": "invariants: []"}}', 0.0
        text = self.responses[self._idx]
        self._idx += 1
        return text, 0.0


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------


@dataclass
class AgenticLoop:
    """Run an LLM-driven tool-use loop for one cell.

    Holds the LLM transport, tool sandbox, project root, and budget caps.
    Call ``run(engine, version_slug, additional_context)`` to execute.
    """

    llm: LLMCallable
    sandbox: ReadSandbox
    project_root: Path
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    wall_clock_budget_sec: int = DEFAULT_WALL_CLOCK_BUDGET_SEC
    history_char_cap: int = DEFAULT_HISTORY_CHAR_CAP
    """Set lower in tests; raise to taste in production."""

    def _build_initial_prompt(
        self,
        *,
        engine: str,
        version_slug: str,
        additional_context: str = "",
    ) -> str:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            engine=engine,
            version_slug=version_slug,
            max_calls=self.max_tool_calls,
            max_wall_min=int(self.wall_clock_budget_sec / 60),
            max_read_lines=MAX_READ_LINES,
        )
        if additional_context.strip():
            system_prompt += "\n\n== ADDITIONAL CONTEXT ==\n" + additional_context.strip()
        return system_prompt

    def _dispatch_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        state: dict[str, Any],
    ) -> ToolResult:
        """Dispatch one tool call. Returns a ToolResult (ok=False on error)."""
        t0 = time.time()
        try:
            if tool_name == "read_file":
                payload: Any = tool_read_file(args, sandbox=self.sandbox)
            elif tool_name == "list_validators":
                payload = tool_list_validators(args, sandbox=self.sandbox)
            elif tool_name == "run_miner":
                payload = tool_run_miner(args, project_root=self.project_root)
            elif tool_name == "score_against":
                payload = tool_score_against(args, project_root=self.project_root)
            elif tool_name == "finalise":
                tool_finalise(args, state=state)
                payload = None
            else:
                return ToolResult(
                    tool=tool_name,
                    args=args,
                    ok=False,
                    error=f"unknown tool: {tool_name!r}; available: read_file, list_validators, run_miner, score_against, finalise",
                    elapsed_sec=time.time() - t0,
                )
        except Exception as exc:
            return ToolResult(
                tool=tool_name,
                args=args,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_sec=time.time() - t0,
            )
        return ToolResult(
            tool=tool_name,
            args=args,
            ok=True,
            result=payload,
            elapsed_sec=time.time() - t0,
        )

    def run(
        self,
        *,
        engine: str,
        version_slug: str,
        additional_context: str = "",
    ) -> LoopResult:
        """Run the agentic loop for one cell. Returns LoopResult."""
        result = LoopResult(engine=engine, version_slug=version_slug)
        state: dict[str, Any] = {"finalised": False, "final_yaml": None}

        initial_prompt = self._build_initial_prompt(
            engine=engine,
            version_slug=version_slug,
            additional_context=additional_context,
        )
        t_start = time.time()
        turn_index = 0

        while True:
            # Budget checks
            elapsed = time.time() - t_start
            if elapsed >= self.wall_clock_budget_sec:
                result.stop_reason = "wall_clock"
                break
            if result.total_tool_calls >= self.max_tool_calls:
                result.stop_reason = "max_calls"
                break

            # Build prompt: initial system + history
            history_text = render_history(result.turns, char_cap=self.history_char_cap)
            prompt = (
                initial_prompt
                + ("\n\n== CONVERSATION HISTORY ==\n" + history_text if history_text else "")
                + "\n\nEmit your next tool call (single JSON object, no prose):\n"
            )

            # LLM call
            try:
                raw, llm_dt = self.llm.call(prompt)
            except Exception as exc:
                # LLM transport crash. Record + stop.
                turn = AgentTurn(
                    turn_index=turn_index,
                    llm_raw_response=f"<<LLM_TRANSPORT_ERROR: {type(exc).__name__}: {exc}>>",
                    llm_elapsed_sec=0.0,
                    parsed_call=None,
                    parse_error=f"transport_error: {type(exc).__name__}: {exc}",
                )
                result.turns.append(turn)
                result.stop_reason = "crash"
                break
            result.total_llm_sec += llm_dt

            parsed, parse_err = parse_tool_call(raw)
            turn = AgentTurn(
                turn_index=turn_index,
                llm_raw_response=raw,
                llm_elapsed_sec=llm_dt,
                parsed_call=parsed,
                parse_error=parse_err,
            )

            if parsed is None:
                # Parse failed; give LLM one feedback turn then count it as a call.
                # The history-render machinery shows the parse_error to the LLM
                # so it can re-emit. We don't dispatch any tool.
                result.turns.append(turn)
                turn_index += 1
                # Parse errors DO count toward the budget so a stuck LLM
                # can't infinitely loop on malformed JSON.
                result.total_tool_calls += 1
                continue

            # Dispatch tool
            tool_name = parsed["tool"]
            args = parsed.get("args", {})
            tr = self._dispatch_tool(tool_name, args, state=state)
            turn.tool_result = tr
            result.turns.append(turn)
            result.total_tool_calls += 1
            turn_index += 1

            if state["finalised"]:
                result.finalised = True
                result.final_yaml = state["final_yaml"]
                result.stop_reason = "finalised"
                break

        result.total_wall_sec = time.time() - t_start
        if not result.stop_reason:
            result.stop_reason = "unknown"
        return result


# ---------------------------------------------------------------------------
# Convenience: build a sandbox for the standard trial layout
# ---------------------------------------------------------------------------


def build_default_sandbox(*, project_root: Path, engine: str, version_slug: str) -> ReadSandbox:
    """Construct the standard sandbox roots for an H7 cell:

    - venv source root for the engine version
    - the engine_versions tree (so the LLM can see producers etc., but
      we DO NOT include the reference invariants.proposed.yaml in the
      "encouraged paths" - the system prompt tells the LLM not to read it)
    """
    venv_roots = {
        ("transformers", "v4_57_3"): Path("/tmp/trial_transformers_v4_57_6_venv/src/transformers"),
        ("vllm", "v0_7_3"): Path("/tmp/trial_vllm_v0_7_3_venv/src/vllm"),
        ("tensorrt", "v0_21_0"): Path("/tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt"),
    }
    venv_root = venv_roots.get((engine, version_slug))
    roots = [
        project_root / "engine_versions" / engine / version_slug,
    ]
    if venv_root is not None and venv_root.exists():
        roots.insert(0, venv_root)
    return ReadSandbox(roots=roots)
