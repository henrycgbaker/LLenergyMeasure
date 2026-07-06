"""Read-only preview funnel for ``llem study plan``.

Given a resolved :class:`~llenergymeasure.config.models.StudyConfig` (produced by
the public ``llenergymeasure.api.load_study`` entry point, which already runs the
real sweep-expansion, validation, and dedup machinery), this module computes and
renders a terraform-style preview:

    declared grid points
      - known-invalid (pruned by engine rules)
      = valid
      - duplicates merged away
      = unique
      x cycles
      = runs

plus an honest wall-clock lower bound from the deterministic thermal gaps only.

Nothing here executes experiments, takes locks, touches the GPU, or writes files -
it only reads counts and metadata off the already-resolved StudyConfig.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llenergymeasure.utils.formatting import format_elapsed

if TYPE_CHECKING:
    from llenergymeasure.config.models import StudyConfig

__all__ = ["PlanFunnel", "RuleAttribution", "build_funnel", "render_funnel"]

# Cap on how many distinct engine-rule attributions get their own line before the
# tail is folded into a single "+ N more" summary line.
_MAX_ATTRIBUTIONS = 5

# Leading "[rule_id] " marker(s) that engine rules prepend to their message, and
# the "Value error, " prefix Pydantic wraps a raised ValueError with. Stripping
# both leaves the plain, human-readable message. The marker can appear more than
# once (a known corpus quirk), so match one-or-more.
_MARKER = re.compile(r"^(?:\[[a-z0-9_]+\]\s*)+")
_VALUE_ERROR_PREFIX = "Value error, "

# Trailing ", got {placeholder}." clause left behind when a rule's message
# template was never interpolated (another corpus quirk). Dropping it keeps a
# stray "{field}" token out of the user-facing preview; a rendered ", got 2.0."
# has no braces and is preserved.
_UNRENDERED_TAIL = re.compile(r",?\s*got \{[^}]*\}\.?$")


@dataclass(frozen=True)
class RuleAttribution:
    """One line of the known-invalid breakdown: a rejecting rule and its count."""

    rule_id: str | None
    count: int
    message: str


@dataclass(frozen=True)
class PlanFunnel:
    """Computed counts for the preview funnel, derived from a resolved StudyConfig."""

    declared_total: int
    valid: int
    skipped: int
    attributions: list[RuleAttribution]
    extra_rule_groups: int
    dedup_enabled: bool
    merged_away: int
    unique: int
    n_cycles: int
    runs: int
    experiment_gap_seconds: float | None
    cycle_gap_seconds: float | None
    n_experiment_gaps: int
    n_cycle_gaps: int


def _plain_message(errors: list[dict[str, Any]], rule_id: str | None) -> str:
    """Best plain-language message for a skipped config's failure.

    Reads the serialised Pydantic error entries (public data on
    ``skipped_configs``) and strips the ``Value error,`` wrapper plus the
    ``[rule_id]`` marker so the researcher sees the underlying human message
    (e.g. ``max_input_len must be >= 1, got 0.``).
    """
    for err in errors:
        msg = str(err.get("msg", ""))
        if msg.startswith(_VALUE_ERROR_PREFIX):
            msg = msg[len(_VALUE_ERROR_PREFIX) :]
        stripped = _MARKER.sub("", msg).strip()
        if stripped:
            cleaned = _UNRENDERED_TAIL.sub("", stripped).strip()
            return cleaned or stripped
    if rule_id is not None:
        return rule_id.replace("_", " ")
    return "other validation errors"


def _attributions(
    skipped_configs: list[dict[str, Any]],
) -> tuple[list[RuleAttribution], int]:
    """Group skipped configs by rejecting rule, top-N with a folded tail.

    Rule-attributed groups are sorted by descending count (ties broken by rule id
    for determinism); at most ``_MAX_ATTRIBUTIONS`` get their own line and the
    rest fold into an ``extra_rule_groups`` count. Non-rule failures (rule_id is
    None) are collapsed into a single trailing "other validation errors" line.
    """
    counts: Counter[str | None] = Counter()
    sample: dict[str | None, str] = {}
    for cfg in skipped_configs:
        rule_id = cfg.get("rule_id")
        counts[rule_id] += 1
        if rule_id not in sample:
            sample[rule_id] = _plain_message(cfg.get("errors", []), rule_id)

    rule_items = sorted(
        ((rid, c) for rid, c in counts.items() if rid is not None),
        key=lambda item: (-item[1], item[0]),
    )
    shown = rule_items[:_MAX_ATTRIBUTIONS]
    extra_rule_groups = len(rule_items) - len(shown)

    result = [RuleAttribution(rid, c, sample[rid]) for rid, c in shown]

    none_count = counts.get(None, 0)
    if none_count:
        result.append(RuleAttribution(None, none_count, "other validation errors"))

    return result, extra_rule_groups


def build_funnel(study: StudyConfig) -> PlanFunnel:
    """Compute the preview funnel from an already-resolved StudyConfig.

    Every number comes straight off ``study`` - the expansion, validation, and
    dedup already ran inside ``load_study``. Declared valid is the count of
    per-declared resolved hashes (parallel to the pre-dedup sweep input); unique
    is the post-dedup, per-cycle experiment count (``runs / n_cycles``).
    """
    execution = study.study_execution
    skipped = len(study.skipped_configs)
    valid = len(study.declared_resolved_config_hashes)
    declared_total = valid + skipped

    dedup_enabled = study.dedup_mode == "resolved"
    n_cycles = execution.n_cycles  # ExecutionConfig validates n_cycles >= 1
    runs = len(study.experiments)
    unique = runs // n_cycles
    merged_away = valid - unique if dedup_enabled else 0

    attributions, extra_rule_groups = _attributions(study.skipped_configs)

    return PlanFunnel(
        declared_total=declared_total,
        valid=valid,
        skipped=skipped,
        attributions=attributions,
        extra_rule_groups=extra_rule_groups,
        dedup_enabled=dedup_enabled,
        merged_away=merged_away,
        unique=unique,
        n_cycles=n_cycles,
        runs=runs,
        experiment_gap_seconds=execution.experiment_gap_seconds,
        cycle_gap_seconds=execution.cycle_gap_seconds,
        n_experiment_gaps=max(runs - 1, 0),
        n_cycle_gaps=max(n_cycles - 1, 0),
    )


def _wall_clock_line(funnel: PlanFunnel) -> str:
    """Honest wall-clock lower bound from the deterministic thermal gaps only.

    The runner waits ``experiment_gap_seconds`` before every experiment after the
    first (``runs - 1`` gaps) and ``cycle_gap_seconds`` at every cycle boundary
    (``n_cycles - 1`` gaps). Per-experiment runtime needs the GPU and is never
    invented here. A gap left unset in the file defers to a machine default we
    cannot see, so it is reported rather than counted as zero.
    """
    if funnel.n_experiment_gaps == 0 and funnel.n_cycle_gaps == 0:
        return "Wall clock: single run - no gaps; per-experiment runtime unknown."

    known = 0.0
    counted_any = False
    deferred: list[str] = []

    if funnel.n_experiment_gaps > 0:
        if funnel.experiment_gap_seconds is not None:
            known += funnel.n_experiment_gaps * funnel.experiment_gap_seconds
            counted_any = True
        else:
            deferred.append("experiment gap uses machine default")

    if funnel.n_cycle_gaps > 0:
        if funnel.cycle_gap_seconds is not None:
            known += funnel.n_cycle_gaps * funnel.cycle_gap_seconds
            counted_any = True
        else:
            deferred.append("cycle gap uses machine default")

    if not counted_any:
        return (
            "Wall clock: no lower bound derivable from this file (gaps unset); "
            "per-experiment runtime unknown."
        )

    line = f"Wall clock: gaps alone >= {format_elapsed(known)}; per-experiment runtime unknown"
    if deferred:
        line += " (" + "; ".join(deferred) + ")"
    return line + "."


def render_funnel(funnel: PlanFunnel, title: str) -> str:
    """Render the funnel as a compact plain-text preview block."""
    nums = [
        funnel.declared_total,
        funnel.skipped,
        funnel.valid,
        funnel.merged_away,
        funnel.unique,
        funnel.n_cycles,
        funnel.runs,
    ]
    width = max(len(str(n)) for n in nums)

    def row(symbol: str, label: str, value: int) -> str:
        return f"  {symbol} {label:<14}{value:>{width}}"

    lines = [f"Study plan: {title}", ""]
    lines.append(row(" ", "declared", funnel.declared_total))
    lines.append(row("-", "known-invalid", funnel.skipped))
    lines.append(row("=", "valid", funnel.valid))
    if funnel.dedup_enabled:
        lines.append(row("-", "dedup-merged", funnel.merged_away))
        lines.append(row("=", "unique", funnel.unique))
    else:
        lines.append("    dedup disabled - every declared config runs")
    lines.append(row("x", "cycles", funnel.n_cycles))
    lines.append(row("=", "runs", funnel.runs))
    lines.append("")

    if funnel.skipped:
        lines.append(f"Known-invalid breakdown ({funnel.skipped}), by engine rule:")
        for attribution in funnel.attributions:
            tag = f"  (rule {attribution.rule_id})" if attribution.rule_id else ""
            lines.append(f"    {attribution.count}x  {attribution.message}{tag}")
        if funnel.extra_rule_groups:
            lines.append(f"    + {funnel.extra_rule_groups} more engine rule(s)")
        lines.append("")

    lines.append(_wall_clock_line(funnel))
    return "\n".join(lines)
