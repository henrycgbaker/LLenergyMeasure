"""Pre-flight study display: Rich panel and text summary.

Renders the study preflight panel (execution controls, engines, workload,
sweep breakdown) and the plain-text summary shown before a study runs. Pure
presentation - reads resolved StudyConfig + (optional) resolved RunnerSpecs
and produces display artefacts only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text

from llenergymeasure.config.introspection import (
    get_display_label,
    get_field_role,
    get_swept_field_paths,
)
from llenergymeasure.config.models import (
    DatasetConfig,
    MeasurementConfig,
    TaskConfig,
)
from llenergymeasure.config.ssot import SOURCE_MULTI_ENGINE_ELEVATION

if TYPE_CHECKING:
    from llenergymeasure.config.models import StudyConfig
    from llenergymeasure.infra.runner_resolution import RunnerSpec


def build_preflight_panel(
    study_config: StudyConfig,
    runner_specs: dict[str, RunnerSpec] | None = None,
    study_dir: Path | None = None,
    probed_energy_sampler: str | None = None,
    sweep_axes: int | None = None,
    sweep_groups: int | None = None,
    n_explicit: int = 0,
) -> Panel:
    """Return a Rich Panel with study preflight summary.

    The panel shows:
    - Border title: "Study: <name>"
    - Execution Controls: experiments, experiment order, gaps, shuffle seed
    - Workload: all workload fields; swept fields annotated with "+"
    - Engines: per-engine runner mode with auto-elevation annotation
    - Sweep: summary line with axis/group counts and unique configs
    - Dimmed design hash and results path at the bottom

    Field labels come from json_schema_extra display_label metadata (SSOT).
    Declared values display normally; defaulted values are dimmed.

    When ``runner_specs`` is provided (resolved by pre-flight), the panel shows
    effective runner modes. Otherwise falls back to YAML-declared runners.

    When ``sweep_axes`` / ``sweep_groups`` are provided (from the raw YAML
    sweep dict via ``count_sweep_structure``), the Sweep section shows the
    breakdown.  Otherwise falls back to counting varying field paths.

    Skipped configs are NOT included; callers display them separately.
    """
    exec_cfg = study_config.study_execution
    n_cycles = exec_cfg.n_cycles
    n_runs = len(study_config.experiments)
    n_configs = n_runs // n_cycles if n_cycles > 0 else n_runs
    hash_display = study_config.study_design_hash or "unknown"
    experiments = study_config.experiments

    # --- Pluralisation helper ---
    def _pl(n: int, singular: str, plural: str | None = None) -> str:
        if n == 1:
            return f"{n} {singular}"
        return f"{n} {plural or singular + 's'}"

    # --- Helpers ---
    def _section(body: Text, title: str) -> None:
        body.append("\n  ")
        body.append(title, style="bold")
        body.append("\n")

    def _line(
        body: Text,
        label: str,
        value: str,
        indent: int = 4,
        value_style: str = "dim",
    ) -> None:
        body.append(f"{' ' * indent}")
        body.append(f"{label:<18}", style="white")
        body.append(f"{value}\n", style=value_style)

    # --- Unique engines (for Engines section) ---
    unique_engines = sorted({exp.engine for exp in experiments})

    # --- Resolve energy sampler display ---
    unique_energy = sorted(
        {
            str(exp.measurement.energy_sampler)
            if exp.measurement.energy_sampler is not None
            else "disabled"
            for exp in experiments
        }
    )
    all_docker = runner_specs is not None and all(
        spec.mode == "docker" for spec in runner_specs.values()
    )
    energy_display = _resolve_energy_display(
        unique_energy, probed_sampler=probed_energy_sampler, skip_probe=all_docker
    )

    # --- Swept field paths (for annotation) ---
    swept_paths = get_swept_field_paths(experiments)

    # --- Assemble body ---
    body = Text()
    experiments_line = (
        f"{_pl(n_configs, 'config')} x {_pl(n_cycles, 'cycle')} = {_pl(n_runs, 'run')}"
    )

    # -- Execution Controls --
    _section(body, "Execution Controls")
    _line(body, "Experiments", experiments_line)
    _line(body, "Experiment order", str(exec_cfg.experiment_order))
    exp_gap = (
        f"{exec_cfg.experiment_gap_seconds}s"
        if exec_cfg.experiment_gap_seconds is not None
        else "0s"
    )
    cyc_gap = f"{exec_cfg.cycle_gap_seconds}s" if exec_cfg.cycle_gap_seconds is not None else "0s"
    _line(body, "Experiment gap", exp_gap)
    _line(body, "Cycle gap", cyc_gap)
    shuffle_val = str(exec_cfg.shuffle_seed) if exec_cfg.shuffle_seed is not None else "auto"
    _line(body, "Shuffle seed", shuffle_val)
    skip_val = "yes" if exec_cfg.skip_preflight else "no"
    _line(body, "Skip preflight", skip_val)

    # -- Engines --
    _section(body, "Engines")
    yaml_runners = study_config.runners or {}
    for b in unique_engines:
        if runner_specs and b in runner_specs:
            spec = runner_specs[b]
            body.append("    ")
            body.append(f"{b:<18}", style="white")
            body.append(f"{spec.mode}", style="dim")
            if getattr(spec, "source", None) == SOURCE_MULTI_ENGINE_ELEVATION:
                body.append(" (auto-elevated)", style="yellow")
            body.append("\n")
            # Show image resolution for Docker engines
            if spec.mode == "docker" and spec.image:
                body.append("    ", style="dim")
                body.append(f"\u21b3 {spec.image}\n", style="dim")
        else:
            mode_str = str(yaml_runners.get(b, "local"))
            _line(body, b, mode_str)

    # -- Workload section --
    # Task fields + energy sampler. Swept fields annotated with "+" and bold.
    workload_rows: list[tuple[str, str, bool, bool]] = []  # (label, value, is_declared, is_swept)

    first_exp = experiments[0]
    task_declared = first_exp.task.model_fields_set

    for field_name, fi in TaskConfig.model_fields.items():
        if field_name == "dataset":
            dataset_first = first_exp.task.dataset
            dataset_declared = dataset_first.model_fields_set
            for ds_field, ds_fi in DatasetConfig.model_fields.items():
                ds_role = get_field_role(ds_fi)
                if ds_role != "workload":
                    continue
                ds_path = f"task.dataset.{ds_field}"
                is_swept = ds_path in swept_paths
                unique_vals = sorted(
                    {str(getattr(exp.task.dataset, ds_field)) for exp in experiments}
                )
                val_str = ", ".join(unique_vals)
                is_decl = ds_field in dataset_declared
                label = get_display_label(ds_fi, ds_field)
                workload_rows.append((label, val_str, is_decl, is_swept))
            continue

        task_path = f"task.{field_name}"
        is_swept = task_path in swept_paths
        unique_vals = sorted({str(getattr(exp.task, field_name)) for exp in experiments})
        val_str = ", ".join(unique_vals)
        label = get_display_label(fi, field_name)
        is_decl = field_name in task_declared
        workload_rows.append((label, val_str, is_decl, is_swept))

    # Energy sampler (from measurement)
    energy_fi = MeasurementConfig.model_fields["energy_sampler"]
    is_swept = "measurement.energy_sampler" in swept_paths
    label = get_display_label(energy_fi, "energy_sampler")
    workload_rows.append((label, energy_display, True, is_swept))

    if workload_rows:
        _section(body, "Workload")
        for label, val_str, is_decl, is_swept in workload_rows:
            annotation = " +" if is_swept else ""
            if is_swept:
                _line(body, label, f"{val_str}{annotation}", value_style="bold")
            elif is_decl:
                _line(body, label, val_str)
            else:
                _line(body, label, val_str, value_style="dim")

    # -- Sweep summary (only when multiple configs) --
    if n_configs > 1:
        _section(body, "Sweep")
        n_from_sweep = n_configs - n_explicit
        has_both = n_from_sweep > 0 and n_explicit > 0
        # Sweep-generated line
        if n_from_sweep > 0 and sweep_axes is not None and sweep_groups is not None:
            if sweep_groups > 0:
                sweep_line = (
                    f"{sweep_axes} axes . {sweep_groups} groups "
                    f"-> {_pl(n_from_sweep, 'config')} from sweep"
                )
            else:
                sweep_line = f"{sweep_axes} axes -> {_pl(n_from_sweep, 'config')} from sweep"
            body.append(f"    {sweep_line}\n")
        elif n_from_sweep > 0:
            n_dims = len(swept_paths)
            sweep_line = f"{_pl(n_dims, 'dimension')} -> {_pl(n_from_sweep, 'config')} from sweep"
            body.append(f"    {sweep_line}\n")
        # Explicit experiments line
        if n_explicit > 0:
            body.append(f"    {_pl(n_explicit, 'explicit experiment')}\n")
        # Total line (only when configs come from both sources)
        if has_both:
            body.append(f"    {_pl(n_configs, 'unique config')} total\n")

    body.append("\n")
    # Hash (dimmed)
    body.append("Study design hash:\n ", style="dim")
    body.append(f"  {hash_display}\n", style="dim")
    # Results path (bold cyan)
    if study_dir is not None:
        body.append("\n")
        body.append("Study results path:\n", style="bold cyan")
        body.append(f"  {study_dir}/\n", style="bold cyan")

    return Panel(
        body,
        title=f"[bold cyan]Study: {study_config.study_name or 'unnamed'}[/]",
        title_align="left",
        padding=(0, 1),
    )


_ENERGY_SAMPLER_NAMES: dict[str, str] = {
    "nvml": "NVMLSampler",
    "zeus": "ZeusSampler",
    "codecarbon": "CodeCarbonSampler",
}


def _resolve_energy_display(
    unique_energy: list[str],
    *,
    probed_sampler: str | None = None,
    skip_probe: bool = False,
) -> str:
    """Build the energy sampler display string, resolving 'auto' when possible.

    When ``skip_probe`` is True (all runners are Docker), the host probe is
    skipped because the container may have different energy samplers available.
    When ``probed_sampler`` is provided it is used to annotate 'auto' entries.
    """
    parts: list[str] = []
    for e in unique_energy:
        if e == "auto":
            if skip_probe or probed_sampler is None:
                parts.append("auto")
            else:
                parts.append(f"{probed_sampler} (auto)")
        elif e in _ENERGY_SAMPLER_NAMES:
            parts.append(_ENERGY_SAMPLER_NAMES[e])
        else:
            parts.append(e)
    return ", ".join(parts)
