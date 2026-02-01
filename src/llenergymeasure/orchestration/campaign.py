"""Campaign orchestration for multi-config comparison experiments.

Manages the execution of multiple experiment configurations across multiple
cycles with warmup, gaps, and ordering controls.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from llenergymeasure.config.campaign_config import CampaignConfig
from llenergymeasure.config.loader import load_config
from llenergymeasure.config.models import ExperimentConfig

if TYPE_CHECKING:
    from llenergymeasure.domain.experiment import AggregatedResult
    from llenergymeasure.orchestration.manifest import (
        CampaignManifest,
        CampaignManifestEntry,
    )


@dataclass
class CampaignExperiment:
    """A single experiment within a campaign.

    Tracks which config and cycle this experiment belongs to.
    """

    config_path: Path
    config_name: str
    config: ExperimentConfig
    cycle_index: int
    experiment_id: str | None = None
    result: AggregatedResult | None = None
    warmup_completed: bool = False
    started_at: datetime | None = None
    completed_at: datetime | None = None
    manifest_entry: CampaignManifestEntry | None = None

    @property
    def backend(self) -> str:
        """Return the backend for this experiment."""
        return getattr(self.config, "backend", "pytorch") or "pytorch"


@dataclass
class CampaignProgress:
    """Tracks progress through a campaign."""

    total_experiments: int
    completed_experiments: int = 0
    current_cycle: int = 0
    current_config_index: int = 0
    warmup_in_progress: bool = False

    @property
    def progress_fraction(self) -> float:
        """Return progress as fraction 0-1."""
        if self.total_experiments == 0:
            return 0.0
        return self.completed_experiments / self.total_experiments


@dataclass
class CampaignRunner:
    """Orchestrates multi-config campaign execution.

    Handles:
    - Loading and validating all config files
    - Generating execution order (interleaved/shuffled/grouped)
    - Running warmup prompts before each config
    - Managing thermal cooldown gaps
    - Tracking results and statistics

    Usage:
        campaign_config = CampaignConfig(...)
        runner = CampaignRunner(campaign_config)

        for experiment in runner.generate_execution_order():
            # Run warmup
            runner.run_warmup(experiment, prompts, run_fn)

            # Wait for config gap
            runner.wait_config_gap()

            # Run actual experiment
            result = run_experiment(experiment)
            runner.record_result(experiment, result)

            # Wait for cycle gap if needed
            if runner.is_cycle_complete():
                runner.wait_cycle_gap()
    """

    campaign: CampaignConfig
    seed: int | None = None

    # Internal state
    _configs: dict[str, ExperimentConfig] = field(default_factory=dict, init=False)
    _experiments: list[CampaignExperiment] = field(default_factory=list, init=False)
    _progress: CampaignProgress | None = field(default=None, init=False)
    _last_experiment_time: float = field(default=0.0, init=False)
    _last_cycle_complete_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Load all configs on initialization."""
        self._load_configs()

    def _load_configs(self) -> None:
        """Load and validate all experiment configs."""
        for config_path in self.campaign.get_config_paths():
            config = load_config(config_path)
            self._configs[config_path.stem] = config

    @property
    def config_names(self) -> list[str]:
        """Return list of config names in original order."""
        return list(self._configs.keys())

    @property
    def num_configs(self) -> int:
        """Return number of configs in campaign."""
        return len(self._configs)

    @property
    def num_cycles(self) -> int:
        """Return number of cycles."""
        return self.campaign.execution.cycles

    @property
    def total_experiments(self) -> int:
        """Return total number of experiments to run."""
        return self.num_configs * self.num_cycles

    def generate_execution_order(self) -> list[CampaignExperiment]:
        """Generate the ordered list of experiments to run.

        Based on campaign.execution.structure:
        - interleaved: A->B->C, A->B->C, A->B->C (fixed order per cycle)
        - shuffled: Random order within each cycle
        - grouped: Ax3, Bx3, Cx3 (all cycles of one config before next)

        Returns:
            List of CampaignExperiment in execution order.
        """
        structure = self.campaign.execution.structure
        config_paths = self.campaign.get_config_paths()
        experiments: list[CampaignExperiment] = []

        if structure == "grouped":
            # All cycles of one config, then next config
            for config_path in config_paths:
                config_name = config_path.stem
                config = self._configs[config_name]
                for cycle in range(self.num_cycles):
                    experiments.append(
                        CampaignExperiment(
                            config_path=config_path,
                            config_name=config_name,
                            config=config,
                            cycle_index=cycle,
                        )
                    )
        else:
            # Interleaved or shuffled - iterate by cycle
            rng = random.Random(self.seed) if self.seed else random.Random()

            for cycle in range(self.num_cycles):
                cycle_configs = list(config_paths)

                if structure == "shuffled":
                    rng.shuffle(cycle_configs)

                for config_path in cycle_configs:
                    config_name = config_path.stem
                    config = self._configs[config_name]
                    experiments.append(
                        CampaignExperiment(
                            config_path=config_path,
                            config_name=config_name,
                            config=config,
                            cycle_index=cycle,
                        )
                    )

        self._experiments = experiments
        self._progress = CampaignProgress(total_experiments=len(experiments))
        return experiments

    def run_warmup(
        self,
        experiment: CampaignExperiment,
        prompts: list[str],
        inference_fn: Callable[[list[str]], None],
    ) -> None:
        """Run warmup prompts for an experiment.

        Uses dual-criteria: stops when EITHER min prompts OR timeout is reached.

        Args:
            experiment: The experiment to warm up for.
            prompts: Full list of prompts (warmup uses first N).
            inference_fn: Function to run inference on prompts.
        """
        warmup_prompts = self.campaign.execution.warmup_prompts
        warmup_timeout = self.campaign.execution.warmup_timeout_seconds

        if warmup_prompts == 0 and warmup_timeout == 0:
            experiment.warmup_completed = True
            return

        # Take warmup prompts from start of dataset
        warmup_subset = prompts[: min(warmup_prompts, len(prompts))]

        start_time = time.time()

        for prompt in warmup_subset:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= warmup_timeout:
                break

            # Run single prompt
            inference_fn([prompt])

        experiment.warmup_completed = True

    def should_wait_config_gap(self) -> bool:
        """Check if we should wait for config gap.

        Returns True if this isn't the first experiment.
        """
        return self._last_experiment_time > 0

    def wait_config_gap(self, callback: Callable[[float], None] | None = None) -> None:
        """Wait for config gap (thermal recovery between configs).

        Args:
            callback: Optional callback called with remaining seconds.
        """
        gap = self.campaign.execution.config_gap_seconds
        if gap <= 0 or not self.should_wait_config_gap():
            return

        elapsed = time.time() - self._last_experiment_time
        remaining = gap - elapsed

        if remaining > 0:
            if callback:
                callback(remaining)
            time.sleep(remaining)

    def is_cycle_complete(self, experiment_index: int) -> bool:
        """Check if a cycle just completed after this experiment.

        Args:
            experiment_index: Index in execution order.

        Returns:
            True if this experiment completes a cycle.
        """
        # Cycle complete when we've run all configs in this cycle
        return (experiment_index + 1) % self.num_configs == 0

    def wait_cycle_gap(self, callback: Callable[[float], None] | None = None) -> None:
        """Wait for cycle gap (full thermal reset between cycles).

        Args:
            callback: Optional callback called with remaining seconds.
        """
        gap = self.campaign.execution.cycle_gap_seconds
        if gap <= 0:
            return

        if callback:
            callback(gap)
        time.sleep(gap)

        self._last_cycle_complete_time = time.time()

    def record_experiment_start(self, experiment: CampaignExperiment) -> None:
        """Record that an experiment has started."""
        experiment.started_at = datetime.now()

    def record_experiment_complete(
        self,
        experiment: CampaignExperiment,
        experiment_id: str,
        result: AggregatedResult | None = None,
    ) -> None:
        """Record experiment completion with results.

        Args:
            experiment: The completed experiment.
            experiment_id: Assigned experiment ID.
            result: Aggregated result if available.
        """
        experiment.experiment_id = experiment_id
        experiment.result = result
        experiment.completed_at = datetime.now()
        self._last_experiment_time = time.time()

        if self._progress:
            self._progress.completed_experiments += 1

    def get_experiments_by_config(self) -> dict[str, list[CampaignExperiment]]:
        """Group completed experiments by config name.

        Returns:
            Dict mapping config name to list of experiments.
        """
        by_config: dict[str, list[CampaignExperiment]] = {}
        for exp in self._experiments:
            if exp.config_name not in by_config:
                by_config[exp.config_name] = []
            by_config[exp.config_name].append(exp)
        return by_config

    def get_completed_experiment_ids(self, config_name: str) -> list[str]:
        """Get experiment IDs for a specific config.

        Args:
            config_name: Name of the config.

        Returns:
            List of experiment IDs.
        """
        return [
            exp.experiment_id
            for exp in self._experiments
            if exp.config_name == config_name and exp.experiment_id is not None
        ]

    # ------------------------------------------------------------------
    # Phase 2: Grid, manifest, resume, health check, cold start methods
    # ------------------------------------------------------------------

    def generate_execution_order_from_grid(self) -> list[CampaignExperiment]:
        """Generate execution order from grid definition if present.

        If ``campaign.grid`` is set, expands the grid into individual configs
        using ``expand_campaign_grid`` and ``validate_campaign_grid``, then
        converts valid configs into CampaignExperiment objects with cycle
        expansion and ordering applied.

        Falls back to :meth:`generate_execution_order` when no grid is defined.

        Returns:
            Ordered list of CampaignExperiment objects.
        """
        if self.campaign.grid is None:
            return self.generate_execution_order()

        from typing import Any

        from llenergymeasure.orchestration.grid import (
            expand_campaign_grid,
            validate_campaign_grid,
        )

        # Build base_config from campaign-level defaults
        base_config: dict[str, Any] = {}
        if self.campaign.model:
            base_config["model_name"] = self.campaign.model

        # Expand grid into config dicts
        config_dicts = expand_campaign_grid(self.campaign.grid, base_config=base_config)
        result = validate_campaign_grid(config_dicts)

        if not result.valid_configs:
            logger.warning("Grid expansion produced 0 valid configs")
            self._experiments = []
            self._progress = CampaignProgress(total_experiments=0)
            return []

        # Convert valid config dicts into CampaignExperiment objects
        base_experiments: list[CampaignExperiment] = []
        for config_dict in result.valid_configs:
            config = ExperimentConfig(**config_dict)
            config_name = config_dict.get("config_name", config.config_name)
            # Register in _configs so num_configs/is_cycle_complete work
            self._configs[config_name] = config
            base_experiments.append(
                CampaignExperiment(
                    config_path=Path(f"<grid:{config_name}>"),
                    config_name=config_name,
                    config=config,
                    cycle_index=0,  # placeholder, set below
                )
            )

        # Apply cycle expansion and ordering
        experiments = self._apply_ordering(base_experiments)

        self._experiments = experiments
        self._progress = CampaignProgress(total_experiments=len(experiments))
        return experiments

    def _apply_ordering(
        self, base_experiments: list[CampaignExperiment]
    ) -> list[CampaignExperiment]:
        """Apply cycle expansion and ordering to a list of base experiments.

        Args:
            base_experiments: One experiment per config (cycle_index=0).

        Returns:
            Expanded and ordered list with cycle_index set correctly.
        """
        structure = self.campaign.execution.structure
        experiments: list[CampaignExperiment] = []

        if structure == "grouped":
            for exp in base_experiments:
                for cycle in range(self.num_cycles):
                    experiments.append(
                        CampaignExperiment(
                            config_path=exp.config_path,
                            config_name=exp.config_name,
                            config=exp.config,
                            cycle_index=cycle,
                        )
                    )
        else:
            rng = random.Random(self.seed) if self.seed else random.Random()
            for cycle in range(self.num_cycles):
                cycle_exps = list(base_experiments)
                if structure == "shuffled":
                    rng.shuffle(cycle_exps)
                for exp in cycle_exps:
                    experiments.append(
                        CampaignExperiment(
                            config_path=exp.config_path,
                            config_name=exp.config_name,
                            config=exp.config,
                            cycle_index=cycle,
                        )
                    )

        return experiments

    def create_manifest(self, execution_order: list[CampaignExperiment]) -> CampaignManifest:
        """Create a campaign manifest from the execution order.

        Each experiment gets a manifest entry with a config hash derived
        from its serialised config dict.

        Args:
            execution_order: The ordered list of experiments to track.

        Returns:
            CampaignManifest with all entries as ``pending``.
        """
        import hashlib
        import json

        from llenergymeasure.core.distributed import get_persistent_unique_id
        from llenergymeasure.orchestration.manifest import (
            CampaignManifest,
            CampaignManifestEntry,
        )

        entries: list[CampaignManifestEntry] = []
        for exp in execution_order:
            config_dict = exp.config.model_dump(mode="json")
            config_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[
                :12
            ]

            entry = CampaignManifestEntry(
                exp_id=get_persistent_unique_id(),
                config_name=exp.config_name,
                config_path=str(exp.config_path),
                config_hash=config_hash,
                backend=exp.backend,
                container=exp.backend,  # service name == backend name
                cycle_index=exp.cycle_index,
                status="pending",
            )
            entries.append(entry)
            exp.manifest_entry = entry

        campaign_config_hash = hashlib.md5(self.campaign.model_dump_json().encode()).hexdigest()[
            :12
        ]

        now = datetime.now()
        return CampaignManifest(
            campaign_id=self.campaign.campaign_id,
            campaign_name=self.campaign.campaign_name,
            created_at=now,
            updated_at=now,
            config_hash=campaign_config_hash,
            total_experiments=len(entries),
            experiments=entries,
        )

    def apply_resume_filter(
        self,
        manifest: CampaignManifest,
        execution_order: list[CampaignExperiment],
    ) -> list[CampaignExperiment]:
        """Filter execution order to only pending or failed experiments.

        Uses the manifest to determine which experiments still need to run.

        Args:
            manifest: Existing campaign manifest.
            execution_order: Full execution order.

        Returns:
            Filtered list of experiments that need to run.
        """
        remaining_ids = {e.exp_id for e in manifest.get_remaining()}
        filtered: list[CampaignExperiment] = []
        for exp in execution_order:
            if exp.manifest_entry and exp.manifest_entry.exp_id in remaining_ids:
                filtered.append(exp)
        return filtered

    def should_health_check(self, experiment_index: int) -> bool:
        """Check whether a health check should run after this experiment.

        Health checks run:
        - After each cycle completes (always, when health_check enabled)
        - Every N experiments if ``health_check.interval_experiments > 0``

        Args:
            experiment_index: Zero-based index in execution order.

        Returns:
            True if a health check should run.
        """
        hc = self.campaign.health_check
        if not hc.enabled:
            return False

        # After cycle completion
        if self.num_configs > 0 and self.is_cycle_complete(experiment_index):
            return True

        # Interval-based
        if hc.interval_experiments > 0:
            return (experiment_index + 1) % hc.interval_experiments == 0

        return False

    def should_cold_start(self) -> bool:
        """Check if cold start is configured for this campaign."""
        return self.campaign.cold_start.force_cold_start


def format_gap_time(seconds: float) -> str:
    """Format gap time for display.

    Args:
        seconds: Time in seconds.

    Returns:
        Human-readable time string.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


__all__ = [
    "CampaignExperiment",
    "CampaignProgress",
    "CampaignRunner",
    "format_gap_time",
]
