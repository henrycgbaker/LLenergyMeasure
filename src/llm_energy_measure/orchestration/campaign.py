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

from llm_energy_measure.config.campaign_config import CampaignConfig
from llm_energy_measure.config.loader import load_config
from llm_energy_measure.config.models import ExperimentConfig

if TYPE_CHECKING:
    from llm_energy_measure.domain.experiment import AggregatedResult


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
