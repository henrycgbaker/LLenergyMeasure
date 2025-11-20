"""
Clean, robust results management system.

Replaces v1.0.0's complex multi-file aggregation with a simple,
type-safe approach using dataclasses and clear structure.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Inference performance metrics."""

    total_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    num_prompts: int
    tokens_per_second: float
    queries_per_second: float
    avg_latency_per_query: float
    avg_tokens_per_prompt: float


@dataclass
class ComputeMetrics:
    """Computational metrics (FLOPs, memory, utilization)."""

    flops: int
    gpu_memory_allocated_mb: float
    gpu_memory_peak_mb: float
    gpu_utilization_percent: List[float] = field(default_factory=list)
    cpu_usage_percent: float = 0.0
    cpu_memory_mb: float = 0.0


@dataclass
class EnergyMetrics:
    """Energy consumption and emissions metrics."""

    duration_seconds: float
    total_energy_kwh: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    ram_energy_kwh: float
    emissions_kg_co2: float
    cpu_power_w: float = 0.0
    gpu_power_w: float = 0.0
    ram_power_w: float = 0.0


@dataclass
class ModelInfo:
    """Model architecture information."""

    model_name: str
    total_parameters: int
    trainable_parameters: int
    precision: str
    quantization: Optional[str] = None
    model_size_mb: float = 0.0


@dataclass
class ExperimentResults:
    """
    Complete experiment results in a single, clean structure.

    This replaces v1.0.0's scattered JSON files with one cohesive dataclass.
    """

    # Identification
    experiment_id: str
    timestamp: str
    config_name: Optional[str] = None

    # Model info
    model: Optional[ModelInfo] = None

    # Metrics
    inference: Optional[InferenceMetrics] = None
    compute: Optional[ComputeMetrics] = None
    energy: Optional[EnergyMetrics] = None

    # Configuration (stored as dict for flexibility)
    config: Dict[str, Any] = field(default_factory=dict)

    # Optional: Generated outputs
    outputs: Optional[List[str]] = None

    # Derived efficiency metrics
    efficiency: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived efficiency metrics after initialization."""
        self._calculate_efficiency_metrics()

    def _calculate_efficiency_metrics(self) -> None:
        """Calculate efficiency metrics from raw data."""
        if not self.inference or not self.energy or not self.compute:
            return

        # Tokens per joule (higher is better)
        total_energy_j = self.energy.total_energy_kwh * 3600 * 1000  # kWh to J
        if total_energy_j > 0:
            self.efficiency["tokens_per_joule"] = self.inference.total_tokens / total_energy_j

            # FLOPs per joule (computational efficiency)
            if self.compute.flops > 0:
                self.efficiency["flops_per_joule"] = self.compute.flops / total_energy_j

        # Tokens per second per watt (throughput efficiency)
        avg_power_w = (
            self.energy.cpu_power_w + self.energy.gpu_power_w + self.energy.ram_power_w
        )
        if avg_power_w > 0:
            self.efficiency["tokens_per_second_per_watt"] = (
                self.inference.tokens_per_second / avg_power_w
            )

        # Energy per query
        if self.inference.num_prompts > 0:
            self.efficiency["kwh_per_query"] = (
                self.energy.total_energy_kwh / self.inference.num_prompts
            )
            self.efficiency["co2_per_query_g"] = (
                self.energy.emissions_kg_co2 * 1000 / self.inference.num_prompts
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentResults":
        """Create from dictionary."""
        # Handle nested dataclasses
        if "model" in data and data["model"]:
            data["model"] = ModelInfo(**data["model"])
        if "inference" in data and data["inference"]:
            data["inference"] = InferenceMetrics(**data["inference"])
        if "compute" in data and data["compute"]:
            data["compute"] = ComputeMetrics(**data["compute"])
        if "energy" in data and data["energy"]:
            data["energy"] = EnergyMetrics(**data["energy"])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentResults":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ResultsManager:
    """
    Manages experiment results storage and retrieval.

    Much simpler than v1.0.0's scattered file approach:
    - Single JSON file per experiment
    - Clean directory structure
    - Easy aggregation
    - Type-safe access
    """

    def __init__(self, results_dir: Path = Path("results")):
        """
        Initialize results manager.

        Args:
            results_dir: Base directory for results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_dir = self.results_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)

        logger.info(f"Results manager initialized: {self.results_dir}")

    def save_experiment(self, results: ExperimentResults) -> Path:
        """
        Save experiment results.

        Args:
            results: ExperimentResults instance

        Returns:
            Path to saved file
        """
        output_file = self.experiments_dir / f"{results.experiment_id}.json"

        logger.info(f"Saving experiment {results.experiment_id} to {output_file}")

        with open(output_file, "w") as f:
            f.write(results.to_json())

        logger.info("Experiment saved successfully")
        return output_file

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResults]:
        """
        Load experiment results by ID.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentResults instance or None if not found
        """
        input_file = self.experiments_dir / f"{experiment_id}.json"

        if not input_file.exists():
            logger.warning(f"Experiment {experiment_id} not found")
            return None

        logger.debug(f"Loading experiment {experiment_id}")

        with open(input_file) as f:
            return ExperimentResults.from_json(f.read())

    def list_experiments(self) -> List[str]:
        """
        List all experiment IDs.

        Returns:
            List of experiment IDs
        """
        experiments = [
            f.stem for f in self.experiments_dir.glob("*.json")
        ]
        logger.debug(f"Found {len(experiments)} experiments")
        return sorted(experiments)

    def aggregate_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Aggregate multiple experiments for analysis.

        Args:
            experiment_ids: Specific IDs to aggregate (None for all)

        Returns:
            List of experiment dictionaries
        """
        if experiment_ids is None:
            experiment_ids = self.list_experiments()

        logger.info(f"Aggregating {len(experiment_ids)} experiments")

        aggregated = []
        for exp_id in experiment_ids:
            results = self.load_experiment(exp_id)
            if results:
                aggregated.append(results.to_dict())

        return aggregated

    def export_to_csv(self, output_file: Path, experiment_ids: Optional[List[str]] = None) -> None:
        """
        Export experiments to CSV for analysis.

        Args:
            output_file: Output CSV file path
            experiment_ids: Specific IDs to export (None for all)
        """
        import csv

        aggregated = self.aggregate_experiments(experiment_ids)

        if not aggregated:
            logger.warning("No experiments to export")
            return

        logger.info(f"Exporting {len(aggregated)} experiments to {output_file}")

        # Flatten nested dicts for CSV
        flat_data = []
        for exp in aggregated:
            flat = self._flatten_dict(exp)
            flat_data.append(flat)

        # Write CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            if flat_data:
                writer = csv.DictWriter(f, fieldnames=flat_data[0].keys())
                writer.writeheader()
                writer.writerows(flat_data)

        logger.info(f"Exported to {output_file}")

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings
                items.append((new_key, ",".join(map(str, v))))
            else:
                items.append((new_key, v))

        return dict(items)

    def generate_summary(self, experiment_ids: Optional[List[str]] = None) -> Dict:
        """
        Generate summary statistics across experiments.

        Args:
            experiment_ids: Specific IDs to summarize (None for all)

        Returns:
            Dictionary with summary statistics
        """
        aggregated = self.aggregate_experiments(experiment_ids)

        if not aggregated:
            return {}

        # Calculate statistics
        tokens_per_second = [
            exp["inference"]["tokens_per_second"]
            for exp in aggregated
            if exp.get("inference")
        ]

        energy_kwh = [
            exp["energy"]["total_energy_kwh"]
            for exp in aggregated
            if exp.get("energy")
        ]

        summary = {
            "total_experiments": len(aggregated),
            "throughput": {
                "mean_tokens_per_second": sum(tokens_per_second) / len(tokens_per_second)
                if tokens_per_second
                else 0,
                "max_tokens_per_second": max(tokens_per_second) if tokens_per_second else 0,
                "min_tokens_per_second": min(tokens_per_second) if tokens_per_second else 0,
            },
            "energy": {
                "total_kwh": sum(energy_kwh),
                "mean_kwh": sum(energy_kwh) / len(energy_kwh) if energy_kwh else 0,
            },
        }

        return summary


def create_results(
    experiment_id: str,
    config: Dict,
    model_info: Optional[Dict] = None,
    inference_metrics: Optional[Dict] = None,
    compute_metrics: Optional[Dict] = None,
    energy_metrics: Optional[Dict] = None,
    outputs: Optional[List[str]] = None,
) -> ExperimentResults:
    """
    Convenience function to create ExperimentResults.

    Args:
        experiment_id: Experiment identifier
        config: Configuration dictionary
        model_info: Model information dict
        inference_metrics: Inference metrics dict
        compute_metrics: Compute metrics dict
        energy_metrics: Energy metrics dict
        outputs: Generated outputs

    Returns:
        ExperimentResults instance
    """
    return ExperimentResults(
        experiment_id=experiment_id,
        timestamp=datetime.now().isoformat(),
        config_name=config.get("config_name"),
        model=ModelInfo(**model_info) if model_info else None,
        inference=InferenceMetrics(**inference_metrics) if inference_metrics else None,
        compute=ComputeMetrics(**compute_metrics) if compute_metrics else None,
        energy=EnergyMetrics(**energy_metrics) if energy_metrics else None,
        config=config,
        outputs=outputs,
    )
