"""End-to-end integration test for per-field ``auto`` sweep auto-expansion.

A study whose sweep uses the ``auto`` shorthand flows through the full
config-loader parse plus study-layer finalisation (dedup), and yields the same
resolved experiment count as the equivalent explicit-values study.

Mirrors the style of ``tests/integration/test_sweep_dedup_end_to_end.py``.
Run time: < 1s - no GPU, all Pydantic + engine-invariants work.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.config.models import StudyConfig
from llenergymeasure.study.loading import finalise_study


def _write_study(tmp_path: Path, name: str, raw: dict) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _finalise(path: Path, **kwargs) -> StudyConfig:
    return finalise_study(load_study_config(path, **kwargs))


def test_temperature_auto_flows_through_dedup(tmp_path: Path) -> None:
    """``temperature: auto`` + do_sample sweep dedups exactly like explicit temps.

    temperature ``auto`` -> [0.0, 1.0, 2.0]; crossed with do_sample [True, False]
    gives 6 declared configs. The three do_sample=False (greedy) variants are
    measurement-equivalent and collapse, matching the explicit-values study.
    """
    auto_study = {
        "study_name": "autoexpand_temp",
        "engine": "transformers",
        "task": {"model": "gpt2", "dataset": {"source": "arc", "n_prompts": 10}},
        "sweep": {
            "transformers.sampling_params.temperature": "auto",
            "transformers.sampling_params.do_sample": [True, False],
        },
    }
    auto_config = _finalise(_write_study(tmp_path, "auto", auto_study))

    # 6 declared (3 temps x 2 do_sample), greedy family collapses.
    assert len(auto_config.declared_resolved_config_hashes) == 6
    assert auto_config.dedup_mode == "resolved"
    assert sum(g["member_count"] for g in auto_config.pre_run_equivalence_groups) == 6

    # Equivalent explicit-values study yields the identical resolved experiment set.
    explicit_study = {
        **auto_study,
        "study_name": "explicit_temp",
        "sweep": {
            "transformers.sampling_params.temperature": [0.0, 1.0, 2.0],
            "transformers.sampling_params.do_sample": [True, False],
        },
    }
    explicit_config = _finalise(_write_study(tmp_path, "explicit", explicit_study))
    assert len(auto_config.experiments) == len(explicit_config.experiments)
