"""Tests for the HarnessConfig residual (llem-orchestration knobs)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.harness import (
    HarnessConfig,
    TensorRTHarness,
    TransformersHarness,
    VLLMHarness,
)
from llenergymeasure.config.models import ExperimentConfig


def test_transformers_harness_carries_the_seven_residual_knobs() -> None:
    h = TransformersHarness(
        batch_size=4,
        torch_compile=True,
        torch_compile_mode="reduce-overhead",
        torch_compile_backend="inductor",
        allow_tf32=True,
        autocast_enabled=True,
        autocast_dtype="bfloat16",
    )
    assert h.batch_size == 4
    assert h.torch_compile is True
    assert h.torch_compile_mode == "reduce-overhead"
    assert h.torch_compile_backend == "inductor"
    assert h.allow_tf32 is True
    assert h.autocast_enabled is True
    assert h.autocast_dtype == "bfloat16"


def test_transformers_harness_defaults_are_all_none() -> None:
    h = TransformersHarness()
    assert h.batch_size is None
    assert h.torch_compile is None
    assert h.autocast_dtype is None


def test_batch_size_must_be_ge_one() -> None:
    with pytest.raises(ValidationError):
        TransformersHarness(batch_size=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"torch_compile_mode": "reduce-overhead"},
        {"torch_compile_backend": "inductor"},
        {"torch_compile_mode": "max-autotune", "torch_compile": False},
    ],
)
def test_v2_compile_knobs_require_torch_compile_true(kwargs: dict) -> None:
    with pytest.raises(ValidationError, match="torch_compile=True"):
        TransformersHarness(**kwargs)


def test_v2_compile_knobs_accepted_when_torch_compile_true() -> None:
    h = TransformersHarness(torch_compile=True, torch_compile_mode="max-autotune")
    assert h.torch_compile_mode == "max-autotune"


def test_autocast_dtype_is_literal_constrained() -> None:
    with pytest.raises(ValidationError):
        TransformersHarness(autocast_dtype="float32")


def test_transformers_harness_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TransformersHarness(unknown_knob=1)


def test_vllm_and_tensorrt_harness_have_no_residual_and_forbid_extra() -> None:
    assert VLLMHarness().model_dump() == {}
    assert TensorRTHarness().model_dump() == {}
    with pytest.raises(ValidationError):
        VLLMHarness(batch_size=4)
    with pytest.raises(ValidationError):
        TensorRTHarness(batch_size=4)


def test_harness_wrapper_forbids_extra_engine_keys() -> None:
    with pytest.raises(ValidationError):
        HarnessConfig(unknown_engine={})


def test_harness_section_round_trips_through_experiment_config() -> None:
    ec = ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        harness={"transformers": {"batch_size": 8, "allow_tf32": True}},
    )
    assert ec.harness is not None
    assert ec.harness.transformers is not None
    assert ec.harness.transformers.batch_size == 8
    assert ec.harness.transformers.allow_tf32 is True
