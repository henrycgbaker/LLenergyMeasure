"""Firing tests for the re-homed single-field bound rules.

The re-homed bounds (item 6b of the pin-advance) live in the shipped rules.yaml
rather than as generated Field constraints. Each rule must resolve at
``Rule.try_match`` against the generated nested config shape
(``<engine>.engine_params.<field>`` / ``<engine>.sampling_params.<field>``); a
path that does not resolve silently never fires. These tests construct a
violating config through the public ``ExperimentConfig`` path and assert the
rule rejects it, covering every engine and both sub-sections so the path shape
is proven for the whole re-homed set.
"""

from __future__ import annotations

import pytest

from llenergymeasure.config.models import ExperimentConfig

# (engine, section, field, violating_value, rule_id_fragment)
_CASES = [
    # vLLM - cited sampling ranges
    ("vllm", "sampling_params", "presence_penalty", -3.0, "presence_penalty"),
    ("vllm", "sampling_params", "presence_penalty", 3.0, "presence_penalty"),
    ("vllm", "sampling_params", "frequency_penalty", -3.0, "frequency_penalty"),
    ("vllm", "sampling_params", "top_p", 1.5, "top_p"),
    ("vllm", "sampling_params", "top_p", 0.0, "top_p"),
    ("vllm", "sampling_params", "min_p", -0.1, "min_p"),
    # vLLM - engine positivity floors
    ("vllm", "engine_params", "tensor_parallel_size", 0, "tensor_parallel_size"),
    ("vllm", "engine_params", "pipeline_parallel_size", 0, "pipeline_parallel_size"),
    ("vllm", "engine_params", "kv_cache_memory_bytes", 0, "kv_cache_memory_bytes"),
    # tensorrt - engine floors
    ("tensorrt", "engine_params", "max_seq_len", 0, "max_seq_len"),
    ("tensorrt", "engine_params", "tensor_parallel_size", 0, "tensor_parallel_size"),
    ("tensorrt", "engine_params", "max_num_tokens", 0, "max_num_tokens"),
    # tensorrt - sampling floors/ranges
    ("tensorrt", "sampling_params", "temperature", -1.0, "temperature"),
    ("tensorrt", "sampling_params", "top_p", 1.5, "top_p"),
    ("tensorrt", "sampling_params", "n", 0, "n"),
    # transformers - engine floors
    ("transformers", "engine_params", "num_beams", 0, "num_beams"),
    ("transformers", "engine_params", "no_repeat_ngram_size", -1, "no_repeat_ngram_size"),
    ("transformers", "engine_params", "prompt_lookup_num_tokens", 0, "prompt_lookup_num_tokens"),
    # transformers - sampling floors/ranges
    ("transformers", "sampling_params", "min_new_tokens", 0, "min_new_tokens"),
    ("transformers", "sampling_params", "top_p", 2.0, "top_p"),
    ("transformers", "sampling_params", "temperature", -0.5, "temperature"),
]


@pytest.mark.parametrize("engine,section,field,value,fragment", _CASES)
def test_rehomed_rule_rejects_violating_config(engine, section, field, value, fragment):
    """A violating value on the canonical nested path is rejected with the rule id."""
    section_payload = {section: {field: value}}
    with pytest.raises(ValueError, match=fragment):
        ExperimentConfig(task={"model": "gpt2"}, engine=engine, **{engine: section_payload})


def test_in_range_value_is_accepted():
    """A value inside every bound constructs cleanly (rules do not over-fire)."""
    cfg = ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        vllm={
            "engine_params": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            "sampling_params": {"presence_penalty": 0.5, "top_p": 0.9, "min_p": 0.0},
        },
    )
    assert cfg.active_sampling_params().top_p == 0.9
