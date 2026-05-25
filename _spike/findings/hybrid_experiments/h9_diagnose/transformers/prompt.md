You are a structural-gap analyst for an AST-based static-analysis
catalogue. The output (a) below is what a deterministic AST walker
emitted from the engine source. You are looking for STRUCTURAL gap
PATTERNS that explain why certain validations the source contains do
NOT appear in (a). Each gap pattern is a categorical reason - e.g. the
walker doesn't descend into orelse branches, or doesn't track local-
variable aliases.

For each gap pattern you can identify, emit ONE entry in the JSON
output. Cite a SPECIFIC example field name from this engine that
exhibits the gap. Be honest if you can't find an example - emit no
entries rather than fabricate.

Gap categories you should consider:
- branch-descent: walker handles if/raise but not if/elif/else, not
  else-raise, not nested branches.
- nested-config: walker doesn't recurse into Pydantic / dataclass
  fields typed as another config class.
- local-var-alias: walker compares self.X but source has x = self.X
  and compares the local x.
- normalisation-only: source uses normalisation pattern (no raise);
  walker only emits from if/raise patterns; these validators dormant
  for the walker.
- type-blindness: walker synthesises probe values without consulting
  the declared type; runtime fails for int-typed fields probed with
  string.
- defensive-import: walker fails at import time on bumped versions
  because some symbol was renamed in a dep.
- other: any structural gap that doesn't fit the above.

OUTPUT FORMAT: a single JSON object (no markdown, no commentary, no
code fences, just raw JSON). Use these EXACT keys:

{
  "gaps": [
    {
      "category": "branch-descent | nested-config | local-var-alias | normalisation-only | type-blindness | defensive-import | other",
      "severity": "blocks-correctness | reduces-recall | minor",
      "example_field": "<field name in transformers that exhibits the gap>",
      "structural_reason": "<1-2 sentences; use plain prose without colons or backticks>",
      "cross_engine_pattern": "yes-already-known | yes-new-here | engine-specific | unknown",
      "fix_estimate_loc": <integer>,
      "mergeable_into_spike_refactor": "yes | no | needs-broader-design"
    }
  ]
}

IMPORTANT: in structural_reason, avoid colons and backticks (the parser
is strict). Use plain English. If you must reference code, describe it
in words (e.g. "the if/raise pattern" not "`if X: raise`").

If you find no gaps, output: {"gaps": []}

=== ENGINE: transformers vv4_57_3 ===

=== (a) OUTPUT (41 entries) ===
- id: transformers_beam_search_num_beams_eq_1
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.num_beams": 1}
  message: `num_beams` is set to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation m
- id: transformers_cache_choice_cache_implementation_not_in_allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.cache_implementation": {"present": true, "not_in": ["dynamic", "static"]}}
  message: Invalid `cache_implementation` (nonsense). Choose one of: ('static', 'offloaded_static', 'sliding_window', 'hybrid', 'hy
- id: transformers_cache_choice_use_cache_eq_false
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.use_cache": false}
  message: You have set `use_cache` to `False`, but cache_implementation is set to static. cache_implementation will have no effect
- id: transformers_compile_config_type_compile_config_exceeds_zero
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.compile_config": {">": 0}}
  message: You provided `compile_config` as an instance of <class 'int'>, but it must be an instance of `CompileConfig`.
- id: transformers_compile_config_type_compile_config_type_not_in_CompileConfig
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.compile_config": {"present": true, "type_is_not": ["CompileConfig"]}}
  message: You provided `compile_config` as an instance of <class 'str'>, but it must be an instance of `CompileConfig`.
- id: transformers_dormant_early_stopping_set_true
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.num_beams": 1, "transformers.sampling.early_stopping": {"present": true}}
  message: 
- id: transformers_dormant_epsilon_cutoff_ne_0_0
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.epsilon_cutoff": {"!=": 0.0, "present": true}}
  message: 
- id: transformers_dormant_eta_cutoff_ne_0_0
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.eta_cutoff": {"!=": 0.0, "present": true}}
  message: 
- id: transformers_early_stopping_type_early_stopping_exceeds_zero
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.early_stopping": {">": 0}}
  message: `early_stopping` must be a boolean or 'never', but is 1.5.
- id: transformers_early_stopping_type_early_stopping_not_in_allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.early_stopping": {"present": true, "not_in": [false, true, "never"]}}
  message: `early_stopping` must be a boolean or 'never', but is sometimes.
- id: transformers_early_stopping_type_early_stopping_type_not_in_bool_or_int_or_str
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.early_stopping": {"present": true, "type_is_not": ["bool", "int", "str"]}}
  message: `early_stopping` must be a boolean or 'never', but is 1.5.
- id: transformers_greedy_strips_epsilon_cutoff
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.epsilon_cutoff": {"present": true, "not_equal": 0.0}}
  message: `do_sample` is set to `False`. However, `epsilon_cutoff` is set to `{declared_value}` -- this flag is only used in sampl
- id: transformers_greedy_strips_eta_cutoff
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.eta_cutoff": {"present": true, "not_equal": 0.0}}
  message: `do_sample` is set to `False`. However, `eta_cutoff` is set to `{declared_value}` -- this flag is only used in sample-ba
- id: transformers_greedy_strips_min_p
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.min_p": {"present": true}}
  message: `do_sample` is set to `False`. However, `min_p` is set to `{declared_value}` -- this flag is only used in sample-based g
- id: transformers_greedy_strips_temperature
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.temperature": {"present": true, "not_equal": 1.0}}
  message: `do_sample` is set to `False`. However, `temperature` is set to `{declared_value}` -- this flag is only used in sample-b
- id: transformers_greedy_strips_top_k
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.top_k": {"present": true, "not_equal": 50}}
  message: `do_sample` is set to `False`. However, `top_k` is set to `{declared_value}` -- this flag is only used in sample-based g
- id: transformers_greedy_strips_top_p
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.top_p": {"present": true, "not_equal": 1.0}}
  message: `do_sample` is set to `False`. However, `top_p` is set to `{declared_value}` -- this flag is only used in sample-based g
- id: transformers_greedy_strips_typical_p
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.do_sample": false, "transformers.sampling.typical_p": {"present": true, "not_equal": 1.0}}
  message: `do_sample` is set to `False`. However, `typical_p` is set to `{declared_value}` -- this flag is only used in sample-bas
- id: transformers_negative_pad_token_id
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.pad_token_id": {"<": 0}}
  message: `pad_token_id` should be positive but got -1. This will cause errors when batch generating, if there is padding. Please 
- id: transformers_no_return_dict_strips_output_attentions
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.return_dict_in_generate": false, "transformers.sampling.output_attentions": {"present": true, "not_equal": false}}
  message: `return_dict_in_generate` is NOT set to `True`, but `output_attentions` is. When `return_dict_in_generate` is not `True`
- id: transformers_no_return_dict_strips_output_hidden_states
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.return_dict_in_generate": false, "transformers.sampling.output_hidden_states": {"present": true, "not_equal": false}}
  message: `return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is. When `return_dict_in_generate` is not `Tr
- id: transformers_no_return_dict_strips_output_scores
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.return_dict_in_generate": false, "transformers.sampling.output_scores": {"present": true, "not_equal": false}}
  message: `return_dict_in_generate` is NOT set to `True`, but `output_scores` is. When `return_dict_in_generate` is not `True`, `o
- id: transformers_num_return_vs_beams_do_sample_eq_false_and_num_beams_eq_1
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.num_beams": 1, "transformers.sampling.num_return_sequences": {">": 1}, "transformers.sampling.do_sample": false}
  message: Greedy methods without beam search do not support `num_return_sequences` different than 1 (got 2).
- id: transformers_num_return_vs_beams_num_beams_lt_num_return_sequences
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.num_beams": {"<": "@num_return_sequences"}}
  message: `num_return_sequences` (4) has to be smaller or equal to `num_beams` (2).
- id: transformers_num_return_vs_beams_num_beams_not_divisible_by_num_return_sequences
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.num_beams": {"not_divisible_by": "@num_return_sequences"}}
  message: `num_return_sequences` (4) has to be smaller or equal to `num_beams` (2).
- id: transformers_output_token_ids_max_new_tokens_le_zero
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.max_new_tokens": {"<=": 0}}
  message: `max_new_tokens` must be greater than 0, but is -1.
- id: transformers_output_token_ids_max_new_tokens_not_in_allowlist
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.max_new_tokens": {"present": true, "not_in": [1, 16]}}
  message: `max_new_tokens` must be greater than 0, but is -1.
- id: transformers_output_token_ids_pad_token_id_not_in_allowlist
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.pad_token_id": {"present": true, "not_in": [0, 50256]}}
  message: `pad_token_id` should be positive but got -1. This will cause errors when batch generating, if there is padding. Please 
- id: transformers_raises_bnb_4bit_quant_type_not_type_str
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.bnb_4bit_quant_type": {"type_is_not": "str"}}
  message: bnb_4bit_quant_type must be a string
- id: transformers_raises_bnb_4bit_use_double_quant_not_type_bool
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.bnb_4bit_use_double_quant": {"type_is_not": "bool"}}
  message: bnb_4bit_use_double_quant must be a boolean
- id: transformers_raises_compile_config_not_type_compileconfig
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.compile_config": {"type_is_not": "CompileConfig", "present": true}}
  message: You provided `compile_config` as an instance of {type(self.compile_config)}, but it must be an instance of `CompileConfi
- id: transformers_raises_llm_int8_enable_fp32_cpu_offload_not_type_bool
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.llm_int8_enable_fp32_cpu_offload": {"type_is_not": "bool"}}
  message: llm_int8_enable_fp32_cpu_offload must be a boolean
- id: transformers_raises_llm_int8_has_fp16_weight_not_type_bool
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.llm_int8_has_fp16_weight": {"type_is_not": "bool"}}
  message: llm_int8_has_fp16_weight must be a boolean
- id: transformers_raises_llm_int8_skip_modules_not_type_list
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.llm_int8_skip_modules": {"type_is_not": "list", "present": true}}
  message: llm_int8_skip_modules must be a list of strings
- id: transformers_raises_llm_int8_threshold_not_type_float
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.llm_int8_threshold": {"type_is_not": "float"}}
  message: llm_int8_threshold must be a float
- id: transformers_raises_load_in_4bit_not_type_bool
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.load_in_4bit": {"type_is_not": "bool"}}
  message: load_in_4bit must be a boolean
- id: transformers_raises_load_in_8bit_not_type_bool
  severity: error
  native_type: transformers.BitsAndBytesConfig
  miner_source_method: post_init
  match.fields: {"transformers.load_in_8bit": {"type_is_not": "bool"}}
  message: load_in_8bit must be a boolean
- id: transformers_raises_num_beams_eq_1
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.num_return_sequences": {"!=": 1}, "transformers.sampling.num_beams": 1}
  message: Greedy methods without beam search do not support `num_return_sequences` different than 1 (got {num_return_sequences}).
- id: transformers_single_beam_strips_early_stopping
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.num_beams": 1, "transformers.sampling.early_stopping": {"present": true, "not_equal": false}}
  message: `num_beams` is set to 1. However, `early_stopping` is set to `{declared_value}` -- this flag is only used in beam-based 
- id: transformers_single_beam_strips_length_penalty
  severity: dormant
  native_type: transformers.GenerationConfig
  miner_source_method: validate
  match.fields: {"transformers.sampling.num_beams": 1, "transformers.sampling.length_penalty": {"present": true, "not_equal": 1.0}}
  message: `num_beams` is set to 1. However, `length_penalty` is set to `{declared_value}` -- this flag is only used in beam-based 
- id: transformers_watermarking_type_watermarking_config_type_not_in_WatermarkingConfig
  severity: error
  native_type: transformers.GenerationConfig
  miner_source_method: __init__
  match.fields: {"transformers.sampling.watermarking_config": {"present": true, "type_is_not": ["WatermarkingConfig"]}}
  message: transformers.generation.configuration_utils.WatermarkingConfig() argument after ** must be a mapping, not int

=== ENGINE SOURCE EXCERPTS ===
=== generation/configuration_utils.py ===
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generation configuration class and utilities."""

import copy
import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
    GENERATION_CONFIG_NAME,
    ExplicitEnum,
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    is_remote_url,
    is_torch_available,
    logging,
)


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)
METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")
STATIC_CACHE_IMPLEMENTATIONS = ("static", "offloaded_static")
DYNAMIC_CACHE_IMPLEMENTATIONS = ("dynamic", "dynamic_full", "offloaded", "quantized")
# All the following are redundant and deprecated, but kept for BC
DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS = (
    "sliding_window",
    "hybrid",
    "hybrid_chunked",
    "offloaded_hybrid",
    "offloaded_hybrid_chunked",
)
ALL_STATIC_CACHE_IMPLEMENTATIONS = STATIC_CACHE_IMPLEMENTATIONS + DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS
ALL_CACHE_IMPLEMENTATIONS = ALL_STATIC_CACHE_IMPLEMENTATIONS + DYNAMIC_CACHE_IMPLEMENTATIONS


if is_torch_available():
    from .logits_process import SynthIDTextWatermarkLogitsProcessor, WatermarkLogitsProcessor


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    DOLA_GENERATION = "dola_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationConfig(PushToHubMixin):
    # no-format
    """
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

    <Tip>

    A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
    the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
    description of the possible manipulations, as well as examples of their usage.

    </Tip>

    Arg:
        > Parameters that control the length of the output

        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*, defaults to 0):
            The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
            `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
        min_new_tokens (`int`, *optional*):
            The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        max_time (`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still finish
            the current pass after allocated time has been passed.
        stop_strings (`str or list[str]`, *optional*):
            A string or a list of strings that should terminate generation if the model outputs them.

        > Parameters that control the generation strategy used

        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.

        > Parameters that control the cache

        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        cache_implementation (`str`, *optional*, default to `None`):
            Name of the cache class that will be instantiated in `generate`, for faster decoding. Possible values are:

            - `"dynamic"`: [`DynamicCache`]
            - `"static"`: [`StaticCache`]
            - `"offloaded"`: [`DynamicCache(offloaded=True)`]
            - `"offloaded_static"`: [`StaticCache(offloaded=True)`]
            - `"quantized"`: [`QuantizedCache`]

            If none is specified, we will use the default cache for the model (which is often [`DynamicCache`]). See
            our [cache documentation](https://huggingface.co/docs/transformers/en/kv_cache) for further information.
        cache_config (`dict`, *optional*, default to `None`):
            Arguments used in the key-value cache class can be passed in `cache_config`.
        return_legacy_cache (`bool`, *optional*, default to `True`):
            Whether to return the legacy or new format of the cache when `DynamicCache` is used by default.

        > Parameters for manipulation of the model output logits

        temperature (`float`, *optional*, defaults to 1.0):
            The value used to module the next token probabilities. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 50.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation. This value is set in a model's `generation_config.json` file. If it isn't set, the default value is 1.0
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        typical_p (`float`, *optional*, defaults to 1.0):
            Local typicality measures how similar the conditional probability of predicting a target token next is to
            the expected conditional probability of predicting a random token next, given the partial text already
            generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
            add up to `typical_p` or higher are kept for generation. See [this
            paper](https://huggingface.co/papers/2202.00666) for more details.
        epsilon_cutoff (`float`, *optional*, defaults to 0.0):
            If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
            `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
            size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
        eta_cutoff (`float`, *optional*, defaults to 0.0):
            Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
            0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) *
            exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
            probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
            depending on the size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://huggingface.co/papers/2210.15191) for more details.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://huggingface.co/papers/1909.05858) for more details.
        encoder_repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for encoder_repetition_penalty. An exponential penalty on sequences that are not in the
            original input. 1.0 means no penalty.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids (`list[list[int]]`, *optional*):
            List of list of token ids that are not allowed to be generated. Check
            [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
        renormalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors (including the custom
            ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
            are normalized but some logit processors break the normalization.
        forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`int` or list[int]`, *optional*, defaults to `model.config.forced_eos_token_id`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
        remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
            penalty starts and `decay_factor` represents the factor of exponential decay
        suppress_tokens (`list[int]`, *optional*):
            A list of tokens that will be suppressed at generation. The `SuppressTokens` logit processor will set their
            log probs to `-inf` so that they are not sampled.
        begin_suppress_tokens  (`list[int]`, *optional*):
            A list of tokens that will be suppressed at the beginning of the generation. The `SuppressBeginTokens` logit
            processor will set their log probs to `-inf` so that they are not sampled.
        sequence_bias (`dict[tuple[int], float]`, *optional*)):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. Check
            [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.
        token_healing (`bool`, *optional*, defaults to `False`):
            Heal tail tokens of prompts by replacing them with their appropriate extensions.
            This enhances the quality of completions for prompts affected by greedy tokenization bias.
        guidance_scale (`float`, *optional*):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        watermarking_config (`BaseWatermarkingConfig` or `dict`, *optional*):
            Arguments used to watermark the model outputs by adding a small bias to randomly selected set of "green"
            tokens. See the docs of [`SynthIDTextWatermarkingConfig`] and [`WatermarkingConfig`] for more
            details. If passed as `Dict`, it will be converted to a `WatermarkingConfig` internally.

        > Parameters that define the output variables of generate

        num_return_sequences (`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*):
            Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for
            more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`], as opposed to returning exclusively the generated
            sequence. This flag must be set to `True` to return the generation cache (when `use_cache` is `True`)
            or optional outputs (see flags starting with `output_`)

        > Special tokens that can be used at generation time

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, list[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        > Generation parameters exclusive to encoder-decoder models

        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        decoder_start_token_id (`int` or `list[int]`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length
            `batch_size`. Indicating a list enables different start ids for each element in the batch
            (e.g. multilingual models with different target languages in one batch)

        > Generation parameters exclusive to assistant generation
        is_assistant (`bool`, *optional*, defaults to `False`):
            Whether the model is an assistant (draft) model.
        num_assistant_tokens (`int`, *optional*, defaults to 20):
            Defines the number of _speculative tokens_ that shall be generated by the assistant model before being
            checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation
            more _speculative_ : If the assistant model is performant larger speed-ups can be reached, if the assistant
            model requires lots of corrections, lower speed-ups are reached.
        num_assistant_tokens_schedule (`str`, *optional*, defaults to `"constant"`):
            Defines the schedule at which max assistant tokens shall be changed during inference.
            - `"heuristic"`: When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else
              reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
            - `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
            - `"constant"`: `num_assistant_tokens` stays unchanged during generation
        assistant_confidence_threshold (`float`, *optional*, defaults to 0.4):
            The confidence threshold for the assistant model. If the assistant model's confidence in its prediction for the current token is lower
            than this threshold, the assistant model stops the current token generation iteration, even if the number of _speculative tokens_
            (defined by `num_assistant_tokens`) is not yet reached. The assistant's confidence threshold is adjusted throughout the speculative iterations to reduce the number of unnecessary draft and target forward passes, biased towards avoiding false negatives.
            `assistant_confidence_threshold` value is persistent over multiple generation calls with the same assistant model.
            It is an unsupervised version of the dynamic speculation lookahead
            from Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models <https://huggingface.co/papers/2405.04304>.
        prompt_lookup_num_tokens (`int`, *optional*):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.
        assistant_early_exit(`int`, *optional*):
            If set to a positive integer, early exit of the model will be used as an assistant. Can only be used with
            models that support early exit (i.e. models where logits from intermediate layers can be interpreted by the LM head).
        assistant_lookbehind(`int`, *optional*, defaults to 10):
            If set to a positive integer, the re-encodeing process will additionally consider the last `assistant_lookbehind` assistant tokens
            to correctly align tokens. Can only be used with different tokenizers in speculative decoding.
            See this [blog](https://huggingface.co/blog/universal_assisted_generation) for more details.
        target_lookbehind(`int`, *optional*, defaults to 10):
            If set to a positive integer, the re-encodeing process will additionally consider the last `target_lookbehind` target tokens
            to correctly align tokens. Can only be used with different tokenizers in speculative decoding.
            See this [blog](https://huggingface.co/blog/universal_assisted_generation) for more details.

        > Parameters related to performances and compilation

        compile_config (CompileConfig, *optional*):
            If using a compilable cache, this controls how `generate` will `compile` the forward pass for faster
            inference.
        disable_compile (`bool`, *optional*):
            Whether to disable the automatic compilation of the forward pass. Automatic compilation happens when
            specific criteria are met, including using a compilable cache. Please open an issue if you find the
            need to use this flag.
    """

    extra_output_flags = ("output_attentions", "output_hidden_states", "output_scores", "output_logits")

    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", True)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)

        self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)
        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)

        watermarking_config = kwargs.pop("watermarking_config", None)
        if watermarking_config is None:
            self.watermarking_config = None
        elif isinstance(watermarking_config, BaseWatermarkingConfig):
            self.watermarking_config = watermarking_config
        else:
            self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.is_assistant = False
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "constant")
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", 0.4)
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        ## assistant generation for different tokenizers, the windows size for assistant/target model
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", 10)
        self.target_lookbehind = kwargs.pop("target_lookbehind", 10)

        # Performance
        self.compile_config = kwargs.pop("compile_config", None)
        self.disable_compile = kwargs.pop("disable_compile", False)

        # Deprecated (moved to the Hub). TODO joao, manuel: remove in v4.62.0
        self.low_memory = kwargs.pop("low_memory", None)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.constraints = kwargs.pop("constraints", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate()

    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        return self_without_metadata == other_without_metadata

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
        # property and part of the `__repr__`
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams == 1:
            if self.do_sample is False:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.
... [TRUNCATED 46547 chars]

=== utils/quantization_config.py ===
#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import dataclasses
import importlib.metadata
import json
import os
from dataclasses import dataclass, is_dataclass
from enum import Enum
from inspect import Parameter, signature
from typing import Any, Optional, Union

from packaging import version

from ..utils import (
    is_auto_awq_available,
    is_compressed_tensors_available,
    is_gptqmodel_available,
    is_hqq_available,
    is_quark_available,
    is_torch_available,
    is_torchao_available,
    logging,
)
from .import_utils import is_auto_gptq_available


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    VPTQ = "vptq"
    QUANTO = "quanto"
    EETQ = "eetq"
    HIGGS = "higgs"
    HQQ = "hqq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FBGEMM_FP8 = "fbgemm_fp8"
    TORCHAO = "torchao"
    BITNET = "bitnet"
    SPQR = "spqr"
    FP8 = "fp8"
    QUARK = "quark"
    FPQUANT = "fp_quant"
    AUTOROUND = "auto-round"
    MXFP4 = "mxfp4"


class AWQLinearVersion(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    EXLLAMA = "exllama"
    IPEX = "ipex"

    @staticmethod
    def from_str(version: str):
        version = version.lower()
        if version == "gemm":
            return AWQLinearVersion.GEMM
        elif version == "gemv":
            return AWQLinearVersion.GEMV
        elif version == "exllama":
            return AWQLinearVersion.EXLLAMA
        elif version == "ipex":
            return AWQLinearVersion.IPEX
        else:
            raise ValueError(f"Unknown AWQLinearVersion {version}")


class AwqBackendPackingMethod(str, Enum):
    AUTOAWQ = "autoawq"
    LLMAWQ = "llm-awq"


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    quant_method: QuantizationMethod

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class AutoRoundConfig(QuantizationConfigMixin):
    """This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded AutoRound quantization.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        group_size (`int`, *optional*, defaults to 128): Group-size value
        sym (`bool`, *optional*, defaults to `True`): Symmetric quantization or not
        backend (`str`, *optional*, defaults to `"auto"`): The kernel to use, e.g., ipex,marlin, exllamav2, triton, etc. Ref. https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        backend: str = "auto",
        **kwargs,
    ):
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.backend = backend
        self.packing_format = "auto_round:gptq"
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.quant_method = QuantizationMethod.AUTOROUND
        self.post_init()

    def post_init(self):
        r"""Safety checker that arguments are correct."""
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")

    def get_loading_attributes(self):
        loading_attributes_dict = {"backend": self.backend}
        return loading_attributes_dict

    def to_dict(self):
        config_dict = super().to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        quant_method = config_dict["quant_method"]
        if "auto-round" not in quant_method and "gptq" not in quant_method and "awq" not in quant_method:
            raise NotImplementedError(
                "Failed to convert to auto_round format. Only `gptqv1`, `awq`, and `auto-round` formats are supported."
            )

        if "gptq" in quant_method and "meta" in config_dict:
            raise NotImplementedError("Failed to convert gptq format to auto_round format. Only supports `gptqv1`")

        if "awq" in quant_method and config_dict.get("version", "gemm") != "gemm":
            raise NotImplementedError(
                "Failed to convert awq format to auto_round format. Only supports awq format with gemm version"
            )

        if "auto-round" not in quant_method:
            config_dict["packing_format"] = f"auto_round:{quant_method}"

        return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)


@dataclass
class HqqConfig(QuantizationConfigMixin):
    """
    This is wrapper around hqq's BaseQuantizeConfig.

    Args:
        nbits (`int`, *optional*, defaults to 4):
            Number of bits. Supported values are (8, 4, 3, 2, 1).
        group_size (`int`, *optional*, defaults to 64):
            Group-size value. Supported values are any value that is divisible by weight.shape[axis]).
        view_as_float (`bool`, *optional*, defaults to `False`):
            View the quantized weight as float (used in distributed training) if set to `True`.
        axis (`Optional[int]`, *optional*):
            Axis along which grouping is performed. Supported values are 0 or 1.
        dynamic_config (dict, *optional*):
            Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config.
            If set, each layer specified by its id will use its dedicated quantization configuration.
        skip_modules (`list[str]`, *optional*, defaults to `['lm_head']`):
            List of `nn.Linear` layers to skip.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = 64,
        view_as_float: bool = False,
        axis: Optional[int] = None,
        dynamic_config: Optional[dict] = None,
        skip_modules: list[str] = ["lm_head"],
        **kwargs,
    ):
        if is_hqq_available():
            from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig
        else:
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )

        for deprecated_key in ["quant_zero", "quant_scale", "offload_meta"]:
            if deprecated_key in kwargs:
                logger.info(
                    deprecated_key + " is deprecated. This parameter will be ignored in quantization settings."
                )

        if axis is None:
            axis = 1
            logger.info("Setting axis=1 as faster backends such as TorchAO or BitBlas are only compatible with it.")

        if axis not in [0, 1]:
            raise ValueError("Invalid axis value. Only 0 and 1 are allowed.")

        if dynamic_config is not None:
            self.quant_config = {}
            for key in dynamic_config:
                self.quant_config[key] = HQQBaseQuantizeConfig(**dynamic_config[key])
        else:
            self.quant_config = HQQBaseQuantizeConfig(
                **{
                    "nbits": nbits,
                    "group_size": group_size,
                    "view_as_float": view_as_float,
                    "axis": axis,
                }
            )

        self.quant_method = QuantizationMethod.HQQ
        self.skip_modules = skip_modules

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        pass

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        """
        Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py
        """
        instance = cls()
        instance.quant_config = config["quant_config"]
        instance.skip_modules = config["skip_modules"]
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return {
            "quant_config": self.quant_config,
            "quant_method": self.quant_method,
            "skip_modules": self.skip_modules,
        }

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = HqqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://huggingface.co/papers/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`list[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`):
            This sets the storage type to pack the quantized 4-bit params.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        if kwargs:
            logger.info(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_8bit = value

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

        if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class ExllamaVersion(int, Enum):
    ONE = 1
    TWO = 2


@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[list[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symmetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        checkpoint_format (`str`, *optional*, defaults to `"gptq"`):
            GPTQ weight format. `gptq`(v1) is supported by both gptqmodel and auto-gptq. `gptq_v2` is gptqmodel only.
        meta (`dict[str, any]`, *optional*):
            Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
            i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]
        backend (`str`, *optional*):
            Controls which gptq kernel to be used. Valid values for gptqmodel are `auto`, `auto_trainable` and more. For auto-gptq, only
            valid value is None and `auto_trainable`. Ref gptqmodel backends: https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py
        use_cuda_fp16 (`bool`, *optional*, defaults to `False`):
            Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16. Auto-gptq only.
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`list[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        use_exllama (`bool`, *optional*):
            Whether to use exllama backend. Defaults to `Tr
... [TRUNCATED 65861 chars]


=== KNOWN GAPS FROM PRIOR REVIEW (post_trial_a_gap_closure.md) ===
G-trf-1: defensive imports - the transformers producer's imports crash
on bumped versions (v4_55_4 / v5_9_0). Defensive try/except + AST
fallback would let the walker emit zero invariants gracefully instead
of crashing.


You may RE-CONFIRM known gaps (mark cross_engine_pattern as
yes-already-known) AND surface NEW ones (mark yes-new-here). Be
explicit which is which. Now emit the JSON. Start with `{`.
