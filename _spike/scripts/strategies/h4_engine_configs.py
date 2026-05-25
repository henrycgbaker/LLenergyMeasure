"""Engine configs for the H4 (LLM-modifies-miner) runner.

Each entry pins: which producer module the LLM patches, which walker
function the patched copy exposes, which source venv path is on sys.path
for the subprocess invocation, and the curated engine-source excerpts
the LLM should see (kept small to fit Ollama's 32k context).

The gap_summary text is paraphrased from
``_spike/findings/post_trial_a_gap_closure.md``. The engine_excerpts are
literal snippets from the source venvs at
``/tmp/trial_<engine>_<version>_venv/src/<engine>/``.

This module is import-only; the runner at
``_spike/scripts/strategies/run_h4_modify_miner.py`` picks the entries
up via ``ENGINE_CONFIGS``.
"""

from __future__ import annotations

from _spike.scripts.strategies.run_h4_modify_miner import EngineConfig


# ---------------------------------------------------------------------------
# vllm v0_7_3 - the most gap-rich engine for H4
# ---------------------------------------------------------------------------

VLLM_GAP_SUMMARY = """\
G-vllm-1: EngineArgs.__post_init__ has ZERO `raise` statements at this
version. All validation is normalisation (`if self.X is None: self.X = default`).
The walker only emits invariants from `if X: raise` patterns, so EngineArgs
contributes zero invariants today. Possible fix: extend the walker to emit
normalisation patterns as `severity=dormant` (different shape than error).

G-vllm-2: ModelConfig._verify_quantization, _verify_tokenizer_mode,
_verify_cuda_graph, _verify_bnb_config use LOCAL VARIABLES in their
predicates. Example: `tokenizer_mode = self.tokenizer_mode.lower(); if
tokenizer_mode not in ["auto", "slow", ...]: raise`. The walker's
`_self_attr` predicate extractor only matches `self.X`, not a local var
that aliases `self.X`. Need light alias tracking: detect
`<local> = self.<field>.<method>()` or `<local> = self.<field>` and
treat the local as a proxy for the field in subsequent predicates within
the same function body.

G-vllm-3: CacheConfig._verify_cache_dtype uses `if X: pass / elif Y: log /
else: raise`. The walker handles the `if X: raise` pattern. For elif/else
chains it descends into nested `ast.If`s in the orelse, but does NOT emit
invariants for `else:` (non-If) bodies that contain a raise. Patch: in the
final `else:` branch (i.e. `orelse` that is NOT a single `If` node),
descend into it with the NEGATION of all preceding `if`/`elif` conditions
accumulated in the predicate frame, then emit on detected raises.
"""

# vllm CacheConfig._verify_cache_dtype + _verify_prefix_caching - the
# elif/else case study (G-vllm-3).
VLLM_EXCERPT_CACHE_DTYPE = """\
class CacheConfig:
    ...
    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            logger.info(
                "Using fp8 data type to store kv cache. ...")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return
        if self.sliding_window is not None:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. ...")
"""

# vllm ModelConfig._verify_tokenizer_mode + _verify_quantization - the
# local-variable-compare case study (G-vllm-2).
VLLM_EXCERPT_MODEL_CONFIG_VERIFY = """\
class ModelConfig:
    ...
    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow", "mistral", "custom"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. ...")
        self.tokenizer_mode = tokenizer_mode

    def _verify_quantization(self) -> None:
        supported_quantization = QUANTIZATION_METHODS
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed_tensors",
            "compressed-tensors", "experts_int8", "quark"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()
        # (HF-config-quant-method tying block omitted)
        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. ...")
            from vllm.platforms import current_platform
            current_platform.verify_quantization(self.quantization)
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully optimized yet. ...",
                    self.quantization)

    def _verify_cuda_graph(self) -> None:
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)
        MODEL_NOT_SUPPORT_CUDA_GRAPH = ['mllama']
        if (self.hf_config.model_type in MODEL_NOT_SUPPORT_CUDA_GRAPH
                and not self.enforce_eager):
            logger.warning(...)
            self.enforce_eager = True
"""

# vllm EngineArgs.__post_init__ - normalisation-only, no raises (G-vllm-1).
VLLM_EXCERPT_ENGINE_ARGS = """\
# vllm/engine/arg_utils.py
@dataclass
class EngineArgs:
    # ... 50+ fields ...
    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model
        # Override the default value of enable_prefix_caching if not set.
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = bool(envs.VLLM_USE_V1)
        # Override max_num_seqs if not set.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256 if not envs.VLLM_USE_V1 else 1024
        # support `EngineArgs(compilation_config={...})` without manual ctor.
        if isinstance(self.compilation_config, (int, dict)):
            self.compilation_config = CompilationConfig.from_cli(
                str(self.compilation_config))
        # Setup plugins.
        from vllm.plugins import load_general_plugins
        load_general_plugins()
"""

VLLM_CONFIG = EngineConfig(
    engine="vllm",
    version_slug="v0_7_3",
    producer_rel="engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py",
    walker_fn="walk_vllm_static",
    venv_source_parent="/tmp/trial_vllm_v0_7_3_venv/src",
    gap_summary=VLLM_GAP_SUMMARY,
    engine_excerpts=[
        ("CacheConfig._verify_cache_dtype + _verify_prefix_caching (G-vllm-3)", VLLM_EXCERPT_CACHE_DTYPE),
        ("ModelConfig._verify_tokenizer_mode / _verify_quantization / _verify_cuda_graph (G-vllm-2)", VLLM_EXCERPT_MODEL_CONFIG_VERIFY),
        ("EngineArgs.__post_init__ (G-vllm-1: zero raises, all normalisation)", VLLM_EXCERPT_ENGINE_ARGS),
    ],
)


# ---------------------------------------------------------------------------
# tensorrt v0_21_0 - probe synth + nested-config + deprecation noise
# ---------------------------------------------------------------------------

TENSORRT_GAP_SUMMARY = """\
G-trt-1 (type-blind probe-value synthesis): `_value_satisfying("present",
True)` returns `"x"` for a "present" predicate regardless of the field's
declared type. For an int-typed field this produces a probe like
`{field: "x"}` which raises TypeError at runtime, masking the invariant
as "infrastructure error" even though the invariant itself is correct.
The walker has no field-type information at predicate-build time. Patch
path: thread declared field type (from the class-body AST) into the
Predicate dataclass; have `_value_satisfying` pick `0`, `True`, `"x"`,
or `[]` based on the declared type.

G-trt-2 (DeprecationWarning poisoning of negative-case capture): NOT a
walker gap - lives in `scripts/_invariant_validation_common.py`'s
emission capture. Out of scope for H4 walker patches.

G-trt-3 (nested-config dispatch): SchedulerConfig, QuantConfig,
KvCacheConfig are Pydantic-validator-bearing nested config classes. The
walker doesn't recurse into them via class-level type references. Patch
path: scan class body for fields annotated as `<NestedConfig> | None` or
similar; add those nested classes to the walk list with appropriate
namespace prefixes. (This is broader than a single anchor patch - the
walker needs a new dispatch table. Score this as
"needs_broader_refactor" if you can't fit a localised patch.)

NOTE on the harness: the bumped-version (a) runner subprocess imports
the walker explicitly via importlib.util.spec_from_file_location, so the
walker module name is `h4_patched_producer` not the canonical
`engine_versions.tensorrt.v0_21_0.producers.static_invariant_miner`. Do
not depend on the module's __name__.
"""

# tensorrt _value_satisfying excerpt - the type-blind site.
TENSORRT_EXCERPT_VALUE_SATISFYING = """\
def _value_satisfying(op: str, rhs: Any) -> Any:
    if op == "present":
        return rhs if rhs is not True else "x"   # <--- G-trt-1: returns
                                                  # `"x"` regardless of the
                                                  # field's declared type
    if op == "absent":
        return None
    ...
    if op == "==":
        return rhs
    ...
"""

# tensorrt nested-config example from source.
TENSORRT_EXCERPT_NESTED_CONFIG = """\
# Inside llmapi/llm_args.py at v0.21.0:
class TrtLlmArgs(BaseLlmArgs):
    # ... many fields ...
    scheduler_config: Optional[SchedulerConfig] = Field(default=None, ...)
    quant_config: Optional[QuantConfig] = Field(default=None, ...)
    kv_cache_config: Optional[KvCacheConfig] = Field(default=None, ...)

class SchedulerConfig(BaseModel):
    capacity_scheduler_policy: CapacitySchedulerPolicy = (
        CapacitySchedulerPolicy.MAX_UTILIZATION
    )
    context_chunking_policy: Optional[ContextChunkingPolicy] = None
    # Has @field_validator on multiple fields that the walker does not see
    # because SchedulerConfig is never added to _CLASS_TARGETS.
    @field_validator("dynamic_batch_config")
    def check_dynamic_batch_config(cls, v): ...

class QuantConfig(BaseModel):
    quant_algo: Optional[QuantAlgo] = None
    kv_cache_quant_algo: Optional[QuantAlgo] = None
    @model_validator(mode="after")
    def validate_field_consistency(self): ...
"""

TENSORRT_CONFIG = EngineConfig(
    engine="tensorrt",
    version_slug="v0_21_0",
    producer_rel="engine_versions/tensorrt/v0_21_0/producers/static_invariant_miner.py",
    walker_fn="walk_tensorrt",
    venv_source_parent="/tmp/trial_tensorrt_v0_21_0_venv/src",
    gap_summary=TENSORRT_GAP_SUMMARY,
    engine_excerpts=[
        ("_value_satisfying (G-trt-1: type-blind probe synth)", TENSORRT_EXCERPT_VALUE_SATISFYING),
        ("Nested-config classes - SchedulerConfig / QuantConfig / KvCacheConfig (G-trt-3)", TENSORRT_EXCERPT_NESTED_CONFIG),
    ],
)


# ---------------------------------------------------------------------------
# transformers v4_57_3 - mature; only known gap is defensive imports
# ---------------------------------------------------------------------------

TRANSFORMERS_GAP_SUMMARY = """\
G-trf-1 (defensive imports for version bumps): The producer's import
block hits hard ImportError on bumped transformers versions (v4_55_4
/ v5_9_0) when symbols renamed in `tokenizers` or `huggingface_hub`.
The crash IS the brittleness signal during the trial (Phase 3a.2
data) - but defensive `try ... except ImportError` would let the
walker emit zero invariants gracefully instead of crashing the
miner subprocess. Patch path: wrap the most-fragile imports in
try/except blocks that fall back to lightweight stubs so the
AST walk can still proceed against the bumped source.

The transformers walker is MATURE (1599 LoC, 41 active invariants).
There are NO known coverage gaps at v4_57_3 - this is the gold-
standard walker. So patches here are mostly defensive-import
hardening rather than coverage extensions. If you can't find a
clean localised import-hardening patch, emit zero patches and one
diagnosis stating "no gap visible at active version".
"""

TRANSFORMERS_EXCERPT_IMPORTS = """\
# Sample of imports in the transformers v4_57_3 producer that may
# fail on bumped versions:
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import logging, is_torch_available
from huggingface_hub import HfApi
# tokenizers has had API renames; the producer's tokenizers imports
# fail at module load on transformers v5_9_0 since `tokenizers` itself
# was bumped in lock-step.
"""

TRANSFORMERS_CONFIG = EngineConfig(
    engine="transformers",
    version_slug="v4_57_3",
    producer_rel="engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py",
    walker_fn="walk_transformers",
    venv_source_parent="/tmp/trial_transformers_v4_57_6_venv/src",
    gap_summary=TRANSFORMERS_GAP_SUMMARY,
    engine_excerpts=[
        ("Transformers imports at risk of bump-breakage (G-trf-1)", TRANSFORMERS_EXCERPT_IMPORTS),
    ],
)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

ENGINE_CONFIGS: dict[str, EngineConfig] = {
    "vllm": VLLM_CONFIG,
    "tensorrt": TENSORRT_CONFIG,
    "transformers": TRANSFORMERS_CONFIG,
}
