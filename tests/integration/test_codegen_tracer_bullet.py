"""Tracer-bullet test for the Phase 2-T codegen pipeline.

Validates the architectural payoff of the engine-knowledge-as-data design:
the generated Pydantic config (from mined schema + curation) accepts
values that the hand-written class rejects with narrow ``Literal`` /
bounds. See `.product/designs/engine-knowledge-as-data.md` § Move 3 +
roadmap § Phase 2 acceptance criterion (tracer-bullet).

The tracer-bullet field used here is ``temperature``:

- Hand-written ``TransformersSamplingConfig.temperature`` carries
  ``Field(ge=0.0, le=2.0)`` - rejects temperature > 2.0.
- Generated ``SamplingParams.temperature`` (from mined
  ``schema.discovered.json`` which records only ``"type": "number",
  "default": 1.0`` with no bounds) accepts arbitrary floats.

The engine itself (HuggingFace ``GenerationConfig``) accepts any
non-negative float for temperature; researchers occasionally want
temperature > 2.0 for exploration. The hand-written narrow bound was an
llem invention; the data-driven class drops the invention in favour of
the schema's actual surface.

The tracer-bullet pattern generalises: any time a hand-written class
narrows beyond what the engine accepts, the generated class will accept
the broader set. Phase 1 Move 1 deepening will expand the mined surface
to cover more such cases (kwargs-docstring walker, BitsAndBytesConfig
LANDMARK, etc.).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.engine_configs import (
    TransformersConfig,
    TransformersSamplingConfig,
)
from llenergymeasure.config.models import ExperimentConfig, TaskConfig
from llenergymeasure.engines.transformers import (
    Config,
    EngineParams,
    SamplingParams,
)


class TestPublicAPISurface:
    """Design's naming change must hold: import paths match the spec."""

    def test_engines_module_re_exports_generated_classes(self) -> None:
        # The design's naming-change table promises:
        #   from llenergymeasure.engines.transformers import Config,
        #   EngineParams, SamplingParams
        # All three classes must be importable from the engine subpackage
        # (not just the internal .config submodule).
        assert Config.__name__ == "Config"
        assert EngineParams.__name__ == "EngineParams"
        assert SamplingParams.__name__ == "SamplingParams"

    def test_generated_classes_are_pydantic_models(self) -> None:
        # extra='allow' contract per design § Data shape.
        assert Config.model_config.get("extra") == "allow"
        assert SamplingParams.model_config.get("extra") == "allow"
        assert EngineParams.model_config.get("extra") == "allow"


class TestTracerBulletTemperature:
    """The Phase 2 acceptance criterion: a value previously rejected by
    hand-curated bounds now validates through the generated class.
    """

    def test_handwritten_rejects_temperature_above_two(self) -> None:
        # Hand-written carries Field(ge=0.0, le=2.0) - rejects > 2.0.
        with pytest.raises(ValidationError) as exc_info:
            TransformersSamplingConfig(temperature=3.0)
        # The error names the field and the bound.
        assert "temperature" in str(exc_info.value)

    def test_generated_accepts_temperature_above_two(self) -> None:
        # Generated class has no upper bound (mined schema only records
        # type=number, default=1.0). Accepts arbitrary positive floats.
        sp = SamplingParams(temperature=3.0)
        assert sp.temperature == 3.0

    def test_generated_accepts_zero_temperature(self) -> None:
        # Greedy decoding edge case; both classes should accept this one.
        sp = SamplingParams(temperature=0.0)
        assert sp.temperature == 0.0

    def test_generated_accepts_temperature_far_above_handwritten_bound(self) -> None:
        # Researchers occasionally use very high temperatures for
        # exploration / sampling-diversity studies. The schema-driven
        # class doesn't impose llem's narrow bound.
        sp = SamplingParams(temperature=10.0)
        assert sp.temperature == 10.0


class TestTracerBulletDtypeHalf:
    """The design's canonical tracer-bullet (`.product/designs/
    engine-knowledge-as-data.md` § Phase 2 verification): a config
    value previously rejected by hand-curated ``Literal`` (e.g.
    ``dtype: "half"``) now validates through the generated class.

    transformers accepts ``dtype="half"`` (alias for float16); the
    hand-written ``TransformersConfig.dtype`` is
    ``Literal["float32", "float16", "bfloat16"] | None`` which
    rejects it. The Move 1 kwargs-docstring walker lifts
    ``dtype: str`` from the from_pretrained docstring (which simply
    documents it as ``str``); the generated ``EngineParams.dtype``
    is therefore ``str | None`` and accepts the engine's actual
    valid set.
    """

    def test_handwritten_rejects_half(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TransformersConfig(dtype="half")
        assert "dtype" in str(exc_info.value)

    def test_generated_accepts_half(self) -> None:
        ep = EngineParams(dtype="half")
        assert ep.dtype == "half"

    def test_generated_accepts_any_string_engine_actually_takes(self) -> None:
        # Engine accepts torch.dtype objects + several string aliases:
        # 'float32', 'float16', 'bfloat16', 'half', 'auto'. The generated
        # class doesn't enforce a narrow set; it lets the engine validate.
        for v in ("float32", "float16", "bfloat16", "half", "auto"):
            ep = EngineParams(dtype=v)
            assert ep.dtype == v


class TestTracerBulletAttnImplementation:
    """Same pattern: hand-written attn_implementation is
    ``Literal["sdpa", "flash_attention_2", "flash_attention_3", "eager"] | None``
    - narrow. Mined ``str | None`` accepts whatever transformers ships
    next (e.g. a new ``"sage_attention"`` backend) without llem-side
    code change.
    """

    def test_handwritten_rejects_novel_backend(self) -> None:
        with pytest.raises(ValidationError):
            TransformersConfig(attn_implementation="hypothetical_new_backend_v2")

    def test_generated_accepts_novel_backend(self) -> None:
        ep = EngineParams(attn_implementation="hypothetical_new_backend_v2")
        assert ep.attn_implementation == "hypothetical_new_backend_v2"


class TestTracerBulletBitsAndBytesFields:
    """BitsAndBytesConfig fields land in the generated class via the
    Move 1 BNB LANDMARK + companion-config docstring walk. Five fields:
    load_in_4bit, load_in_8bit, bnb_4bit_compute_dtype,
    bnb_4bit_quant_type, bnb_4bit_use_double_quant.
    """

    def test_load_in_4bit_lifted_to_generated_class(self) -> None:
        ep = EngineParams(load_in_4bit=True)
        assert ep.load_in_4bit is True

    def test_bnb_4bit_quant_type_lifted_and_typed(self) -> None:
        # bnb_4bit_quant_type docstring: (`str`, *optional*, defaults to `fp4`)
        ep = EngineParams(bnb_4bit_quant_type="nf4")
        assert ep.bnb_4bit_quant_type == "nf4"
        # default surfaces too
        ep_default = EngineParams()
        assert ep_default.bnb_4bit_quant_type == "fp4"

    def test_bnb_4bit_compute_dtype_untyped(self) -> None:
        # docstring: (`torch.dtype` or str, ...) - walker leaves type
        # unset (torch.dtype isn't a JSON-Schema primitive); generated
        # class types as Any | None.
        ep = EngineParams(bnb_4bit_compute_dtype="bfloat16")
        assert ep.bnb_4bit_compute_dtype == "bfloat16"


class TestProvenancePreservation:
    """The x-source / x-source-ref provenance keys (Move 1 walker
    populates them) survive end-to-end through datamodel-codegen as
    ``json_schema_extra`` on the generated ``Field``."""

    def test_dtype_carries_kwargs_docstring_provenance(self) -> None:
        # datamodel-codegen renders --field-extra-keys-mapped keys
        # as Field(json_schema_extra={...}). Pydantic exposes that on
        # model_fields[name].json_schema_extra.
        field_info = EngineParams.model_fields["dtype"]
        extra = field_info.json_schema_extra
        assert isinstance(extra, dict)
        assert extra.get("x-source") == "kwargs_docstring"
        assert (
            extra.get("x-source-ref")
            == "transformers.PreTrainedModel.from_pretrained.__doc__"
        )

    def test_load_in_4bit_carries_bnb_provenance(self) -> None:
        field_info = EngineParams.model_fields["load_in_4bit"]
        extra = field_info.json_schema_extra
        assert isinstance(extra, dict)
        assert extra.get("x-source") == "kwargs_docstring"
        assert (
            extra.get("x-source-ref")
            == "transformers.BitsAndBytesConfig.__doc__"
        )


class TestExperimentConfigWireUp:
    """Phase 2-T pilot wire-up: the architectural payoff reaches the
    ExperimentConfig boundary via the spike's ``engine_v2`` field. The
    full type swap (replacing ``transformers: TransformersConfig`` with
    ``transformers: engines.transformers.Config``) is deferred pending
    a real loader migration; the wire-up here demonstrates the
    mechanism works without disturbing 125 references across the
    codebase.
    """

    def test_engine_v2_accepts_nested_shape(self) -> None:
        cfg = ExperimentConfig(
            task=TaskConfig(model="gpt2"),
            engine="transformers",
            engine_v2={
                "engine_params": {"dtype": "half"},
                "sampling_params": {"temperature": 0.7},
            },
        )
        typed = cfg.as_v2_config()
        # as_v2_config returns the parsed engines.transformers.Config
        assert type(typed).__name__ == "Config"
        assert typed.engine_params.dtype == "half"
        assert typed.sampling_params.temperature == 0.7

    def test_tracer_bullet_through_experiment_config(self) -> None:
        # The design's canonical demo, threaded through the
        # ExperimentConfig boundary.

        # Hand-written `transformers:` rejects dtype="half"
        with pytest.raises(ValidationError):
            ExperimentConfig(
                task=TaskConfig(model="gpt2"),
                engine="transformers",
                transformers={"dtype": "half"},  # narrow Literal rejects
            )

        # Generated `engine_v2:` accepts the same value end-to-end
        cfg = ExperimentConfig(
            task=TaskConfig(model="gpt2"),
            engine="transformers",
            engine_v2={"engine_params": {"dtype": "half"}},
        )
        assert cfg.as_v2_config().engine_params.dtype == "half"

    def test_tracer_bullet_temperature_through_experiment_config(self) -> None:
        # temperature=3.0 also flows through (hand-written rejects > 2.0).
        with pytest.raises(ValidationError):
            ExperimentConfig(
                task=TaskConfig(model="gpt2"),
                engine="transformers",
                transformers={"sampling": {"temperature": 3.0}},
            )
        cfg = ExperimentConfig(
            task=TaskConfig(model="gpt2"),
            engine="transformers",
            engine_v2={"sampling_params": {"temperature": 3.0}},
        )
        assert cfg.as_v2_config().sampling_params.temperature == 3.0

    def test_engine_v2_validation_errors_surface_at_construction(self) -> None:
        # Bad engine_v2 shape - missing required nested structure -
        # surfaces as ValidationError at ExperimentConfig construction
        # time, not lazily at as_v2_config() call time.
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(
                task=TaskConfig(model="gpt2"),
                engine="transformers",
                # engine_params must be an object, not a list
                engine_v2={"engine_params": ["not", "a", "dict"]},
            )
        assert "engine_v2" in str(exc_info.value).lower() or "EngineParams" in str(
            exc_info.value
        )

    def test_engine_v2_none_returns_none(self) -> None:
        # No engine_v2 set - as_v2_config returns None.
        cfg = ExperimentConfig(
            task=TaskConfig(model="gpt2"),
            engine="transformers",
        )
        assert cfg.as_v2_config() is None

    def test_engine_v2_works_for_vllm(self) -> None:
        # Wire-up is engine-agnostic; vllm parses through vllm.Config.
        cfg = ExperimentConfig(
            task=TaskConfig(model="gpt2"),
            engine="vllm",
            engine_v2={
                "engine_params": {"tensor_parallel_size": 2, "gpu_memory_utilization": 0.85},
                "sampling_params": {"temperature": 0.5},
            },
        )
        typed = cfg.as_v2_config()
        assert typed.engine_params.tensor_parallel_size == 2
        assert typed.engine_params.gpu_memory_utilization == 0.85


class TestExtraAllowContract:
    """``extra='allow'`` per design - unknown fields are stored, not
    rejected. Critical for forward-compat with engine versions that
    introduce new fields between Renovate bumps.
    """

    def test_unknown_field_is_accepted_and_preserved(self) -> None:
        # Imagine HF transformers 5.0 introduces a new sampling field
        # `temperature_schedule`. The schema for v4.57.3 doesn't know
        # about it, but extra='allow' lets it through.
        sp = SamplingParams(temperature=0.7, novel_future_field="adaptive")
        dumped = sp.model_dump()
        assert dumped["novel_future_field"] == "adaptive"

    def test_handwritten_also_allows_extra_for_back_compat(self) -> None:
        # Confirm hand-written has the same extra='allow' contract -
        # this is a property of the existing class, not new.
        c = TransformersSamplingConfig(temperature=0.7, undocumented_param=1)
        dumped = c.model_dump()
        assert dumped["undocumented_param"] == 1


class TestNestedConfigShape:
    """``Config`` nests ``EngineParams`` + ``SamplingParams`` per design
    § Naming changes (revised 2026-05-24)."""

    def test_config_holds_both_sections_as_typed_subclasses(self) -> None:
        cfg = Config(
            engine_params={},
            sampling_params={"temperature": 0.7, "top_k": 40},
        )
        assert isinstance(cfg.engine_params, EngineParams)
        assert isinstance(cfg.sampling_params, SamplingParams)
        assert cfg.sampling_params.temperature == 0.7
        assert cfg.sampling_params.top_k == 40

    def test_default_values_come_from_mined_schema(self) -> None:
        # mined schema for transformers GenerationConfig records:
        # temperature default 1.0, top_k default 50, num_beams default 1.
        # These come through into the generated class as Pydantic
        # defaults (NOT None, unlike the hand-written class).
        sp = SamplingParams()
        assert sp.temperature == 1.0  # generated default from schema
        assert sp.top_k == 50  # generated default from schema
        assert sp.num_beams == 1  # generated default from schema

    def test_handwritten_uses_none_default(self) -> None:
        # Property of the existing hand-written class; useful contrast.
        # llem's None-as-default convention says "let engine apply its own
        # default at execution time"; the generated class instead exposes
        # the engine's real default.
        c = TransformersSamplingConfig()
        assert c.temperature is None  # hand-written default is None
        assert c.top_k is None  # hand-written default is None
