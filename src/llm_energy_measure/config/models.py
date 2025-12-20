"""Configuration models for LLM Bench experiments."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BatchingConfig(BaseModel):
    """Batching configuration for inference."""

    batch_size: int = Field(default=1, ge=1, description="Batch size for inference")
    dynamic_batching: bool = Field(default=False, description="Enable dynamic batching")


class ShardingConfig(BaseModel):
    """Model sharding configuration for multi-GPU."""

    strategy: Literal["none", "tensor_parallel", "pipeline_parallel"] = Field(
        default="none", description="Sharding strategy"
    )
    num_shards: int = Field(default=1, ge=1, description="Number of shards")


class LatencySimulation(BaseModel):
    """Configuration for simulating network latency."""

    enabled: bool = Field(default=False, description="Enable latency simulation")
    delay_min_ms: float = Field(default=0.0, ge=0, description="Minimum delay in ms")
    delay_max_ms: float = Field(default=0.0, ge=0, description="Maximum delay in ms")

    @model_validator(mode="after")
    def validate_delay_range(self) -> "LatencySimulation":
        if self.delay_min_ms > self.delay_max_ms:
            raise ValueError("delay_min_ms must be <= delay_max_ms")
        return self


class DecoderConfig(BaseModel):
    """Decoder/generation configuration."""

    temperature: float = Field(default=1.0, ge=0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0, le=1, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    repetition_penalty: float = Field(default=1.0, ge=0, description="Repetition penalty")


class QuantizationConfig(BaseModel):
    """Quantization configuration for model loading."""

    quantization: bool = Field(default=False, description="Enable quantization")
    load_in_4bit: bool = Field(default=False, description="Load in 4-bit (BNB)")
    load_in_8bit: bool = Field(default=False, description="Load in 8-bit (BNB)")
    bnb_4bit_compute_dtype: str = Field(default="float16", description="Compute dtype for 4-bit")
    bnb_4bit_quant_type: str = Field(default="nf4", description="Quantization type (nf4, fp4)")
    bnb_4bit_use_double_quant: bool = Field(default=False, description="Use double quantization")

    @model_validator(mode="after")
    def validate_quantization_exclusivity(self) -> "QuantizationConfig":
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
        if (self.load_in_4bit or self.load_in_8bit) and not self.quantization:
            # Auto-enable quantization flag
            object.__setattr__(self, "quantization", True)
        return self


class ExperimentConfig(BaseModel):
    """Main experiment configuration.

    This is the central configuration object that controls all aspects
    of an LLM benchmarking experiment.
    """

    # Identity
    config_name: str = Field(..., min_length=1, description="Unique config identifier")
    model_name: str = Field(..., min_length=1, description="HuggingFace model name/path")

    # Model properties
    is_encoder_decoder: bool = Field(default=False, description="Is encoder-decoder model")
    task_type: Literal["text_generation", "translation", "summarisation"] = Field(
        default="text_generation", description="Task type"
    )
    inference_type: Literal["pure_generative", "reasoning"] = Field(
        default="pure_generative", description="Inference type"
    )

    # Token limits
    max_input_tokens: int = Field(default=512, ge=1, description="Max input tokens")
    max_output_tokens: int = Field(default=128, ge=1, description="Max output tokens")
    min_output_tokens: int = Field(default=0, ge=0, description="Min output tokens")

    # Input configuration
    num_input_prompts: int = Field(default=1, ge=1, description="Number of prompts")
    save_outputs: bool = Field(default=False, description="Save generated outputs")
    decode_token_to_text: bool = Field(default=False, description="Decode tokens to text")

    # Distributed configuration
    gpu_list: list[int] = Field(default_factory=lambda: [0], description="GPU indices to use")
    num_processes: int = Field(default=1, ge=1, description="Number of processes")

    # Sub-configurations
    batching_options: BatchingConfig = Field(
        default_factory=BatchingConfig, description="Batching config"
    )
    sharding_config: ShardingConfig = Field(
        default_factory=ShardingConfig, description="Sharding config"
    )
    latency_simulation: LatencySimulation = Field(
        default_factory=LatencySimulation, description="Latency simulation config"
    )
    decoder_config: DecoderConfig = Field(
        default_factory=DecoderConfig, description="Decoder/generation config"
    )
    quantization_config: QuantizationConfig = Field(
        default_factory=QuantizationConfig, description="Quantization config"
    )

    # Precision and backend
    fp_precision: Literal["float32", "float16", "bfloat16"] = Field(
        default="float16", description="Floating point precision"
    )
    backend: Literal["pytorch", "tensorrt", "vllm"] = Field(
        default="pytorch", description="Inference backend"
    )

    # Experiment tracking
    cycle_id: int | None = Field(default=None, description="Experiment cycle ID")
    query_rate: float = Field(default=1.0, ge=0, description="Query rate (queries/sec)")

    # Extra metadata (for extensibility)
    extra_metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_config(self) -> "ExperimentConfig":
        # Validate num_processes <= len(gpu_list)
        if self.num_processes > len(self.gpu_list):
            raise ValueError(
                f"num_processes ({self.num_processes}) must be <= "
                f"len(gpu_list) ({len(self.gpu_list)})"
            )

        # Validate min_output_tokens <= max_output_tokens
        if self.min_output_tokens > self.max_output_tokens:
            raise ValueError(
                f"min_output_tokens ({self.min_output_tokens}) must be <= "
                f"max_output_tokens ({self.max_output_tokens})"
            )

        return self

    @field_validator("gpu_list", mode="before")
    @classmethod
    def ensure_gpu_list(cls, v: Any) -> list[int]:
        if isinstance(v, int):
            return [v]
        return list(v)
