"""
Pydantic configuration models for type-safe experiment configuration.

This replaces the v1.0 nested dictionary approach with modern Pydantic models
providing validation, IDE support, and better maintainability.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class BatchingConfig(BaseModel):
    """Batching configuration."""

    batch_size: int = Field(16, ge=1, description="Fixed batch size for inference")
    adaptive: bool = Field(False, description="Enable adaptive batching based on token count")
    adaptive_max_tokens: Optional[int] = Field(
        None, ge=1, description="Maximum tokens per batch for adaptive batching"
    )
    max_batch_size: Optional[int] = Field(
        None, ge=1, description="Maximum batch size for adaptive batching"
    )

    @model_validator(mode="after")
    def validate_adaptive_settings(self) -> "BatchingConfig":
        """Ensure adaptive batching has required parameters."""
        if self.adaptive and (self.adaptive_max_tokens is None or self.max_batch_size is None):
            raise ValueError(
                "adaptive_max_tokens and max_batch_size required when adaptive=True"
            )
        return self


class QuantizationConfig(BaseModel):
    """Quantization configuration."""

    enabled: bool = Field(False, description="Enable quantization")
    load_in_4bit: bool = Field(False, description="Load model in 4-bit quantization")
    load_in_8bit: bool = Field(False, description="Load model in 8-bit quantization")
    compute_dtype: Literal["float16", "bfloat16"] = Field(
        "float16", description="Compute dtype for quantized operations"
    )
    quant_type: Optional[Literal["nf4", "fp4", "int8"]] = Field(
        None, description="Quantization type"
    )

    @model_validator(mode="after")
    def validate_quantization(self) -> "QuantizationConfig":
        """Ensure only one quantization method is enabled."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
        if (self.load_in_4bit or self.load_in_8bit) and not self.enabled:
            self.enabled = True
        return self


class DecoderConfig(BaseModel):
    """Decoder/generation configuration."""

    mode: Literal["greedy", "top_k", "top_p"] = Field("greedy", description="Decoding mode")
    temperature: float = Field(1.0, gt=0.0, le=2.0, description="Sampling temperature")
    top_k: Optional[int] = Field(None, ge=1, description="Top-K sampling parameter")
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0, description="Top-P (nucleus) sampling")
    do_sample: bool = Field(False, description="Enable sampling")

    @model_validator(mode="after")
    def validate_decoder_params(self) -> "DecoderConfig":
        """Validate decoder parameters consistency."""
        if self.mode == "top_k" and self.top_k is None:
            raise ValueError("top_k must be specified when mode='top_k'")
        if self.mode == "top_p" and self.top_p is None:
            raise ValueError("top_p must be specified when mode='top_p'")

        # Enable sampling for non-greedy modes
        if self.mode != "greedy":
            self.do_sample = True

        return self


class LatencySimulationConfig(BaseModel):
    """Latency simulation configuration."""

    enabled: bool = Field(False, description="Enable latency simulation")
    delay_min: float = Field(0.0, ge=0.0, description="Minimum delay in seconds")
    delay_max: float = Field(0.0, ge=0.0, description="Maximum delay in seconds")
    simulate_burst: bool = Field(False, description="Simulate bursty traffic")
    burst_interval: float = Field(1.0, gt=0.0, description="Burst interval in seconds")
    burst_size: int = Field(1, ge=1, description="Number of requests per burst")

    @field_validator("delay_max")
    @classmethod
    def validate_delay_range(cls, v: float, info) -> float:
        """Ensure delay_max >= delay_min."""
        if "delay_min" in info.data and v < info.data["delay_min"]:
            raise ValueError("delay_max must be >= delay_min")
        return v


class FSDPConfig(BaseModel):
    """Fully Sharded Data Parallel configuration."""

    use_orig_params: bool = Field(False, description="Use original parameters")
    cpu_offload: bool = Field(False, description="Offload parameters to CPU")


class ShardingConfig(BaseModel):
    """Model sharding configuration."""

    strategy: Literal["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"] = Field(
        "NO_SHARD", description="Sharding strategy"
    )
    fsdp_config: FSDPConfig = Field(default_factory=FSDPConfig, description="FSDP configuration")


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    This replaces the v1.0 nested dictionary configuration with a type-safe
    Pydantic model providing validation and better developer experience.
    """

    # Metadata
    config_name: Optional[str] = Field(None, description="Human-readable config name")
    suite: Optional[str] = Field(None, description="Experiment suite identifier")
    cycle_id: Optional[int] = Field(None, ge=1, description="Experimental cycle number")

    # Model configuration
    model_name: str = Field(..., description="Hugging Face model name or path")
    is_encoder_decoder: bool = Field(False, description="Whether model is encoder-decoder")
    backend: Literal["pytorch", "vllm"] = Field("pytorch", description="Inference backend")

    # Task configuration
    task_type: Literal["text_generation"] = Field(
        "text_generation", description="Task type"
    )
    inference_type: Literal["pure_generative"] = Field(
        "pure_generative", description="Inference type"
    )

    # Hardware
    gpu_list: List[int] = Field(default_factory=lambda: [0], description="List of GPU IDs")
    num_processes: int = Field(1, ge=1, description="Number of distributed processes")

    # Inference parameters
    max_input_tokens: int = Field(128, ge=1, description="Maximum input tokens")
    max_output_tokens: int = Field(128, ge=1, description="Maximum output tokens")
    min_output_tokens: int = Field(128, ge=1, description="Minimum output tokens")
    num_input_prompts: int = Field(128, ge=1, description="Number of input prompts")

    # Precision
    precision: Literal["float32", "float16", "bfloat16", "float8"] = Field(
        "float16", description="Model precision"
    )

    # Sub-configurations
    batching: BatchingConfig = Field(
        default_factory=BatchingConfig, description="Batching configuration"
    )
    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig, description="Quantization configuration"
    )
    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig, description="Decoder configuration"
    )
    latency: LatencySimulationConfig = Field(
        default_factory=LatencySimulationConfig, description="Latency simulation"
    )
    sharding: ShardingConfig = Field(
        default_factory=ShardingConfig, description="Sharding configuration"
    )

    # Output settings
    save_outputs: bool = Field(True, description="Save generated outputs")
    decode_token_to_text: bool = Field(True, description="Decode tokens to text")
    results_dir: str = Field("results", description="Results output directory")

    # Query rate
    query_rate: float = Field(1.0, gt=0.0, description="Queries per second")

    @field_validator("num_processes")
    @classmethod
    def validate_num_processes(cls, v: int, info) -> int:
        """Ensure num_processes <= len(gpu_list)."""
        if "gpu_list" in info.data and v > len(info.data["gpu_list"]):
            raise ValueError("num_processes cannot exceed number of GPUs in gpu_list")
        return v

    @classmethod
    def from_legacy_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """
        Create config from v1.0 legacy dictionary format.

        Args:
            config_dict: Old nested dictionary configuration

        Returns:
            ExperimentConfig instance
        """
        # Map old keys to new structure
        return cls(
            config_name=config_dict.get("config_name"),
            suite=config_dict.get("suite"),
            cycle_id=config_dict.get("cycle_id"),
            model_name=config_dict["model_name"],
            is_encoder_decoder=config_dict.get("is_encoder_decoder", False),
            backend=config_dict.get("backend", "pytorch"),
            task_type=config_dict.get("task_type", "text_generation"),
            inference_type=config_dict.get("inference_type", "pure_generative"),
            gpu_list=config_dict.get("gpu_list", [0]),
            num_processes=config_dict.get("num_processes", 1),
            max_input_tokens=config_dict.get("max_input_tokens", 128),
            max_output_tokens=config_dict.get("max_output_tokens", 128),
            min_output_tokens=config_dict.get("min_output_tokens", 128),
            num_input_prompts=config_dict.get("num_input_prompts", 128),
            precision=config_dict.get("fp_precision", "float16"),
            batching=BatchingConfig(
                batch_size=config_dict.get("batching_options", {}).get(
                    "batch_size___fixed_batching", 16
                ),
                adaptive=config_dict.get("batching_options", {}).get("adaptive_batching", False),
                adaptive_max_tokens=config_dict.get("batching_options", {}).get(
                    "adaptive_max_tokens"
                ),
                max_batch_size=config_dict.get("batching_options", {}).get(
                    "max_batch_size___adaptive_batching"
                ),
            ),
            quantization=QuantizationConfig(
                enabled=config_dict.get("quantization_config", {}).get("quantization", False),
                load_in_4bit=config_dict.get("quantization_config", {}).get("load_in_4bit", False),
                load_in_8bit=config_dict.get("quantization_config", {}).get("load_in_8bit", False),
            ),
            decoder=DecoderConfig(
                mode=config_dict.get("decoder_config", {}).get("decoding_mode", "greedy"),
                temperature=config_dict.get("decoder_config", {}).get("decoder_temperature", 1.0),
                top_k=config_dict.get("decoder_config", {}).get("decoder_top_k"),
                top_p=config_dict.get("decoder_config", {}).get("decoder_top_p"),
            ),
            latency=LatencySimulationConfig(
                enabled=config_dict.get("latency_simulation", {}).get("simulate", False),
                delay_min=config_dict.get("latency_simulation", {}).get("delay_min", 0.0),
                delay_max=config_dict.get("latency_simulation", {}).get("delay_max", 0.0),
                simulate_burst=config_dict.get("latency_simulation", {}).get(
                    "simulate_burst", False
                ),
                burst_interval=config_dict.get("latency_simulation", {}).get("burst_interval", 1.0),
                burst_size=config_dict.get("latency_simulation", {}).get("burst_size", 1),
            ),
            sharding=ShardingConfig(
                strategy=config_dict.get("sharding_config", {}).get("sharding_strategy", "NO_SHARD"),
                fsdp_config=FSDPConfig(
                    use_orig_params=config_dict.get("sharding_config", {})
                    .get("fsdp_config", {})
                    .get("use_orig_params", False),
                    cpu_offload=config_dict.get("sharding_config", {})
                    .get("fsdp_config", {})
                    .get("cpu_offload", False),
                ),
            ),
            save_outputs=config_dict.get("save_outputs", True),
            decode_token_to_text=config_dict.get("decode_token_to_text", True),
            query_rate=config_dict.get("query_rate", 1.0),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump(exclude_none=True)

    class Config:
        """Pydantic config."""

        validate_assignment = True
        extra = "forbid"  # Don't allow extra fields
