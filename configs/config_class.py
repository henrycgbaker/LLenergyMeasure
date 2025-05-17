from dataclasses import dataclass, field, asdict, fields
from typing import List, Literal, Any, Optional, Dict

@dataclass
class ExperimentConfig:
    config_name: str
    model_name: str
    is_encoder_decoder: bool = False
    task_type: Literal["text_generation", "translation", "summarisation"] = "text_generation"
    inference_type: Literal["pure_generative", "reasoning"] = "pure_generative"
    max_input_tokens: int = 512
    max_output_tokens: int = 128
    min_output_tokens: int = 0
    num_input_prompts: int = 1 
    save_outputs: bool = False
    decode_token_to_text: bool = False
    gpu_list: List[int] = field(default_factory=lambda: [0])
    num_processes: int = 1  
    batching_options: dict = field(default_factory=dict)
    sharding_config: dict = field(default_factory=dict)
    query_rate: float = 1.0
    latency_simulation: Optional[Dict[str, Any]] = field(default_factory=dict)
    decoder_config: dict = field(default_factory=dict) 
    fp_precision: Literal["float32", "float16", "float8"] = "float32"
    quantization_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    backend: Literal["pytorch", "tensorRT", "deepserve", "vllm"] = "pytorch"  
    cycle_id: Optional[int] = None

    extra_metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __init__(self, **kwargs):
        # Get all expected field names (excluding extra_metadata)
        expected_fields = {f.name for f in fields(self) if f.name != "extra_metadata"}
        # For each expected field, set it from kwargs.
        for field_name in expected_fields:
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
            else:
                raise ValueError(f"Missing required configuration field: {field_name}")
        # Whatever is left in kwargs is considered extra metadata.
        self.extra_metadata = kwargs
        
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)


def load_experiment_config(config_path: str = "experiment_configs.yaml") -> ExperimentConfig:
    """
    Loads experiment configuration settings from a YAML file.
    """
    import yaml
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    return ExperimentConfig(**config_dict)


import copy