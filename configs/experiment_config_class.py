from dataclasses import dataclass, field, asdict
from typing import List, Literal, Any, Optional, Dict

@dataclass
class ExperimentConfig:
    model_name: str
    is_encoder_decoder: bool = False
    task_type: Literal["text_generation", "translation", "summarisation"] = "text_generation"
    inference_type: Literal["pure_generative", "reasoning"] = "pure_generative"
    max_input_tokens: int = 512
    max_output_tokens: int = 128
    num_input_prompts: int = 1 
    save_outputs: bool = False
    decode_token_to_text: bool = False
    gpu_list: List[int] = field(default_factory=lambda: [0])
    num_processes: int = 1  
    batching_options: dict = field(default_factory=dict)
    sharding_config: dict = field(default_factory=dict)
    query_rate: float = 1.0
    latency_simulation: Optional[Dict[str, Any]] = field(default_factory=dict)
    decoder_temperature: float = 1.0
    fp_precision: Literal["float32", "float16", "float8"] = "float32"
    quantization_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    backend: Literal["pytorch", "tensorRT", "deepserve", "vllm"] = "pytorch"  

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)

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
