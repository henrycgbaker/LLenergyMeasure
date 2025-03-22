from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class ExperimentConfig:
    model_name: str
    is_encoder_decoder: Literal["decoder_only", "encoder_decoder"] = "decoder_only"
    task_type: Literal["text_generation", "translation", "summarisation"] = "text_generation"
    inference_type: Literal["pure_generative", "reasoning"] = "pure_generative"
    max_input_tokens: int = 512
    max_output_tokens: int = 128
    num_input_prompts: int = 1 
    gpu_list: List[int] = field(default_factory=lambda: [0])
    num_processes: int = 1  
    batching_options: dict = field(default_factory=dict)
    sharding_config: dict = field(default_factory=dict)
    query_rate: float = 1.0
    decoder_temperature: float = 1.0
    fp_precision: str = "float32"  # "float8" / "float16"
    backend: str = "pytorch"  # "tensorRT" / "deepserve" / "vllm"

def load_experiment_config(config_path: str = "experiment_configs.yaml") -> ExperimentConfig:
    """
    Loads experiment configuration settings from a YAML file.
    """
    import yaml
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    return ExperimentConfig(**config_dict)
