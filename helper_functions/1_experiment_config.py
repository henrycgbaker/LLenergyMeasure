from dataclasses import dataclass, field
from typing import List

@dataclass
class ExperimentConfig:
    model_name: str
    is_encoder_decoder: "NA" # "decoder_only" / "encoder_decoder"
    task_type: "text_generation" # "translation" / "summarisation"
    inference_type: "purely_generative" # / "reasoning"
    max_input_tokens: int = 512
    max_output_tokens: int = 128
    num_runs_inputs: int 
    gpu_list: List[int] = field(default_factory=lambda: [0])
    num_processes: int = 1
    batching_options: dict = field(default_factory=dict)
    sharding_config: dict = field(default_factory=dict)
    query_rate: float = 1.0
    decoder_temperature: float = 1.0
    fp_precision: str = "float32" # "float8" / "float 16" 
    quantisation: bool = False
    backend: str = "pytorch" # "tensorRT" / "deepserve"(?) / "vllm"

def load_experiment_config(config_path: str = "experiment_configs.yaml") -> ExperimentConfig:
    """
    Loads experiment configuration settings from a YAML file.
    """
    import yaml
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    return ExperimentConfig(**config_dict)
