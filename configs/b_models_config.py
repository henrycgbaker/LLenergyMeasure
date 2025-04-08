import copy
from configs.a_default_config import base_config

huggingface_models = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]

models_config_list = []

for model in huggingface_models:
    cfg = copy.deepcopy(base_config)
    cfg["model_name"] = model
    cfg["config_name"] = f"model_variation_{model}"
    cfg["suite"] = "models"
    models_config_list.append(cfg)

__all__ = ["models_configs"]

if __name__ == "__main__":
    for cfg in models_config_list:
        print(cfg["config_name"])
