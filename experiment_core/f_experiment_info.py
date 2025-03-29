import sys
import os
from datetime import datetime
from transformers import AutoConfig

def get_cores_info(codecarbon_data):
    """
    Extracts GPU, CPU, and OS information from CodeCarbon's tracker data.
    """
    return {
        "gpu_model": getattr(codecarbon_data, "gpu_model", "Unknown"),
        "gpu_count": getattr(codecarbon_data, "gpu_count", "Unknown"),
        "cpu_model": getattr(codecarbon_data, "cpu_model", "Unknown"),
        "cpu_count": getattr(codecarbon_data, "cpu_count", "Unknown"),
        "os": getattr(codecarbon_data, "os", "Unknown"),
    }

def get_region_info(codecarbon_data):
    """
    Extracts region information (country and region) from CodeCarbon's tracker data.
    """
    return {
        "country_name": getattr(codecarbon_data, "country_name", "Unknown"),
        "region": getattr(codecarbon_data, "region", "Unknown"),
    }

def get_experiment_setup(experiment_config, model, codecarbon_data, experiment_id):
    """
    Gathers experiment setup information.
    
    Parameters:
      - experiment_config: The experiment configuration object.
      - model: The model instance (or its identifier).
      - codecarbon_data: CodeCarbon's tracker data.
      
    Returns:
      A dictionary containing setup information.
    """
    print(f"[DEBUG] Enter get_experiment_setup: Experiment ID: {experiment_id}")

    cores_info = get_cores_info(codecarbon_data)
    region_info = get_region_info(codecarbon_data)
    
    # Load the model configuration using the model name from the experiment config.
    model_config = AutoConfig.from_pretrained(experiment_config.model_name)
    is_encoder_decoder = getattr(model_config, "is_encoder_decoder", experiment_config.is_encoder_decoder)
    
    setup_info = {
        "experiment_id": experiment_id,
        "date_time": datetime.now().strftime("%B %d, %Y at %I:%M:%S %p"),
        "model": experiment_config.model_name,
        "is_encoder_decoder": is_encoder_decoder,
        "task_type": experiment_config.task_type,
        "available_gpu_count": cores_info["gpu_count"],
        "gpu_model": cores_info["gpu_model"],
        "available_cpu_count": cores_info["cpu_count"],
        "cpu_model": cores_info["cpu_model"],
        "os": cores_info["os"],
        "python_version": sys.version,
        "country": region_info["country_name"],
        "region": region_info["region"],
    }
    
    print(f"[DEBUG] Exiting get_experiment_setup with result: {setup_info}")

    return setup_info

def get_experimental_variables(experiment_config, model, accelerator):
    """
    Collects experimental variables and accelerator configuration details.
    
    Parameters:
      - experiment_config: The experiment configuration object.
      - model: The model instance.
      - accelerator: The accelerator instance.
      
    Returns:
      A dictionary of experimental variables.
    """
    print(f"[DEBUG] Enter get_experimental_variables: Accelerator index: {accelerator.local_process_index}")

    effective_fp_precision = str(next(model.parameters()).dtype)
    
    experimental_variables = {
        "max_input_tokens": experiment_config.max_input_tokens,
        "max_output_tokens": experiment_config.max_output_tokens,
        "number_input_prompts": getattr(experiment_config, "num_input_prompts", None),
        "decode_token_to_text": getattr(experiment_config, "decode_token_to_text", None),
        "decoder_temperature": experiment_config.decoder_temperature,
        "query_rate": experiment_config.query_rate,
        "fp_precision": effective_fp_precision,
        "quantisation": experiment_config.quantization_config,
        "batching_options": experiment_config.batching_options,
        "sharding_config": experiment_config.sharding_config,
        "accelerate_config": {
            "distributed_type": str(accelerator.distributed_type),
            "num_processes": accelerator.num_processes,
        },
        "inference_type": experiment_config.inference_type,
        "backend": experiment_config.backend,
    }

    print(f"[DEBUG] Exiting get_experimental_variables with result: {experimental_variables}")

    return experimental_variables

def get_model_architecture(model):
    """
    Extracts key architectural features from the model.
    
    Parameters:
      - model: The model instance.
      
    Returns:
      A dictionary containing architecture details.
    """
    print(f"[DEBUG] Enter get_model_architecture: Process ID: {os.getpid()}")

    model_features = {}
    total_params = sum(p.numel() for p in model.parameters())
    model_features["total_params"] = total_params

    if hasattr(model, "config"):
        config = model.config
        model_features["num_hidden_layers"] = getattr(config, "num_hidden_layers", getattr(config, "num_layers", None))
        model_features["hidden_size"] = getattr(config, "hidden_size", None)
        model_features["num_attention_heads"] = getattr(config, "num_attention_heads", None)
        model_features["intermediate_size"] = getattr(config, "intermediate_size", None)
    else:
        model_features["architecture"] = "Unknown (no config attribute)"
        
    print(f"[DEBUG] Exiting get_model_architecture with result: {model_features}")

    return model_features

