import os
import torch
import psutil
import subprocess
from fvcore.nn import FlopCountAnalysis
import os
import json
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import uuid
from datetime import datetime
import psutil
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import yaml
from helper_functions.getter_utils import get_compute_performance_metrics  


def get_experiment_setup(model_name, 
                             codecarbon_data, 
                             task_type, 
                             is_encoder_decoder):
    setup = {
        "model": model_name,
        "is_encoder_decoder": is_encoder_decoder,
        "task_type": task_type,
        "gpu_count": codecarbon_data.gpu_count,
        "gpu_model": codecarbon_data.gpu_model,
        "cpu_count": codecarbon_data.cpu_count,
        "cpu_model": codecarbon_data.cpu_model,
        "os": codecarbon_data.os,
        "python_version": sys.version,
        "country": codecarbon_data.country_name,
        "region": codecarbon_data.region,
        "date": datetime.now().strftime("%B %d, %Y at %I:%M:%S %p"),

    }
    return setup

def get_experimental_variables(model, accelerator, config, inference_metrics):
    used_gpu = str(accelerator.device)
    first_param = next(model.parameters())
    effective_fp_precision = str(first_param.dtype)
    effective_batch_size = config.batching_options.get("max_batch_size")
    sharding_config = None
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
        if isinstance(model, FullyShardedDataParallel):
            sharding_config = {
                "reshard_after_forward": str(getattr(model, "reshard_after_forward", None)),
                "cpu_offload": str(getattr(model, "cpu_offload", None)),
                "backward_prefetch": str(getattr(model, "backward_prefetch", None)),
            }
    except ImportError:
        pass

    return {
         "max_input_tokens": config.max_input_tokens,
         "max_output_tokens": config.max_output_tokens,
         "number_runs": inference_metrics["num_runs"],
         "total_token_inputted": inference_metrics["total_input_tokens"],
         "total_tokens_outputted": inference_metrics["total_generated_tokens"],
         "effective_batch_size": effective_batch_size,
         "used_gpu": used_gpu,
         "decoder_temperature": config.decoder_temperature,
         "query_rate": config.query_rate,
         "fp_precision": effective_fp_precision,
         "quantisation": config.quantisation,
         "batching_options": config.batching_options,
         "sharding_config": sharding_config,
         "accelerate_config": {
              "distributed_type": str(accelerator.distributed_type),
              "num_processes": accelerator.num_processes,
              "local_process_index": accelerator.local_process_index
         },
        "inference_type": config.inference_type,
        "backend": config.backend,
    }




def get_experiment_results(metrics, codecarbon_data, model=None, tokenizer=None, device=None):
    energy_kwh = codecarbon_data.energy_consumed
    energy_joules = energy_kwh * 3.6e6  # Convert kWh to Joules
    tokens_per_joule = (metrics["total_generated_tokens"] / energy_joules) if energy_joules > 0 else 0

    # Placeholder for task-specific performance
    task_specific_performance = {}

    # Import the metrics function from metrics module
    compute_metrics = get_compute_performance_metrics(model=model, tokenizer=tokenizer, device=device)

    return {
        "inference_performance": {
            "total_inference_time_sec": metrics["total_time"],
            "average_latency_ms_per_batch": metrics["avg_latency_ms"],
            "throughput_queries_per_sec": metrics["throughput_qps"],
            "throughput_tokens_per_sec": metrics["tokens_per_sec"],
        },
        "energy_performance": {
            "cpu_power": codecarbon_data.cpu_power,
            "gpu_power": codecarbon_data.gpu_power,
            "ram_power": codecarbon_data.ram_power,
            "cpu_energy": codecarbon_data.cpu_energy,
            "gpu_energy": codecarbon_data.gpu_energy,
            "ram_energy": codecarbon_data.ram_energy,
            "total_energy_consumed_kwh": energy_kwh,
            "total_energy_consumed_joules": energy_joules,
            "energy_efficiency_tokens_per_joule": tokens_per_joule,
            "final_emissions": codecarbon_data.emissions
        },
        "compute_performance": compute_metrics,
        "task-specific_performance": task_specific_performance
    }
