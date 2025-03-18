import os
import torch
import psutil
import subprocess
import ptflops
import logging
logger = logging.getLogger(__name__)

def get_flops(model, input_ids):
    """
    Computes the FLOPs for a given model and a batch of inputs.
    
    Parameters:
      - model: The neural network model.
      - input_ids: A tensor of shape (batch_size, sequence_length)
      
    Returns:
      The number of FLOPs (as a float) for that batch, or None if computation fails.
    """
    batch_size, sequence_length = input_ids.shape
    flops_single, _ = ptflops.get_model_complexity_info(
        model,
        input_res=(sequence_length,),  # input resolution for a single sample
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    if flops_single is None:
        logger.warning("ptflops returned None for model complexity. Unable to compute FLOPs.")
        return None
    return flops_single * batch_size


def get_memory(device):
    """
    Returns a dictionary with current and peak memory usage on the given CUDA device.
    """
    torch.cuda.reset_peak_memory_stats(device)
    
    return {
        "current_memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "current_memory_reserved_bytes": torch.cuda.memory_reserved(device),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    

def get_gpu_cpu_utilisation(device):
    """
    Retrieves GPU and CPU utilisation statistics.
    
    Returns:
      A dictionary with GPU utilization percentages and CPU usage information.
    """
    utilisation_info = {}
    # GPU utilization using nvidia-smi
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        )
        lines = result.decode("utf-8").strip().splitlines()
        gpu_utils = [float(line.strip()) for line in lines if line.strip()]
        utilisation_info["gpu_utilization_percent"] = gpu_utils
    except Exception as e:
        utilisation_info["gpu_utilization_percent"] = f"Error: {str(e)}"
    
    # CPU utilisation
    try:
        utilisation_info["cpu_usage_percent"] = psutil.cpu_percent(interval=1.0, percpu=False)
        process = psutil.Process(os.getpid())
        utilisation_info["cpu_memory_usage_bytes"] = process.memory_info().rss
    except Exception as e:
        utilisation_info["cpu_usage_percent"] = f"Error: {str(e)}"
    
    return utilisation_info


