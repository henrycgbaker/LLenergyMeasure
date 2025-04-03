import io
import contextlib
import os
import torch
import psutil
import subprocess
import ptflops
import logging
import concurrent.futures

logging.getLogger("codecarbon").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import concurrent.futures
from fvcore.nn import FlopCountAnalysis

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_dict):
        # Disable caching to avoid returning a DynamicCache.
        return self.model(**input_dict, use_cache=False)

def get_flops_fvcore_for_sample(model, sample_length, device):
    print(f"[DEBUG] get_flops_fvcore_for_sample: Received model type: {type(model)}")
    try:
        current_device = next(model.parameters()).device
        print(f"[DEBUG] Model parameter device before .to(device): {current_device}")
    except Exception as e:
        print(f"[DEBUG] Could not get model parameter device: {e}")

    # Ensure the model is on the same device as the dummy input.
    model = model.to(device)
    print(f"[DEBUG] Model moved to device: {device}")

    # Create dummy input tensors for one sample.
    dummy_input = torch.zeros((1, sample_length), dtype=torch.long).to(device)
    attention_mask = torch.ones_like(dummy_input)
    input_dict = {"input_ids": dummy_input, "attention_mask": attention_mask}

    # Wrap the model in a nn.Module to allow fvcore to inspect it.
    wrapped_model = ModelWrapper(model)
    print(f"[DEBUG] Wrapped model type: {type(wrapped_model)}")
    
    try:
        flops_analysis = FlopCountAnalysis(wrapped_model, (input_dict,))
        total_flops = flops_analysis.total()
        return total_flops
    except Exception as e:
        print(f"[DEBUG] Exception in get_flops_fvcore_for_sample: {e}")
        raise

def get_flops_fvcore(model, input_ids, timeout_per_sample=10):
    batch_size = input_ids.shape[0]
    sample_lengths = [input_ids[i].shape[0] for i in range(batch_size)]
    
    if len(set(sample_lengths)) == 1:
        print(f"[DEBUG] All samples have length {sample_lengths[0]}. Computing FLOPs for one sample with fvcore.")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_flops_fvcore_for_sample, model, sample_lengths[0], input_ids.device)
                flops_single = future.result(timeout=timeout_per_sample)
            print(f"[DEBUG] Computed FLOPs for one sample (fvcore): {flops_single}")
            return flops_single * batch_size
        except Exception as e:
            print(f"[DEBUG] FLOPs computation with fvcore failed on representative sample: {e}")
            return None
    else:
        total_flops = 0.0
        for i in range(batch_size):
            sample_length = input_ids[i].shape[0]
            print(f"[DEBUG] get_flops_fvcore: Processing sample {i} with sample_length {sample_length}")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(get_flops_fvcore_for_sample, model, sample_length, input_ids.device)
                    flops_single = future.result(timeout=timeout_per_sample)
                if flops_single is None:
                    print(f"[DEBUG] FLOPs computation with fvcore returned None for sample {i}. Skipping.")
                    continue
                print(f"[DEBUG] get_flops_fvcore: Sample {i} computed FLOPs: {flops_single}")
                total_flops += flops_single
            except Exception as e:
                print(f"[DEBUG] FLOPs computation with fvcore failed for sample {i}: {e}")
                continue
        print(f"[DEBUG] get_flops_fvcore: Total FLOPs computed: {total_flops}")
        return total_flops

def combine_comp_metrics_fvcore(model, device, tokenised_input_ids, accelerator, experiment_config):
    """
    Combines compute-related metrics (FLOPs, memory stats, and utilisation)
    using the fvcore-based FLOPs functions.
    """
    print(f"[DEBUG] Enter combine_comp_metrics_fvcore: Accelerator index: {accelerator.local_process_index}")
    
    # Debug: log the type and content of experiment_config
    print(f"[DEBUG] experiment_config type: {type(experiment_config)}")
    # If experiment_config is not a simple dict, you might want to print specific attributes:
    if hasattr(experiment_config, "quantization_config"):
        print(f"[DEBUG] experiment_config.quantization_config: {experiment_config.quantization_config}")
    else:
        print("[DEBUG] experiment_config does not have attribute 'quantization_config'")
    
    flops = 0.0
    if accelerator.is_main_process:
        # If quantization is enabled, use the cached FLOPs value.
        quantised = (hasattr(experiment_config, "quantization_config") and 
                     experiment_config.quantization_config and 
                     experiment_config.quantization_config.get("quantization", False))
        if quantised:
            flops = experiment_config.quantization_config.get("cached_flops_for_quantised_models", 0.0)
            print(f"[DEBUG] Using cached FLOPs for quantized model: {flops}")
        else:
            # Debug: also print the type of model passed to get_flops_fvcore
            print(f"[DEBUG] Model type in combine_comp_metrics_fvcore: {type(model)}")
            flops = get_flops_fvcore(model, tokenised_input_ids)
            if flops is None:
                print("[DEBUG] FLOPs computation with fvcore failed. Falling back to 0.0.")
                flops = 0.0

    memory = get_memory(device)
    utilisation = get_gpu_cpu_utilisation(device)
    
    print(f"[DEBUG] Exiting combine_comp_metrics_fvcore with result: flops = {flops}; memory = {memory}; compute_util = {utilisation}")
    return {
        "flops": flops,
        "memory": memory,
        "compute_utilisation": utilisation,
    }

    
#old ptflops workflow
def get_flops_for_sample(model, sample_length, device):
    def input_constructor(input_res):
        dummy_input = torch.zeros((1,) + input_res, dtype=torch.long).to(device)
        # Add an attention mask of ones matching the input shape.
        attention_mask = torch.ones_like(dummy_input)
        return {"input_ids": dummy_input, "attention_mask": attention_mask}

    with io.StringIO() as buf:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            flops_single, _ = ptflops.get_model_complexity_info(
                model,
                input_res=(sample_length,),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
                input_constructor=input_constructor
            )
    return flops_single


def get_flops(model, input_ids, timeout_per_sample=10):
    """
    Computes total FLOPs for a batch of tokenised input samples.
    If all samples are the same length, compute FLOPs for one sample and multiply.
    Otherwise, fall back to per-sample computation with a timeout.
    """
    batch_size = input_ids.shape[0]
    sample_lengths = [input_ids[i].shape[0] for i in range(batch_size)]
    if len(set(sample_lengths)) == 1:
        # All samples have the same length. Compute once and multiply.
        print(f"[DEBUG] All samples have length {sample_lengths[0]}. Computing FLOPs for one sample.")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_flops_for_sample, model, sample_lengths[0], input_ids.device)
                flops_single = future.result(timeout=timeout_per_sample)
            print(f"[DEBUG] Computed FLOPs for one sample: {flops_single}")
            return flops_single * batch_size
        except Exception as e:
            print(f"[DEBUG] FLOPs computation failed on representative sample: {e}")
            return None
    else:
        # Fallback: compute each sample individually.
        total_flops = 0.0
        for i in range(batch_size):
            sample_length = input_ids[i].shape[0]
            print(f"[DEBUG] get_flops: Processing sample {i} with sample_length {sample_length}")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(get_flops_for_sample, model, sample_length, input_ids.device)
                    flops_single = future.result(timeout=timeout_per_sample)
                if flops_single is None:
                    print(f"[DEBUG] FLOPs computation returned None for sample {i}. Skipping.")
                    continue
                print(f"[DEBUG] get_flops: Sample {i} computed FLOPs: {flops_single}")
                total_flops += flops_single
            except Exception as e:
                print(f"[DEBUG] FLOPs computation failed for sample {i}: {e}")
                continue
        print(f"[DEBUG] get_flops: Total FLOPs computed: {total_flops}")
        return total_flops


def get_memory(device):
    """
    Returns a dictionary with current and peak memory usage on the given CUDA device.
    """
    torch.cuda.reset_peak_memory_stats(device)
    
    return {
        "gpu_current_memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "gpu_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_current_memory_reserved_bytes": torch.cuda.memory_reserved(device),
        "gpu_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
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


def combine_comp_metrics(model, device, tokenised_input_ids, accelerator, experiment_config):
    """
    Combines compute-related metrics: FLOPs, memory stats, and device utilisation.
    Only process 0 computes FLOPs to avoid duplication.
    If the model is quantized, uses a cached FLOPs value since the full-precision FLOPs
    are architecture-dependent.
    """
    print(f"[DEBUG] Enter combine_comp_metrics: Accelerator index: {accelerator.local_process_index}")
    
    flops = 0.0
    if accelerator.is_main_process:
        # Check if quantization is enabled. (TO DO: MOVE THIS TO 'GET_FLOPS')
        quantised = experiment_config.quantization_config and experiment_config.quantization_config.get("quantization", False)
        if quantised:
            # Use cached FLOPs value for quantized models!!!!
            flops = experiment_config.quantization_config.get("cached_flops_for_quantised_models", 0.0)
            print(f"[DEBUG] Using cached FLOPs for quantized model: {flops}")
        else:
            # If unquantized, compute FLOPs normally.
            flops = get_flops(model, tokenised_input_ids)
            if flops is None:
                print("[DEBUG] FLOPs computation failed. Falling back to 0.0.")
                flops = 0.0

    memory = get_memory(device)
    utilisation = get_gpu_cpu_utilisation(device)
    
    print(f"[DEBUG] Exiting combine_comp_metrics with result: flops = {flops}; memory = {memory}; compute_util = {utilisation}")
    return {
        "flops": flops,
        "memory": memory,
        "compute_utilisation": utilisation,
    }
