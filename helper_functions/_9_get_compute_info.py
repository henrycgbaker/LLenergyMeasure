import io
import contextlib
import os
import torch
import psutil
import subprocess
import ptflops
import logging

logging.getLogger("codecarbon").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def get_flops(model, input_ids_batch):
    """
    Computes total FLOPs for a batch of tokenised input samples.
    Each sample is measured individually to account for variable sequence lengths.

    Parameters:
        model: The model to measure.
        input_ids_batch: Tensor of shape (batch_size, seq_len)

    Returns:
        Total estimated FLOPs (float) for the batch.
    """
    total_flops = 0.0
    batch_size = input_ids_batch.shape[0]


    for i in range(batch_size):
        sample_length = input_ids_batch[i].shape[0]

        def input_constructor(input_res):
            dummy_input = torch.zeros((1,) + input_res, dtype=torch.long).to(input_ids_batch.device)
            return {"input_ids": dummy_input}

        try:
            with io.StringIO() as buf, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                flops_single, _ = ptflops.get_model_complexity_info(
                    model,
                    input_res=(sample_length,),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                    input_constructor=input_constructor
                )
            if flops_single is None:
                logger.warning(f"FLOPs computation returned None for sample {i}. Skipping this sample.")
                continue
            total_flops += flops_single
        except Exception as e:
            logger.warning(f"FLOPs computation failed for sample {i}: {e}")
            continue

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


def combine_comp_metrics(model, device, tokenised_input_ids, accelerator):
    """
    Combines compute-related metrics: FLOPs, memory stats, and device utilisation.
    Only process 0 computes FLOPs and utilisation (to avoid duplication).
    """
    flops = None
    utilisation = None

    if accelerator.local_process_index == 0:
        flops = get_flops(model, tokenised_input_ids)
    
    memory = get_memory(device)

    utilisation = get_gpu_cpu_utilisation(device)

    return {
        "FLOPs": flops,
        "Memory": memory,
        "Compute_utilisation": utilisation,
    }

