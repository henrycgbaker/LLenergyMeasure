from _8_get_compute_info import get_flops, get_memory, get_gpu_cpu_utilisation

def combine_inference_metrics(inference_results):
    raw_metrics = {
        "number_input_prompts": inference_results.get("num_input_prompts", 0),
        "total_input_tokens": inference_results.get("total_input_tokens", 0),
        "total_generated_tokens": inference_results.get("total_generated_tokens", 0),
    }
    
    performance_metrics = {
        "total_inference_time_sec": inference_results.get("total_time_sec", 0),
        "average_latency_ms_per_batch": inference_results.get("avg_latency_ms", 0),
        "throughput_queries_per_sec": inference_results.get("throughput_qps", 0),
        "throughput_tokens_per_sec": inference_results.get("tokens_per_sec", 0),
    }
    
    return {"raw_inference_metrics": raw_metrics, "inference_performance": performance_metrics}


    
def combine_energy_metrics(codecarbon_data, inference_results):
    """
    Combines energy tracking data with inference metrics to derive energy efficiency.
    
    Parameters:
      - codecarbon_data: Tracker data from CodeCarbon.
      - inference_results: Dictionary with inference metrics.
      
    Returns:
      A dictionary of energy-related metrics.
    """
    energy_kwh = getattr(codecarbon_data, "energy_consumed", 0)
    energy_joules = energy_kwh * 3.6e6  # 1 kWh = 3.6e6 joules
    tokens_per_joule = (inference_results.get("total_generated_tokens", 0) / energy_joules) if energy_joules > 0 else 0
    
    return {
        "cpu_power": getattr(codecarbon_data, "cpu_power", None),
        "gpu_power": getattr(codecarbon_data, "gpu_power", None),
        "ram_power": getattr(codecarbon_data, "ram_power", None),
        "cpu_energy": getattr(codecarbon_data, "cpu_energy", None),
        "gpu_energy": getattr(codecarbon_data, "gpu_energy", None),
        "ram_energy": getattr(codecarbon_data, "ram_energy", None),
        "total_energy_kwh": energy_kwh,
        "total_energy_joules": energy_joules,
        "energy_efficiency_tokens_per_joule": tokens_per_joule,
        "final_emissions": getattr(codecarbon_data, "emissions", None),
    }

def combine_comp_metrics(model, device, tokenised_input_ids):
    """
    Combines compute-related metrics: FLOPs, memory stats, and device utilisation.
    
    Parameters:
      - model: The model instance.
      - device: The CUDA device.
      - tokenised_input_ids: Input tensor used for FLOPs calculation.
      
    Returns:
      A dictionary with compute metrics.
    """
    flops = get_flops(model, tokenised_input_ids)
    memory = get_memory(device)
    utilisation = get_gpu_cpu_utilisation(device)
    
    return {
        "FLOPs": flops,
        "Memory": memory,
        "Compute_utilisation": utilisation,
    }

def combine_per_process_results(process_inference_metrics, 
                                process_energy_metrics,
                                process_comp_metrics,
                                accelerator):
    process_id = getattr(accelerator, "pid", None)
    return {
        "process_id": process_id,
        "experiment_results": {
            "inference_performance": process_inference_metrics,
            "energy_performance": process_energy_metrics,
            "compute_performance": process_comp_metrics,
        }
    }