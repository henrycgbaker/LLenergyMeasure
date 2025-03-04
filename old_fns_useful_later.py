# IN RUN_EXPERIMENT()        
    # Each process computes its own local result.
        local_result = extract_experiment_results(
            inference_metrics, codecarbon_data, model=model, tokenizer=tokenizer, device=accelerator.device
            )   
 # Gather the local_result from all processes into a list.
        # Create a list with a placeholder for each process
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        results_list = [None for _ in range(world_size)]

        # Gather local_result from all processes
        dist.all_gather_object(results_list, local_result)

        # Use results_list for aggregation
        aggregated_results = aggregate_process_results(results_list)

# THEN THIS HELPER FUNCTION: 

def aggregate_process_results(results_list): # IM NOT SURE IF THIS IS NEEDED
    """
    Aggregate a list of experiment_results (one per process) into a single dictionary.
    
    For example, this function sums total inference time, total tokens, and energy,
    and averages the latency. Adjust as needed for your metrics.
    """
    aggregated = {
        "inference_performance": {},
        "energy_performance": {},
        "compute_performance": {}
    }
    
    # average inference performance across processes
    avg_process_time = sum(r["inference_performance"].get("total_inference_time_sec", 0) for r in results_list) / len(results_list)
    total_tokens = sum(r["inference_performance"].get("total_tokens_generated", 0) for r in results_list) / len(results_list)
    total_runs = sum(r["inference_performance"].get("num_runs", 0) for r in results_list) / len(results_list)
    avg_latency = sum(r["inference_performance"].get("average_latency_ms_per_batch", 0) for r in results_list) / len(results_list)
    avg_throughput = sum(r["inference_performance"].get("throughput_tokens_per_sec", 0) for r in results_list) / len(results_list)
    
    aggregated["inference_performance"] = {
        "total_inference_time_sec": avg_process_time,
        "total_tokens_generated": total_tokens,
        "num_runs": total_runs,
        "average_latency_ms_per_batch": avg_latency,
        "throughput_tokens_per_sec": avg_throughput
    }
    
    # Aggregate energy performance: sum energy consumed and recalc tokens per joule.
    total_energy_kwh = sum(r["energy_performance"].get("total_energy_consumed_kwh", 0) for r in results_list)
    total_energy_joules = sum(r["energy_performance"].get("total_energy_consumed_joules", 0) for r in results_list)
    tokens_per_joule = total_tokens / total_energy_joules if total_energy_joules > 0 else 0
    
    aggregated["energy_performance"] = {
        "total_energy_consumed_kwh": total_energy_kwh,
        "total_energy_consumed_joules": total_energy_joules,
        "energy_efficiency_tokens_per_joule": tokens_per_joule
    }
    
    # compute performance
    aggregated["compute_performance"] = results_list[0].get("compute_performance", {})
    
    return aggregated
