def combine_inference_metrics(inference_results, accelerator):
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
    return {"raw_inference_metrics": raw_metrics, 
            "inference_performance": performance_metrics}