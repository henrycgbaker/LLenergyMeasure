import os
import torch
import psutil
import subprocess
from fvcore.nn import FlopCountAnalysis
from src.thesis_package.model_wrapper import ModelWrapper

def detect_cpu_vendor():
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        if "AuthenticAMD" in cpuinfo:
            return "AMD"
        elif "GenuineIntel" in cpuinfo:
            return "Intel"
    except Exception as e:
        print("Error reading /proc/cpuinfo:", e)
    return "Unknown"

def get_compute_performance_metrics(model=None, tokenizer=None, device=None, input_length=128):
    compute_metrics = {}
    cpu_vendor = detect_cpu_vendor()
    compute_metrics["cpu_vendor"] = cpu_vendor
    compute_metrics["gpu"] = str(device) if device is not None else "No device provided"
    
    if torch.cuda.is_available() and device is not None:
        torch.cuda.reset_peak_memory_stats(device)
        compute_metrics["current_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
        compute_metrics["max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        compute_metrics["current_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
        compute_metrics["max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
            )
            lines = result.decode("utf-8").strip().splitlines()
            gpu_utils = [float(line.strip()) for line in lines if line.strip()]
            compute_metrics["gpu_utilization_percent"] = gpu_utils
        except Exception as e:
            compute_metrics["gpu_utilization_percent"] = f"Error: {str(e)}"
    
    try:
        process = psutil.Process(os.getpid())
        cpu_usage = process.cpu_percent(interval=1.0)
        compute_metrics["cpu_usage_percent"] = cpu_usage
        cpu_mem = process.memory_info().rss
        compute_metrics["cpu_memory_usage_bytes"] = cpu_mem
    except Exception:
        compute_metrics["cpu_usage_percent"] = "psutil not installed"
    
    if model is not None and tokenizer is not None and device is not None:
        try:
            model_to_trace = model.module if hasattr(model, "module") else model
            model_to_trace.eval()
            wrapped_model = ModelWrapper(model_to_trace)
            dummy_input_ids = torch.ones((1, input_length), dtype=torch.long).to(device)
            flops_analyzer = FlopCountAnalysis(wrapped_model, dummy_input_ids)
            compute_metrics["flops_forward_pass"] = flops_analyzer.total()
        except Exception as e:
            compute_metrics["flops_forward_pass"] = f"Error: {str(e)}"
    else:
        compute_metrics["flops_forward_pass"] = "model, tokenizer, or device not provided"
    
    return compute_metrics
