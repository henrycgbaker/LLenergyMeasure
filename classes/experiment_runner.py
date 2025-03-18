import os
import sys
import torch
import torch.distributed as dist
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adjust paths
project_root = os.getcwd()  
if project_root not in sys.path:
    sys.path.append(project_root)
    
helper_functions_path = os.path.join(project_root, "helper_functions")
if helper_functions_path not in sys.path:
    sys.path.append(helper_functions_path)

# from same subdirectory:
from .experiment_config import ExperimentConfig, load_experiment_config

# from parallel subdirectory:
from _1_model_loader import load_model_tokenizer  # DEPRECATE ModelWrapper?
from _2_distributed_setup import distributed_env_torch, get_original_generate_method
from _3_prompt_processing import filter_n_prompts, sort_prompts
from _4_setup_energy_tracking import start_energy_tracking, stop_energy_tracking
from _5_inference_helper_fns import adaptive_batching, calculate_inference_metrics
from _6_run_inference_by_task import run_gen_inference
from _7_get_experiment_info import (
    get_cores_info, 
    get_region_info, 
    get_experiment_setup, 
    get_experimental_variables, 
    get_model_architecture
)
from _8_get_compute_info import get_flops, get_memory, get_gpu_cpu_utilisation
from _9_combine_process_metrics import (
    combine_inference_metrics, 
    combine_energy_metrics, 
    combine_comp_metrics, 
    combine_per_process_results
)
from _10_aggregate_experiment_results import make_json_serializable, aggregate_experiments
from _11_save_results import get_persistent_unique_id, save_results

# --------------------------------------------------

class ExperimentRunner: 
    def __init__(self, experiment_config, prompts, **inference_kwargs):
        self.config = experiment_config
        self.prompts = prompts
        self.inference_kwargs = inference_kwargs
        
    def run_torch(self):
        # Extract experiment configuration parameters
        model_name = self.config.model_name
        task_type = self.config.task_type.value if hasattr(self.config.task_type, "value") else self.config.task_type
        inference_type = self.config.inference_type 
        num_input_prompts = self.config.num_input_prompts
        max_input_tokens = self.config.max_input_tokens
        gpu_list = self.config.gpu_list
        prompts = self.prompts

        # Load model and tokenizer
        model_undistributed, tokenizer = load_model_tokenizer(model_name, self.config.backend, self.config.fp_precision)
        logger.info(f"{model_name} loaded using backend {self.config.backend} with precision {self.config.fp_precision}.")
            
        # Save original generate method before distribution
        orig_generate_method = get_original_generate_method(model_undistributed)
        if orig_generate_method is None:
            logger.warning("Could not locate the original generate method.")
            
        # Distributed setup 
        # ADD SHARDING OPTIONS HERE USING FSDP
        model, tokenizer, accelerator = distributed_env_torch(model_undistributed, tokenizer, gpu_list)
        logger.info(f"[Process {os.getpid()}] Model is on device: {accelerator.device}")

        # Reassign generate method if necessary
        if orig_generate_method:
            if hasattr(model, "module"):
                model.module.generate = orig_generate_method
            model.generate = orig_generate_method
            accelerator.print("Original generate method reassigned")
    
        # Run a dummy forward pass for lazy GPU allocation (ensure synced when using a distributed env)
        dummy_input = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=max_input_tokens).input_ids.to(accelerator.device)
        with torch.no_grad():
            _ = model(dummy_input)
        logger.info(f"[Process {os.getpid()}] Dummy forward pass complete.")
        
        # Print allocated GPUs        
        gpu_info = ", ".join([f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())])
        accelerator.print(f"Allocated GPUs: {gpu_info}")

        # filter & sort prompts based on non-tokenised string length (optimises inference)  
        prompts_n_filtered = filter_n_prompts(prompts=prompts, num_input_prompts=num_input_prompts)
        prompts_sorted = sort_prompts(prompts_n_filtered)
        accelerator.print(f"Prompts processed: {len(prompts_sorted)} prompts. Longest prompt has {len(prompts_sorted[-1]) if prompts_sorted else 0} characters.")

        # Start energy tracking
        self.tracker = start_energy_tracking()
        accelerator.print("Energy tracking started")
        
        # Run inference 
        # NB: all batching, tokenisation, and inference are handled inside run_gen_inference
        if inference_type is not None and inference_type == "pure_generative":
            accelerator.print(f"{inference_type} task type")
            self.raw_text_outputs, input_ids, raw_inference_results = run_gen_inference(
                model=model,
                experiment_config=self.config,
                prompts=prompts_sorted,
                tokenizer=tokenizer,
                accelerator=accelerator
            )
            logger.info(f"[Process {os.getpid()}] Inference complete.")

        # Stop energy tracking
        codecarbon_data = stop_energy_tracking(self.tracker)
        accelerator.print("Energy tracking stopped")
        
        # Collect per-process results 
        process_inference_metrics = combine_inference_metrics(raw_inference_results, codecarbon_data)
        process_power_energy_metrics = combine_energy_metrics(codecarbon_data, raw_inference_results)
        process_comp_metrics = combine_comp_metrics(model=model, device=accelerator.device, tokenised_input_ids=input_ids)

        local_process_result = combine_per_process_results(
            process_inference_metrics, 
            process_power_energy_metrics, 
            process_comp_metrics,
            accelerator=accelerator
        )
        
        # Synchronize processes 
        accelerator.wait_for_everyone()
        
        # Gather results across distributed processes -> all_results
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            all_results = [None] * world_size
            logger.info(f"[Process {os.getpid()}] Gathering results from {world_size} processes.")
            dist.all_gather_object(all_results, local_process_result)
        else:
            all_results = [local_process_result]
            
        # all_results -> all_process_results
        if accelerator.local_process_index == 0:
            aggregated_result = aggregate_experiments(all_results)
            self.all_process_results = aggregated_result
        else:
            self.all_process_results = None  # Only the main process aggregates
            
        # Collect experiment meta information
        self.experiment_setup = get_experiment_setup(experiment_config=self.config, model=model, codecarbon_data=codecarbon_data)
        self.experiment_variables = get_experimental_variables(experiment_config=self.config, model=model, accelerator=accelerator)
        self.model_architecture = get_model_architecture(model=model)
        
        # Save experiment results to a JSON log
        unique_id = get_persistent_unique_id()
        experiment_title = f"EXPERIMENT_{unique_id}"
        experiment_results = {
            "setup": self.experiment_setup,
            "variables": self.experiment_variables,
            "model_architecture": self.model_architecture,
            "results": self.all_process_results
        }
        benchmark_results = {experiment_title: experiment_results}
        
        output_json_path = save_results(task_type, benchmark_results)
        logger.info(f"Benchmark results saved to {output_json_path}")
        
        return benchmark_results
    