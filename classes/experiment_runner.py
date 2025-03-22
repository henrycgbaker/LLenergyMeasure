import os
import sys
import glob
import torch
import torch.distributed as dist
import logging
import json

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
from _1_distributed_setup import get_original_generate_method, get_accelerator, load_model_tokenizer
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
from _8_get_inference_results import combine_inference_metrics
from _9_get_compute_info import get_flops, get_memory, get_gpu_cpu_utilisation, combine_comp_metrics
from _10_get_energy_metrics import combine_energy_metrics
from _11_aggregate_experiment_results import make_json_serializable, aggregate_experiments
from _12_save_results import save_raw_results, save_final_results

# --------------------------------------------------

# MOVE OUT LATER

# A helper to aggregate energy metrics from files
def aggregate_results(results_dir):
    files = glob.glob(os.path.join(results_dir, "energy_metrics_rank_*.json"))
    aggregated = []
    for file in files:
        with open(file, "r") as f:
            aggregated.append(json.load(f))
    return aggregated

# --------------------------------------------------


class ExperimentRunner: 
    def __init__(self, experiment_config, prompts, **inference_kwargs):
        self.config = experiment_config
        self.prompts = prompts
        self.inference_kwargs = inference_kwargs
        
    def run_torch(self):
        # Extract experiment configuration parameters
        model_name = self.config.model_name
        fp_precision = self.config.fp_precision
        task_type = self.config.task_type.value if hasattr(self.config.task_type, "value") else self.config.task_type
        inference_type = self.config.inference_type 
        num_input_prompts = self.config.num_input_prompts
        max_input_tokens = self.config.max_input_tokens
        gpu_list = self.config.gpu_list
        prompts = self.prompts
        
        # get unique id for experiment (used for saving results in JSONs)
        unique_id = get_persistent_unique_id()

        # Distributed setup 
        # TO DO ADD SHARDING OPTIONS HERE USING FSDP
        accelerator = get_accelerator(gpu_list)
        accelerator.print(f"Accelerator set up")
        
        # Load model and tokenizer
        with accelerator.main_process_first():
            model_undistributed, tokenizer = load_model_tokenizer(
                model_name=model_name, 
                backend=None, 
                fp_precision=fp_precision
                )
        accelerator.print(f"{model_name} loaded using backend {self.config.backend} with precision {self.config.fp_precision}.")

        # Save original generate method before distribution
        orig_generate_method = get_original_generate_method(model_undistributed)
        if orig_generate_method is None:
            logger.warning("Could not locate the original generate method.")
            
        # Prepare model/tokenizer for distributed use
        model, tokenizer = accelerator.prepare(model_undistributed, tokenizer)
        accelerator.print(f"Model and tokenizer prepared")
        
        # Reassign generate method if necessary
        if orig_generate_method:
            if hasattr(model, "module"):
                model.module.generate = orig_generate_method
            model.generate = orig_generate_method
            accelerator.print("Original generate method reassigned")
        
        logger.info(f"[Process {os.getpid()}] Model is on device: {accelerator.device}")
    
        # Run a dummy forward pass for lazy GPU allocation (ensure synced when using a distributed env)
        dummy_input = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=max_input_tokens).input_ids.to(accelerator.device)
        with torch.no_grad():
            _ = model(dummy_input)
        logger.info(f"[Process {os.getpid()}] Dummy forward pass complete.")
        accelerator.wait_for_everyone()
        
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
        
        # get experiment-wide meta info (constant to all processes)
        self.experiment_setup = get_experiment_setup(experiment_config=self.config, model=model, codecarbon_data=codecarbon_data)
        self.experiment_variables = get_experimental_variables(experiment_config=self.config, model=model, accelerator=accelerator)
        self.model_architecture = get_model_architecture(model=model)

        # get experiment-wide results (constant to all processes)
        # (i) inferenece
        if accelerator.is_main_process:
            inference_metrics = combine_inference_metrics(raw_inference_results, accelerator)
            save_raw_results(unique_id, "inference_metrics", inference_metrics)
        accelerator.print("Inference metrics saved")
        accelerator.wait_for_everyone()

        # (ii) computation
        if accelerator.is_main_process:
            comp_metrics = combine_comp_metrics(model=model, 
                                                device=accelerator.device, 
                                                tokenised_input_ids=input_ids, 
                                                accelerator=accelerator)
            save_raw_results(unique_id, "compute_metrics", comp_metrics)
        accelerator.print("Computation metrics got")
        accelerator.wait_for_everyone()
        
        # Per-process energy metrics: Every process computes its own
        # (i) Compute local energy metrics
        local_energy_results = combine_energy_metrics(codecarbon_data, accelerator)

        # Save local results to a JSON file uniquely identified by the process rank.
        save_raw_results(unique_id, "local_energy_results", local_energy_results, pid=os.getpid())

        accelerator.print(f"Process {accelerator.local_process_index} saved its energy metrics.")
        accelerator.wait_for_everyone()
        
        import glob
        def aggregate_results(results_dir):
            files = glob.glob(os.path.join(results_dir, "energy_metrics_rank_*.json"))
            aggregated = []
            for file in files:
                with open(file, "r") as f:
                    aggregated.append(json.load(f))
            return aggregated

        # This could run only on the main process or in a separate script.
        all_energy_metrics = aggregate_results("/benchmark_results/local_process_energy_results")
        print("Aggregated energy metrics:", all_energy_metrics)
        
        
        local_process_energy_metrics = combine_energy_metrics(codecarbon_data, accelerator)
        accelerator.wait_for_everyone()

        if hasattr(accelerator, "gather"):
            all_energy_metrics = accelerator.gather(local_process_energy_metrics)
        else:
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                all_energy_metrics = [None] * world_size
                dist.all_gather_object(all_energy_metrics, local_process_energy_metrics)
            else:
                all_energy_metrics = [local_process_energy_metrics]

        self.all_power_energy_results = all_energy_metrics

        accelerator.wait_for_everyone()
        accelerator.print("All energy metrics gathered")
        
        # Save experiment results to a JSON log
        experiment_title = f"EXPERIMENT_{unique_id}"
        experiment_results = {
            "setup": self.experiment_setup,
            "variables": self.experiment_variables,
            "model_architecture": self.model_architecture,
            "results": {
                "inference_metrics": self.inference_metrics,
                "compute_metrics": self.comp_metrics,
                "energy metrics": self.all_power_energy_results
            }
        }
        benchmark_results = {experiment_title: experiment_results}
        
        output_json_path = save_final_results(task_type, benchmark_results)
        logger.info(f"Benchmark results saved to {output_json_path}")
        
        return benchmark_results
    