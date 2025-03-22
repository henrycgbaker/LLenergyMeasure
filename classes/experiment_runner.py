import os
import sys
import torch
import logging
import torch.distributed as dist
import logging

logging.getLogger("codecarbon").setLevel(logging.ERROR)

# Adjust paths -> import helper functions
project_root = os.getcwd()  
if project_root not in sys.path:
    sys.path.append(project_root)
helper_functions_path = os.path.join(project_root, "helper_functions")
if helper_functions_path not in sys.path:
    sys.path.append(helper_functions_path)

# from parallel subdirectory:
from _1_distributed_setup import get_accelerator, get_persistent_unique_id, get_shared_unique_id, load_model_tokenizer, get_original_generate_method
from _3_prompt_processing import filter_n_prompts, sort_prompts
from _4_setup_energy_tracking import start_energy_tracking, stop_energy_tracking
from _6_run_inference_by_task import run_gen_inference
from _7_get_experiment_info import get_experiment_setup, get_experimental_variables, get_model_architecture
from _8_get_inference_results import combine_inference_metrics
from _9_get_compute_info import combine_comp_metrics
from _10_get_energy_metrics import combine_energy_metrics
from _11_aggregate_experiment_results import aggregate_results
from _12_save_results import save_raw_results, save_final_results

logger = logging.getLogger(__name__)

class ExperimentRunner: 
    def __init__(self, experiment_config, prompts, **inference_kwargs):
        self.config = experiment_config
        self.prompts = prompts
        self.inference_kwargs = inference_kwargs
        
    def run_torch(self):
        # Safely destroy any existing distributed setup from a previous run
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        
        # Extract configuration parameters.
        model_name       = self.config.model_name
        fp_precision     = self.config.fp_precision
        task_type = self.config.task_type.value if hasattr(self.config.task_type, "value") else self.config.task_type
        inference_type   = self.config.inference_type 
        num_input_prompts= self.config.num_input_prompts
        max_input_tokens = self.config.max_input_tokens
        gpu_list         = self.config.gpu_list
        prompts          = self.prompts

        # Initialize Accelerator.
        accelerator = get_accelerator(gpu_list)
        accelerator.print("Accelerator set up")

        # Generate and share unique ID across processs
        unique_id = get_shared_unique_id(accelerator)
        accelerator.print(f"Unique experiment id: {unique_id}")

        # Load model and tokenizer on main process first.
        with accelerator.main_process_first():
            model_undistributed, tokenizer = load_model_tokenizer(
                model_name=model_name, 
                backend=None, 
                fp_precision=fp_precision
            )
        accelerator.print(f"{model_name} loaded using {self.config.backend}, with precision {fp_precision}")

        # Save original generate method.
        orig_generate_method = get_original_generate_method(model_undistributed)
        if orig_generate_method is None:
            logger.warning("Could not locate the original generate method.")
        else:
            accelerator.print("Original generate method saved.")

        # Prepare model/tokenizer for distributed use.
        model, tokenizer = accelerator.prepare(model_undistributed, tokenizer)
        accelerator.print("Model and tokenizer prepared")
        
        accelerator.wait_for_everyone()
        logger.info(f"[Process {os.getpid()}] Model is on device: {accelerator.device}")
        accelerator.wait_for_everyone()

        # Reassign generate method.
        if orig_generate_method:
            if hasattr(model, "module"):
                model.module.generate = orig_generate_method
            model.generate = orig_generate_method
            accelerator.print("Original generate method reassigned")

        # Dummy forward pass.
        dummy_input = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=max_input_tokens).input_ids.to(accelerator.device)
        with torch.no_grad():
            _ = model(dummy_input)
        logger.info(f"[Process {os.getpid()}] Dummy forward pass complete")
        accelerator.wait_for_everyone()

        # filter & sort prompts based on non-tokenised string length (optimises inference)  
        prompts_n_filtered = filter_n_prompts(prompts=prompts, num_input_prompts=num_input_prompts)
        prompts_sorted = sort_prompts(prompts_n_filtered)
        accelerator.print(f"Prompts processed: {len(prompts_sorted)} prompts.")

        # Start energy tracking.
        self.tracker = start_energy_tracking()
        accelerator.print("Energy tracking started")
        
        # Run inference.
        # NB: all batching, tokenisation, and inference are handled inside run_gen_inference
        if inference_type == "pure_generative":
            accelerator.print(f"Task type: {inference_type}")
            self.raw_text_outputs, input_ids, raw_inference_results = run_gen_inference(
                model=model,
                experiment_config=self.config,
                prompts=prompts_sorted,
                tokenizer=tokenizer,
                accelerator=accelerator
            )
            logger.info(f"[Process {os.getpid()}] Inference complete")

        # Stop energy tracking.
        codecarbon_data = stop_energy_tracking(self.tracker)
        accelerator.print("Energy tracking stopped")
        
        # Compute and save experiment-wide meta info.
        self.experiment_setup     = get_experiment_setup(experiment_config=self.config, model=model, codecarbon_data=codecarbon_data, unique_id=unique_id)
        save_raw_results(unique_id, "1_experiment_setup", self.experiment_setup)
        self.experiment_variables = get_experimental_variables(experiment_config=self.config, model=model, accelerator=accelerator)
        save_raw_results(unique_id, "2_experiment_variables", self.experiment_variables)
        self.model_architecture   = get_model_architecture(model=model)
        save_raw_results(unique_id, "3_model_architecture", self.model_architecture)

        # Save experiment-wide results (only main process): inference & compute
        # TO DO: *SHOULD* BE PER PROCESS (THEN AVERAGED OVER PROCESSES)
        if accelerator.is_main_process:
            self.inference_metrics = combine_inference_metrics(raw_inference_results, accelerator)
            save_raw_results(unique_id, "4_inference_metrics", self.inference_metrics)
            self.compute_metrics      = combine_comp_metrics(model=model, device=accelerator.device, tokenised_input_ids=input_ids, accelerator=accelerator)
            save_raw_results(unique_id, "5_compute_metrics", self.compute_metrics)
            logger.info("Main process saved inference and computation metrics.")
        accelerator.print("Experiment-wide inference and compute metrics saved")
        accelerator.wait_for_everyone()

        # Save per-process energy metrics.
        local_energy_results = combine_energy_metrics(codecarbon_data, accelerator)
        save_raw_results(unique_id, "local_energy_results", local_energy_results, pid=accelerator.local_process_index)
        setattr(self, f"{accelerator.local_process_index}_local_energy_results", local_energy_results)
        logger.info(f"Process {accelerator.local_process_index} saved its energy metrics.")
        accelerator.wait_for_everyone()
        accelerator.print("All local process energy metrics saved")
        
        accelerator.print("Experiment finished")

        return {
            "experiment_id": unique_id,
            "inference_metrics": self.inference_metrics if accelerator.is_main_process else None,
            "compute_metrics": self.compute_metrics if accelerator.is_main_process else None,
            "energy_metrics_path": f"raw_results/{unique_id}/local_energy_results_{accelerator.local_process_index}.json",
            "raw_output_count": len(self.raw_text_outputs) if hasattr(self, "raw_text_outputs") else 0
        }

    def aggregate_results(self):     
        unique_id = self.experiment_setup.unique_id
        # take main process infernece and compute from self.
        # TO DO - LATER SAVE PER PROCESS RESULTS FOR BOTH INFERENCE AND COMPUTE IN THE RUN FUNCTION, THEN TAKE AVG OF THESE ACROSS PROCSSES HERE

        # aggregate energy
        # load each of the processes for that experiment - NB the number will vary between experiments
        # saved above like this : setattr(self, f"{accelerator.local_process_index}_local_energy_results", local_energy_results)
        # calculate total energy for whole experiment across all process
        # calculate tokens_per_joule = (inference_results.get("total_generated_tokens", 0) / energy_joules) if energy_joules > 0 else 0

        # save
        experiment_title = f"EXPERIMENT_#{unique_id}"

        experiment_results = {
                "setup": self.experiment_setup,
                "variables": self.experiment_variables,
                "model_architecture": self.model_architecture,
                "results": {
                    "inference_metrics": self.inference_metrics,  
                    "compute_metrics": self.compute_metrics,
                    "energy_metrics": self.all_energy_metrics
                }
            }
        
        experiment_results = {experiment_title: experiment_results}
        
        # save as JSON
        output_json_path = save_final_results(self.config.task_type, experiment_results)
        logger.info(f"Benchmark results saved to {output_json_path}")
              
        # save as df / csv?
        # IMPLEMENT

        return experiment_results