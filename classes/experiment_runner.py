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
from _11_save_results import save_raw_results, save_final_results
from _12_aggregate_experiment_results import load_local_energy_results

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
        torch.cuda.empty_cache()
        
        # Extract configuration parameters.
        model_name       = self.config.model_name
        fp_precision     = self.config.fp_precision
        inference_type   = self.config.inference_type 
        num_input_prompts= self.config.num_input_prompts
        max_input_tokens = self.config.max_input_tokens
        gpu_list         = self.config.gpu_list
        prompts          = self.prompts

        # Initialize Accelerator.
        accelerator = get_accelerator(gpu_list)
        self.accelerator = accelerator
        accelerator.print("Accelerator set up")

        # Generate and share unique ID across processs
        experiment_id = get_shared_unique_id(accelerator)
        self.experiment_id = experiment_id
        accelerator.print(f"Unique experiment id: {experiment_id}")

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
        tracker = start_energy_tracking()
        accelerator.print("Energy tracking started")
        
        # Run inference.
        # NB: all batching, tokenisation, and inference are handled inside run_gen_inference
        if inference_type == "pure_generative":
            accelerator.print(f"Task type: {inference_type}")
            token_id_outputs, input_ids, raw_inference_results = run_gen_inference(
                model=model,
                experiment_config=self.config,
                prompts=prompts_sorted,
                tokenizer=tokenizer,
                accelerator=accelerator
            )
        logger.info(f"[Process {os.getpid()}] Inference complete")
        
        
        # Conditionally decode token_id output.
        if accelerator.is_main_process:
            if self.config.decode_token_to_text:
                try:
                    decoded_texts = []
                    for batch in token_id_outputs:
                        # Convert the batch to a list if needed.
                        if isinstance(batch, torch.Tensor):
                            batch_list = batch.tolist()
                        else:
                            batch_list = batch
                        # Decode the current batch.
                        decoded_batch = tokenizer.batch_decode(batch_list, skip_special_tokens=True)
                        decoded_texts.extend(decoded_batch)
                    text_outputs = decoded_texts
                    accelerator.print("Decoded token outputs successfully.")
                except Exception as e:
                    accelerator.print(f"Error during batch decoding: {e}")
                    text_outputs = None
            else:
                text_outputs = None
        else:
            text_outputs = None

        
        # Stop energy tracking.
        codecarbon_data = stop_energy_tracking(tracker)
        accelerator.print("Energy tracking stopped")
        
        accelerator.wait_for_everyone()

        #  Conditionally save outputs.
        if self.config.save_outputs:
            if self.config.decode_token_to_text:
                outputs = text_outputs
                save_raw_results(experiment_id, "8_text_output", outputs)
                accelerator.print("Saved text outputs")
            else:
                outputs = [tensor.tolist() for tensor in token_id_outputs]
                save_raw_results(experiment_id, "8_token_output", outputs)
                accelerator.print("Saved token outputs")
        else:
            self.outputs = None
            accelerator.print("Did not save output")
        
        # Compute and save experiment-wide meta info.
        self.experiment_setup     = get_experiment_setup(experiment_config=self.config, model=model, codecarbon_data=codecarbon_data, experiment_id=experiment_id)
        save_raw_results(experiment_id, "1_experiment_setup", self.experiment_setup)
        self.experiment_variables = get_experimental_variables(experiment_config=self.config, model=model, accelerator=accelerator)
        save_raw_results(experiment_id, "2_experiment_variables", self.experiment_variables)
        self.model_architecture   = get_model_architecture(model=model)
        save_raw_results(experiment_id, "3_model_architecture", self.model_architecture)

        # Save experiment-wide results (only main process): inference & compute
        # TO DO: *SHOULD* BE PER PROCESS (THEN AVERAGED OVER PROCESSES)
        if accelerator.is_main_process:
            self.inference_metrics = combine_inference_metrics(raw_inference_results, accelerator)
            save_raw_results(experiment_id, "4_inference_metrics", self.inference_metrics)
            self.compute_metrics      = combine_comp_metrics(model=model, device=accelerator.device, tokenised_input_ids=input_ids, accelerator=accelerator)
            save_raw_results(experiment_id, "5_compute_metrics", self.compute_metrics)
            logger.info("Main process saved inference and computation metrics.")
        accelerator.print("Experiment-wide inference and compute metrics saved")
        accelerator.wait_for_everyone()

        # Save per-process energy metrics.
        local_energy_results = combine_energy_metrics(codecarbon_data, accelerator)
        save_raw_results(experiment_id, "6_local_energy_results", local_energy_results, pid=accelerator.local_process_index)
        setattr(self, f"local_energy_results_{accelerator.local_process_index}", local_energy_results)
        logger.info(f"Process {accelerator.local_process_index} saved its energy metrics.")
        accelerator.wait_for_everyone()
        accelerator.print("All local process energy metrics saved")
        
        accelerator.print("Experiment finished")

        return 
                
    def aggregate_results(self):
        """
        Aggregates per-process energy metrics (loaded from JSON files) into a global energy results dict.
        Outputs per-process metrics (dict keyed by process index), averages, derived metrics, and emissions.
        """
        
        logger.info(f"Aggregating per process results")

        # Get the unique experiment ID 
        experiment_id = self.experiment_id
        
        # Load per-process energy results from JSON files
        per_process = load_local_energy_results(experiment_id)
        
        # Prepare dictionaries for each metric.
        metrics = [
            "cpu_power", "gpu_power", "ram_power",
            "cpu_energy", "gpu_energy", "ram_energy",
            "total_energy_kwh", "total_energy_joules"
        ]
        per_process_results = {metric: {} for metric in metrics}
        for pid, energy_dict in per_process.items():
            for metric in metrics:
                per_process_results[metric][pid] = energy_dict.get(metric, 0)
        
        # Compute average for each metric.
        averages = {}
        for metric, pid_dict in per_process_results.items():
            values = list(pid_dict.values())
            averages[metric] = sum(values) / len(values) if values else 0
        
        # Derived metrics: tokens_per_joule, joules_per_token, flops_per_joule.
        if self.inference_metrics is not None:
            raw_inf = self.inference_metrics.get("raw_inference_metrics", {})
            total_generated_tokens = raw_inf.get("total_generated_tokens", 0)
        else:
            total_generated_tokens = 0

        avg_energy_joules = averages.get("total_energy_joules", 0)
        tokens_per_joule = total_generated_tokens / avg_energy_joules if avg_energy_joules > 0 else 0
        joules_per_token = 1 / tokens_per_joule if tokens_per_joule > 0 else 0

        flops = getattr(self, "flops", 0)
        flops_per_joule = flops / avg_energy_joules if avg_energy_joules > 0 else 0

        derived = {
            "tokens_per_joule": tokens_per_joule,
            "joules_per_token": joules_per_token,
            "flops_per_joule": flops_per_joule
        }

        # Aggregate emissions: flatten per-process emissions.
        emissions = {}   # individual emissions per process
        all_emissions = []  # flatten all emissions into one list
        for pid, energy_dict in per_process.items():
            em = energy_dict.get("final_emissions")
            emissions[pid] = em
            if isinstance(em, list):
                all_emissions.extend(em)
            elif em is not None:
                all_emissions.append(em)            
        
        # --- Build the global energy results dictionary ---
        global_energy_results = {
            "experiment_id": experiment_id,
            "process_results": per_process_results,
            "experiment_avg": averages,
            "experiment_derived": derived,
            "experiment_emissions": all_emissions,
        }
        self.global_energy_results = global_energy_results
        
        # Save the aggregated results as JSON 
        save_raw_results(experiment_id, "7_global_energy_results", global_energy_results)
        self.global_energy_results = global_energy_results
        
        logger.info(f"Local process result aggregated to global experiment results")
        
        return global_energy_results
        
    def save_experiment_results(self):
        
        logger.info(f"Saving experiment results")
        
        experiment_id = self.experiment_id
        experiment_title = f"EXPERIMENT_#{experiment_id}"

        experiment_results = {
                "setup": self.experiment_setup,
                "variables": self.experiment_variables,
                "model_architecture": self.model_architecture,
                "results": {
                    "inference_metrics": self.inference_metrics,  
                    "compute_metrics": self.compute_metrics,
                    "energy_metrics": self.global_energy_results
                }
            }
        experiment_results = {experiment_title: experiment_results}
        
        # save as JSON
        output_json_path = save_final_results(self.config.task_type, experiment_results)
        
        logger.info(f"Experiment results saved to {output_json_path}")

        return experiment_results
    
    
    def inspect_attributes(self):
            """Prints all attributes of the ExperimentRunner for inspection."""
            print("ExperimentRunner Attributes:")
            for attr, value in self.__dict__.items():
                print(f"  {attr}: {value}")
                