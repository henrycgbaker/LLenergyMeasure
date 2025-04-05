import os
import sys
import torch
import logging
import torch.distributed as dist
import logging
import torch
import gc

logging.getLogger("codecarbon").setLevel(logging.ERROR)

# Adjust paths -> import helper functions
project_root = os.getcwd()  
if project_root not in sys.path:
    sys.path.append(project_root)
helper_functions_path = os.path.join(project_root, "helper_functions")
if helper_functions_path not in sys.path:
    sys.path.append(helper_functions_path)

# from experiment_core's helper functions
from experiment_core_utils.a_distributed import get_accelerator, get_shared_unique_id, get_original_generate_method, safe_wait
from experiment_core_utils.b_model_loader import load_model_tokenizer
from experiment_core_utils.c_prompt_processing import filter_n_prompts, sort_prompts
from experiment_core_utils.d_energy_tracking import warm_up, start_energy_tracking, stop_energy_tracking
from experiment_core_utils.e_inference import run_gen_inference
from experiment_core_utils.f_experiment_info import get_experiment_setup, get_experimental_variables, get_model_architecture
from experiment_core_utils.g_metrics_inference import combine_inference_metrics
from experiment_core_utils.h_metrics_compute import combine_comp_metrics, combine_comp_metrics_fvcore
from experiment_core_utils.i_metrics_energy import combine_energy_metrics
from experiment_core_utils.j_results_saving import save_raw_results_json, save_final_results_json
from experiment_core_utils.k_results_aggregation import load_local_energy_results

logger = logging.getLogger(__name__)

class ExperimentRunner: 
    def __init__(self, experiment_config, prompts, **inference_kwargs):
        self.config = experiment_config
        self.prompts = prompts
        self.inference_kwargs = inference_kwargs # if i build in non-text gen tasks types (e.g sumamrization etc)

    def run_setup(self):
        # Safely destroy any existing distributed setup from a previous run.
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Warning: could not destroy previous process group: {e}")
        torch.cuda.empty_cache()
        

    def unique_id(self):
        # Initialize Accelerator.
        accelerator = get_accelerator(self.config.gpu_list, self.config.num_processes)
        self.accelerator = accelerator
        accelerator.print("Accelerator set up")

        # Generate and share unique ID across processes.
        experiment_id = get_shared_unique_id(accelerator)
        self.experiment_id = experiment_id
        accelerator.print(f"Unique experiment id: {experiment_id}")

    def run_torch(self):

        # Extract configuration parameters.
        model_name        = self.config.model_name
        inference_type    = self.config.inference_type 
        num_input_prompts = self.config.num_input_prompts
        max_input_tokens  = self.config.max_input_tokens
        
        accelerator = self.accelerator
        experiment_id = self.experiment_id

        # Load model and tokenizer on main process first.
        with accelerator.main_process_first():
            model_undistributed, tokenizer = load_model_tokenizer(self.config)
        accelerator.print(f"{model_name} loaded using {self.config.backend}, with precision {self.config.fp_precision}")
        safe_wait(accelerator, "after load_model_tokenizer")

        # Save original generate method.
        orig_generate_method = get_original_generate_method(model_undistributed)
        if orig_generate_method is None:
            logger.warning("Could not locate the original generate method.")
        else:
            accelerator.print("Original generate method saved.")

        # Prepare model/tokenizer for distributed use.
        model, tokenizer = accelerator.prepare(model_undistributed, tokenizer)
        accelerator.print("Model and tokenizer prepared")
        
        safe_wait(accelerator, "after model preparation")
        logger.info(f"[Process {os.getpid()}] Model is on device: {accelerator.device}")
        safe_wait(accelerator, "after logging device info")

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
        safe_wait(accelerator, "after dummy forward pass")
        
        # Warm-up here on all processes
        warm_up(model, tokenizer, self.config, num_warmup_runs=3)
        safe_wait(accelerator, "after warm up")

        # Filter & sort prompts based on non-tokenised string length.
        prompts_n_filtered = filter_n_prompts(prompts=self.prompts, num_input_prompts=num_input_prompts)
        prompts_sorted = sort_prompts(prompts_n_filtered)
        accelerator.print(f"Prompts processed: {len(prompts_sorted)} prompts.")

        # Start energy tracking.
        tracker = start_energy_tracking()
        accelerator.print("Energy tracking started")
        
        # Run inference.
        if inference_type == "pure_generative":
            accelerator.print(f"Task type: {inference_type}")
            try:
                token_id_outputs, input_ids, raw_inference_results = run_gen_inference(
                    model=model,
                    experiment_config=self.config,
                    prompts=prompts_sorted,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                )
            except Exception as e:
                accelerator.print(f"Error during inference: {e}")
                raise
        logger.info(f"[Process {os.getpid()}][GPU {accelerator.device.index}]: Inference complete")

        # Stop energy tracking.
        codecarbon_data = stop_energy_tracking(tracker)
        logger.info(f"[Process {os.getpid()}][GPU {accelerator.device.index}]: Energy tracking stopped")

        safe_wait(accelerator, "after energy tracking stopped")
        
        # Conditionally decode token_id output (only main process).
        if accelerator.is_main_process:
            if self.config.decode_token_to_text:
                try:
                    decoded_texts = []
                    for batch in token_id_outputs:
                        batch_list = batch.tolist() if isinstance(batch, torch.Tensor) else batch
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

        # Conditionally save text/token outputs.
        if accelerator.is_main_process:
            if self.config.save_outputs:
                if self.config.decode_token_to_text:
                    outputs = text_outputs if text_outputs else []
                else:
                    outputs = [tensor.tolist() for tensor in token_id_outputs] if token_id_outputs else []
                if not isinstance(outputs, list):
                    logger.error(f"[{experiment_id}] Outputs not a list before saving: type={type(outputs)}")
                    outputs = []
                save_raw_results_json(experiment_id=experiment_id, 
                                 type="8_text_output" if self.config.decode_token_to_text else "8_token_output", 
                                 results=outputs, 
                                 pid=None)
                accelerator.print("Saved outputs")
            else:
                self.outputs = None
                accelerator.print("Did not save output")
        
        # Save experiment-wide meta info (only main process).
        if accelerator.is_main_process:
            self.experiment_setup = get_experiment_setup(
                experiment_config=self.config, codecarbon_data=codecarbon_data, experiment_id=experiment_id
            )
            save_raw_results_json(experiment_id, "1_experiment_setup", self.experiment_setup)
            self.experiment_variables = get_experimental_variables(
                experiment_config=self.config, model=model, accelerator=accelerator
            )
            save_raw_results_json(experiment_id, "2_experiment_variables", self.experiment_variables)
            self.model_architecture = get_model_architecture(model=model)
            save_raw_results_json(experiment_id, "3_model_architecture", self.model_architecture)
            logger.info("Main process saved (i) experiment setup, (ii) variables, (iii) model architecture.")
            
        accelerator.print("Experiment-wide meta info saved")

        # Save experiment-wide results (only main process).
        if accelerator.is_main_process:
            self.inference_metrics = combine_inference_metrics(raw_inference_results, accelerator)
            save_raw_results_json(experiment_id, "4_inference_metrics", self.inference_metrics)
            self.compute_metrics = combine_comp_metrics(
                model=model, device=accelerator.device, tokenised_input_ids=input_ids, accelerator=accelerator, experiment_config=self.config
            )
            save_raw_results_json(experiment_id, "5_compute_metrics", self.compute_metrics)
            logger.info("Main process saved inference and computation metrics.")
            
        accelerator.print("Experiment-wide inference and compute metrics saved")
        safe_wait(accelerator, "after saving experiment metrics")
        
        # Save per-process energy metrics.
        try:
            local_energy_results = combine_energy_metrics(codecarbon_data, accelerator)
            logger.info(f"Process {accelerator.local_process_index}: Energy metrics combined successfully.")
        except Exception as e:
            logger.error(f"Process {accelerator.local_process_index}: Error in combine_energy_metrics: {e}")
            local_energy_results = None
            
        save_raw_results_json(experiment_id, "6_local_energy_results", local_energy_results, pid=accelerator.local_process_index)
        setattr(self, f"local_energy_results_{accelerator.local_process_index}", local_energy_results)
        logger.info(f"[Process {os.getpid()}][GPU {accelerator.device.index}] saved its energy metrics.")
                
        accelerator.print("All local process energy metrics saved")
        
        accelerator.print("Experiment finished")

        # Final cleanup: attempt to destroy the process group.
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
                accelerator.print("Destroyed process group successfully")
            except Exception as e:
                accelerator.print(f"Error during process group destruction: {e}")
        
        return
    
                
    def aggregate_results(self):
        """
        Aggregates per-process energy metrics (loaded from JSON files) into a global energy results dict.
        Outputs per-process metrics (dict keyed by process index), averages, derived metrics, and emissions.
        Derived metrics:
        - tokens_per_joule: total_generated_tokens divided by the sum of total_energy_joules across processes.
        - joules_per_token: the reciprocal of tokens_per_joule.
        - flops_per_joule: self.flops divided by the sum of total_energy_joules.
        - joules_per_flop: the reciprocal of flops_per_joule.
        The results are saved as a JSON file.
        """
        
        logger.info("Aggregating per-process energy metrics from disk.")
        # Get the unique experiment ID.
        experiment_id = self.experiment_id

        # Load per-process energy results from JSON files.
        per_process = load_local_energy_results(experiment_id)
        if not per_process:
            logger.warning("No per-process energy JSON files found!")
            return {}

        # List of metrics that are rates (to be averaged) and counts (to be summed).
        energy_keys_avg = ["cpu_power", "gpu_power", "ram_power"]
        energy_keys_sum = ["cpu_energy", "gpu_energy", "ram_energy"]
        combined_keys_sum = ["total_energy_kwh", "total_energy_joules"]
        
        # Prepare dictionaries for per-process values.
        per_process_results = {metric: {} for metric in energy_keys_avg + energy_keys_sum + combined_keys_sum}
        for pid, energy_dict in per_process.items():
            inner = energy_dict.get("energy_results", {})  # Access the nested dictionary.
            for key in energy_keys_avg + energy_keys_sum + combined_keys_sum:
                per_process_results[key][pid] = inner.get(key, 0)
        
        # Compute averages for power rate metrics and rename with "_avg" suffix
        averages = {}
        for key in energy_keys_avg:
            values = list(per_process_results[key].values())
            averages[f"{key}_avg"] = sum(values) / len(values) if values else 0

        # Sum for energy count metrics and rename with "_total" suffix
        for key in energy_keys_sum:
            values = list(per_process_results[key].values())
            averages[f"{key}_total"] = sum(values)
        
        # Sum for combined count
        for key in combined_keys_sum:
            values = list(per_process_results[key].values())
            averages[f"{key}"] = sum(values)    
        
        # At this point, averages["total_energy_joules"] is the sum of energy (in joules) over all processes.
        total_energy_joules_sum = averages.get("total_energy_joules", 0)
        
        # Derived Metrics:
        # Use inference_metrics from the main process to get total generated tokens.
        if self.inference_metrics is not None:
            raw_inf = self.inference_metrics.get("raw_inference_metrics", {})
            total_generated_tokens = raw_inf.get("total_generated_tokens", 0)
        else:
            total_generated_tokens = 0

        tokens_per_joule = total_generated_tokens / total_energy_joules_sum if total_energy_joules_sum > 0 else 0
        joules_per_token = 1 / tokens_per_joule if tokens_per_joule > 0 else 0

        # self.flops is assumed to be computed on the main process.
        flops = self.compute_metrics.get("flops", 0)
        flops_per_joule = flops / total_energy_joules_sum if total_energy_joules_sum > 0 else 0
        joules_per_flop = total_energy_joules_sum / flops if flops > 0 else 0

        derived = {
            "tokens_per_joule": tokens_per_joule,
            "joules_per_token": joules_per_token,
            "flops_per_joule": flops_per_joule,
            "joules_per_flop": joules_per_flop,
        }

        # Aggregate emissions: flatten per-process emissions.
        emissions = {}
        all_emissions = []
        for pid, energy_dict in per_process.items():
            inner = energy_dict.get("energy_results", {})
            em = inner.get("final_emissions")
            emissions[pid] = em
            if isinstance(em, list):
                all_emissions.extend(em)
            elif em is not None:
                all_emissions.append(em)
        
        global_energy_results = {
            "experiment_id": experiment_id,
            "local_process_results": per_process_results,
            "global_experiment_results": averages,
            "global_derived_quantities": derived,
            "per-process_emissions": all_emissions,
        }
        self.global_energy_results = global_energy_results

        # Save the aggregated results as JSON.
        save_raw_results_json(experiment_id, "7_global_energy_results", global_energy_results)
        logger.info("Aggregated global energy results successfully from disk.")
        
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
        output_json_path = save_final_results_json(self.config.task_type, experiment_results)
        logger.info(f"Experiment results saved to {output_json_path}")
        
        # save as tabular row
        # INSERT HERE

        return experiment_results

    def teardown(self):
        print("Starting teardown process...")

        # Only the main process destroys the distributed process group.
        if dist.is_available() and dist.is_initialized():
            if self.accelerator.is_main_process:
                try:
                    print("Destroying distributed process group...")
                    dist.destroy_process_group()
                except Exception as e:
                    print(f"Exception during process group destruction: {e}")
            else:
                # Non-main processes wait for the main process to destroy the group.
                try:
                    dist.barrier()
                except Exception as e:
                    print(f"Exception during barrier wait in teardown: {e}")
        else:
            print("Distributed package is not available or process group not initialized.")

        # Clean up GPU memory.
        try:
            print("Emptying CUDA cache...")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Exception while emptying CUDA cache: {e}")

        # Run garbage collection.
        try:
            print("Running garbage collection...")
            gc.collect()
        except Exception as e:
            print(f"Exception during garbage collection: {e}")

        print("Teardown process complete.")

    def inspect_attributes(self):
            """Prints all attributes of the ExperimentRunner for inspection."""
            print("ExperimentRunner Attributes:")
            for attr, value in self.__dict__.items():
                print(f"  {attr}: {value}")