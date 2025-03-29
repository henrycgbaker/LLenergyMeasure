import itertools
import copy
import random
import time
import logging

logger = logging.getLogger(__name__)

def run_single_experiment_with_retries(runner, max_retries=3, retry_delay=5):
    """
    Attempts to run a single experiment (i.e. one configuration) using runner.run_torch() 
    up to max_retries times.
    """
    attempt = 0
    result = None
    while attempt < max_retries:
        try:
            logger.info(f"Starting experiment run attempt {attempt+1}/{max_retries}")
            result = runner.run_torch()
            # If this is not the main process, return immediately.
            if not runner.accelerator.is_main_process:
                return True, result
            # Only main process continues aggregation and saving.
            runner.aggregate_results()
            runner.save_experiment_results()
            runner.teardown()
            logger.info("Experiment run succeeded.")
            return True, result
        except Exception as e:
            attempt += 1
            logger.error(f"Experiment run failed on attempt {attempt}: {e}", exc_info=True)
            time.sleep(retry_delay)
    logger.error("Experiment run failed after maximum attempts.")
    return False, None

def generate_configurations(base_config, grid_params):
    """
    Generate a list of configuration dictionaries by taking the Cartesian product
    of the grid_params and merging them with the base_config.
    
    Parameters:
      base_config (dict): The constant/default parameters.
      grid_params (dict): Mapping from parameter name to list of possible values.
    
    Returns:
      List of configuration dictionaries.
    """
    # Get grid keys and corresponding value combinations.
    keys = list(grid_params.keys())
    value_combinations = list(itertools.product(*[grid_params[k] for k in keys]))
    configs = []
    for combo in value_combinations:
        config = copy.deepcopy(base_config)
        for key, value in zip(keys, combo):
            # Update nested dictionaries if necessary.
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        # Optionally validate the configuration here to enforce derived constraints.
        # For example: ensure total output tokens remain 100,000, etc.
        if validate_config(config):
            configs.append(config)
    return configs

def validate_config(config):
    """
    Validate and normalize the configuration.
    For instance, ensure that the total output tokens remain 100,000 and FLOPs remain constant.
    This function should also enforce dependencies (e.g., quantization disables fp_precision).
    
    Returns True if valid; False otherwise.
    """
    # Example: Ensure that output tokens (max_output_tokens * num_input_prompts) equals 100,000.
    total_output_tokens = config.get("max_output_tokens", 0) * config.get("num_input_prompts", 0)
    if total_output_tokens != 100000:
        # Optionally adjust one parameter to meet the requirement,
        # or simply return False to discard this configuration.
        return False
    # Additional dependency checks:
    if config.get("quantization_config", {}).get("quantization", False):
        # If quantization is enabled, maybe fp_precision is irrelevant:
        config["fp_precision"] = "float16"  # or enforce a default.
    return True


def run_grid_search(base_config, grid_params, prompts, num_repeats=3, max_retries=3, retry_delay=5):
    """
    Run experiments for all configurations generated from the grid,
    repeating the overall cycle num_repeats times.
    
    Parameters:
      base_config (dict): Default configuration.
      grid_params (dict): Parameter grid.
      prompts (list): List of prompts to be used by the experiment.
      num_repeats (int): How many times to repeat the overall grid search.
      max_retries (int): Maximum retries per configuration.
      retry_delay (int): Delay (seconds) between retries.
    """
    # Generate all configurations (assuming you have a generate_configurations() defined elsewhere)
    config_list = generate_configurations(base_config, grid_params)
    if not config_list:
        print("No valid configurations generated!")
        return
    
    all_results = []
    
    for cycle in range(num_repeats):
        print(f"\n=== Grid Search Cycle {cycle+1}/{num_repeats} ===")
        # Shuffle to interleave order.
        random.shuffle(config_list)
        for config in config_list:
            print(f"\nRunning configuration: {config}")
            # Pass prompts to the ExperimentRunner.
            from experiments.experiment_runner import ExperimentRunner
            runner = ExperimentRunner(config, prompts)
            success, result = run_single_experiment_with_retries(runner, max_retries=max_retries, retry_delay=retry_delay)
            all_results.append({
                "config": config,
                "success": success,
                "result": result
            })
            time.sleep(2)
    
    return all_results
