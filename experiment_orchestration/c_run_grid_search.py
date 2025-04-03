import itertools
import copy
import random
import time
import logging

from experiment_orchestration.a_run_single_experiment import run_single_experiment_with_retries

logger = logging.getLogger(__name__)

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
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        if validate_config(config):
            configs.append(config)
        else:
            logger.warning(f"Configuration rejected by validate_config: {config}")
    return configs


def validate_config(config: dict) -> bool:
    # Ensure total output tokens equal 100,000.
    total_output_tokens = config.get("max_output_tokens", 0) * config.get("num_input_prompts", 0)
    if total_output_tokens != 100000:
        return False

    # If quantization is enabled, reject if no cached FLOPs value is provided.
    quant_config = config.get("quantization_config", {})
    if quant_config.get("quantization", False):
        if quant_config.get("cached_flops_for_quantised_models") in (None, 0):
            # You might even log a warning here.
            return False
        # Optionally enforce that fp_precision is set appropriately.
        config["fp_precision"] = "float16"
        
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
    config_list = generate_configurations(base_config, grid_params)
    if not config_list:
        logger.error("No valid configurations generated!")
        return []
    
    all_results = []
    for cycle in range(num_repeats):
        logger.info(f"=== Grid Search Cycle {cycle+1}/{num_repeats} ===")
        random.shuffle(config_list)
        for config in config_list:
            logger.info(f"Running configuration: {config}")
            from experiment_orchestration.experiment_runner import ExperimentRunner
            from configs.experiment_config_class import ExperimentConfig
            runner = ExperimentRunner(ExperimentConfig.from_dict(config), prompts)
            success, result = run_single_experiment_with_retries(runner, max_retries=max_retries, retry_delay=retry_delay)
            all_results.append({
                "config": config,
                "success": success,
                "result": result
            })
    return all_results

