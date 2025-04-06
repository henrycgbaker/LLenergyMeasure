import json
from configs.experiment_config_class import ExperimentConfig
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner 
from experiment_orchestration_utils.b_run_single_configuration import run_single_configuration

def run_from_config(config_data, prompts, max_retries=3, retry_delay=5):
    """
    Given a configuration (as a dict) and a list of prompts,
    create an ExperimentRunner and run a single configuration.
    Returns (success, result).
    """
    # Convert the configuration dict to the dataclass instance.
    experiment_config = ExperimentConfig(**config_data)
    
    # Create the runner with the loaded prompts.
    runner = ExperimentRunner(experiment_config, prompts)
    
    # Run the experiment using the common function.
    return run_single_configuration(runner, max_retries, retry_delay)

def run_from_file(config_path, prompts):
    """
    Convenience function: load the config JSON from file and run the experiment.
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    return run_from_config(config_data, prompts)
