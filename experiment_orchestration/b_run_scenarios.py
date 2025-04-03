import random, logging
from experiment_orchestration.experiment_runner import ExperimentRunner 
from experiment_orchestration.a_run_single_experiment import run_single_experiment_with_retries
from configs.experiment_config_class import ExperimentConfig


def run_scenarios(scenario_list, prompts, num_repeats=3, max_retries=3, retry_delay=5):
    """
    Run experiments for each scenario configuration in scenario_list,
    repeating the overall cycle num_repeats times.
    
    Parameters:
      scenario_list (list): List of scenario configurations (dicts).
      prompts (list): List of prompts to be used by the experiment.
      num_repeats (int): How many times to repeat the overall experiment.
      max_retries (int): Maximum retries per configuration.
      retry_delay (int): Delay (seconds) between retries.
    """
    scenario_results = []
    for cycle in range(num_repeats):
        logging.info(f"=== Scenarios Run Cycle {cycle+1}/{num_repeats} ===")
        random.shuffle(scenario_list)
        for scenario in scenario_list:
            # Instantiate an ExperimentRunner from the scenario dict
            runner = ExperimentRunner(ExperimentConfig.from_dict(scenario), prompts)
            success, result = run_single_experiment_with_retries(
                runner, max_retries=max_retries, retry_delay=retry_delay
            )
            scenario_results.append({
                "config": scenario,
                "success": success,
            })
    return scenario_results