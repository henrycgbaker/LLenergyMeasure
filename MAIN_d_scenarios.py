#!/usr/bin/env python

from experiment_orchestration_utils.c_launcher_utils import launch_config_accelerate_cli
from configs.d_scenario_configs import scenario_config_list

import logging
from tqdm import tqdm

def main(scenario_config=None,
         models_list=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]):
    
    if models_list is None:
        print("No models provided. Defaulting to TinyLlama")
        
    # configure logging to display INFO-level messages
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    script_path = "MAIN_a_single_experiment.py"
    
    # Log the start of the process
    total_runs = len(models_list) * len(scenario_config_list)
    logging.info(f"Starting experiments for {len(models_list)} models "
                 f"with {len(scenario_config_list)} scenarios each (total runs: {total_runs}).")
    
    # Outer loop: iterate over each model
    for model_name in tqdm(models_list, desc="Models", unit="model"):
        logging.info(f"Processing model: {model_name}")
        # Inner loop: iterate over each scenario configuration
        for scenario_config in tqdm(scenario_config_list, desc=f"Scenarios for {model_name}", 
                                     unit="scenario", leave=False):
            # Inject the current model name into the scenario configuration
            scenario_config["model_name"] = model_name
            scenario_name = scenario_config.get("config_name", "<unnamed scenario>")
            logging.info(f"Launching scenario '{scenario_name}' with model '{model_name}'...")
            try:
                # Launch the experiment with the given config and extra arguments
                launch_config_accelerate_cli(scenario_config, script_path, extra_args=["--launched"])
                logging.info("✅ Completed config: %s", scenario_config["config_name"])

            except Exception as e:
                logging.error(f"❌ Error running model '{model_name}' on scenario '{scenario_name}': {e}")
                # Continue to the next scenario on exception
                continue
            
        logging.info(f"✅✅✅ Completed all scenarios for model: {model_name}")
    
    logging.info("🏁 All model configurations have been executed.")



if __name__ == "__main__":

    models_list = [
        #"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        #"meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.1-8B",
]    
    main(
        scenario_config=scenario_config_list,
        models_list=models_list
    )