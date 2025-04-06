import subprocess
import os
import json
import logging
import tempfile

logger = logging.getLogger(__name__)

from configs.config_class import ExperimentConfig
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner 
from experiment_orchestration_utils.b_single_config_workflow import run_single_configuration

def get_config_file_path(config):
    """
    If config is a dict, write it to a temporary JSON file and return its path.
    Otherwise, assume it's a file path and return it.
    """
    if isinstance(config, dict):
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(config, tmp, indent=2)
        tmp.close()
        return tmp.name
    elif isinstance(config, str):
        return config
    else:
        raise ValueError("Configuration must be either a file path (str) or a dictionary.")

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


def launch_config_accelerate_cli(config, script_path: str, extra_args=None) -> None:
    """
    Launch a single experiment configuration using Accelerate's CLI.
    Reads gpu_list and num_processes from the config, sets environment variables,
    and calls Accelerate CLI to launch the given script. Optionally, extra_args are
    appended to the command line.
    """
    config_file = get_config_file_path(config)
    with open(config_file, "r") as f:
        config_data = json.load(f)
    
    gpu_list = config_data.get("gpu_list", [])
    num_processes = config_data.get("num_processes", len(gpu_list))
    available = len(gpu_list)
    if num_processes > available:
        logger.warning("num_processes (%s) exceeds available GPUs (%s). Using available GPUs instead.", num_processes, available)
        num_processes = available

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    env["ACCELERATE_NUM_PROCESSES"] = str(num_processes)
    env["ACCELERATE_CONFIG_FILE"] = ""  # Disable any pre-existing Accelerate config

    cmd = [
        "accelerate",
        "launch",
        "--num_processes", str(num_processes),
        script_path,
        "--config", config_file
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    logger.info("Launching experiment with command: %s", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Launcher for a single configuration experiment using Accelerate."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration JSON file, or pass a dict stringified (if you modify usage).")
    parser.add_argument("--script", type=str, required=True,
                        help="Path to the experiment script to launch (e.g., MAIN_a_single_experiment.py)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(message)s")
    launch_config_accelerate_cli(args.config, args.script)