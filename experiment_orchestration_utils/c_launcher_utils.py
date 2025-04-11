import subprocess
import os
import json
import logging
import tempfile
import time
import csv
import datetime

logger = logging.getLogger(__name__)

from configs.config_class import ExperimentConfig
from experiment_orchestration_utils.a_experiment_runner_class import ExperimentRunner 
from experiment_orchestration_utils.b_single_config_workflow import run_single_configuration

def get_config_file_path(config):
    if isinstance(config, dict):
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(config, tmp, indent=2)
        tmp.close()
        return tmp.name
    elif isinstance(config, str):
        return config
    else:
        raise ValueError("Configuration must be either a file path (str) or a dictionary.")

def log_failed_experiment(experiment_id, config, error_message, output_file="failed_experiments.csv"):
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = ["experiment_id", "timestamp", "suite", "config", "error_message"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "experiment_id": experiment_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "suite": config.get("suite", "unknown"),
            "config": json.dumps(config),
            "error_message": error_message
        })
    logger.info("Logged failed experiment %s to %s", experiment_id, output_file)

def run_from_config(config_data, prompts, max_retries=1, retry_delay=1):
    experiment_config = ExperimentConfig(**config_data)
    runner = ExperimentRunner(experiment_config, prompts)
    return run_single_configuration(runner, max_retries, retry_delay)

def run_from_file(config_path, prompts):
    with open(config_path, "r") as f:
        config_data = json.load(f)
    return run_from_config(config_data, prompts)

def launch_config_accelerate_cli(config, script_path: str, max_retries=3, extra_args=None) -> None:
    attempt = 1
    last_error = ""
    while attempt <= max_retries:
        logger.info("Launching experiment attempt %d", attempt)
        try:
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
            env["ACCELERATE_CONFIG_FILE"] = ""
            
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
            logger.info("Experiment run succeeded on attempt %d", attempt)
            return  # Success, exit the function.
    
        except subprocess.CalledProcessError as e:
            last_error = f"Attempt {attempt}: {str(e)}"
            logger.error("Experiment run failed on attempt %d: %s", attempt, last_error, exc_info=True)
            # Log this failed attempt immediately
            experiment_id = config.get("experiment_id", "unknown")
            log_failed_experiment(experiment_id, config, last_error, output_file="failed_attempts.csv")
            attempt += 1
            time.sleep(5)
    
    logger.error("Experiment run failed after %d attempts", max_retries)
    experiment_id = config.get("experiment_id", "unknown")
    log_failed_experiment(experiment_id, config, last_error)
    raise RuntimeError(f"Experiment run failed after {max_retries} attempts.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Launcher for a single configuration experiment using Accelerate with retries and failure logging."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration JSON file, or pass a dict stringified (if modified).")
    parser.add_argument("--script", type=str, required=True,
                        help="Path to the experiment script to launch (e.g., MAIN_a_single_experiment.py)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of attempts for the experiment run.")
    parser.add_argument("--retry_delay", type=int, default=5,
                        help="Delay (in seconds) between attempts.")
    parser.add_argument("--extra_args", nargs='*', default=[],
                        help="Any extra command-line arguments to pass to the experiment script.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(process)d] - %(asctime)s - %(levelname)s - %(message)s")
    launch_config_accelerate_cli(args.config, args.script, max_retries=args.max_retries, extra_args=args.extra_args)
