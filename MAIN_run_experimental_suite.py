#!/usr/bin/env python
"""
run_experimental_suite.py

Orchestrates large‑scale experimental suites.
Persists cycle numbering across runs via progress_trackers/cycle_id.txt
"""

import os
import random
import logging
import time
import json
from tqdm import tqdm

from experiment_orchestration_utils.c_launcher_utils import launch_config_accelerate_cli

from configs.c_controlled_configs import controlled_config_list
from configs.d_scenario_configs import scenario_config_list
from configs.e_grid_configs import grid_config_list

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(process)d] - %(asctime)s - %(levelname)s - %(message)s"
)

CYCLES_OF_FULL_SUITE = 10

SINGLE_EXP_SCRIPT = os.path.abspath("MAIN_a_single_experiment.py")
PERSISTENT_TRACKER_DIR = "persistent_progress_trackers"
PROGRESS_FILE = os.path.join(PERSISTENT_TRACKER_DIR,"configs_run_progress.json")
CYCLE_ID_FILE = os.path.join(PERSISTENT_TRACKER_DIR, "cycle_id.txt")


def load_progress():
    """
    Returns a dict mapping run_id -> experiment_id (or None).
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(done_map):
    """
    Atomically dump the entire progress map once per configuration run.
    """
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(done_map, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def load_cycle_id(default=1):
    os.makedirs(PERSISTENT_TRACKER_DIR, exist_ok=True)
    if os.path.exists(CYCLE_ID_FILE):
        try:
            with open(CYCLE_ID_FILE, 'r') as f:
                return int(f.read().strip())
        except Exception:
            logging.warning("Could not read cycle tracker; starting at %s", default)
    # Initialize tracker file with default
    save_cycle_id(default)
    return default


def save_cycle_id(next_cycle):
    """
    Persist the next cycle_id to tracker file.
    Ensures tracker directory exists.
    """
    os.makedirs(PERSISTENT_TRACKER_DIR, exist_ok=True)
    with open(CYCLE_ID_FILE, 'w') as f:
        f.write(str(next_cycle))


def run_cycle(config_list, suite_name, cycle_num, done_map, model_name):
    logging.info("Starting Cycle %s for suite '%s' (model=%s)", cycle_num, suite_name, model_name)

    randomized = config_list.copy()
    random.shuffle(randomized)

    for cfg in tqdm(randomized, desc=f"{suite_name} cycle {cycle_num}", unit="config"):
        # embed cycle id into config
        cfg['cycle_id'] = cycle_num

        name = cfg.get("config_name", "unnamed")
        run_id = f"{model_name}::{suite_name}::{name}::cycle_{cycle_num}"

        if run_id in done_map:
            tqdm.write(f"[SKIP] {run_id} already done")
            continue

        logging.info("Cycle %s: Launching %s", cycle_num, run_id)
        try:
            result = launch_config_accelerate_cli(
                cfg, SINGLE_EXP_SCRIPT, extra_args=["--launched"]
            )
            exp_id = None
            if isinstance(result, dict):
                exp_id = result.get("experiment_id")

            logging.info(
                "Cycle %s: Completed %s (experiment_id=%s)",
                cycle_num, run_id, exp_id
            )

            # record it
            done_map[run_id] = exp_id
            save_progress(done_map)
            logging.info("Saved progress for %s (cycle %s)", run_id, cycle_num)

        except Exception as e:
            logging.error("Cycle %s: Run FAILED for %s: %s", cycle_num, run_id, e)

        time.sleep(2)


def main():
    models_list = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        #"meta-llama/Llama-3.2-1B",
        #"meta-llama/Llama-3.2-3B",
        #"meta-llama/Llama-3.1-8B",
    ]
    
    suites = [
        ("Controlled", controlled_config_list),
        #("Scenario", scenario_config_list),
        #("GridSearch", grid_config_list),
    ]

    done_map = load_progress()
    
    start_cycle = load_cycle_id(default=1)
    end_cycle = start_cycle + CYCLES_OF_FULL_SUITE

    for model in models_list:
        logging.info("=== Model: %s ===", model)

        for cycle in range(start_cycle, end_cycle):
            for suite_name, original_list in tqdm(suites, desc=f"Cycle {cycle}", unit="suite"):
                # expand per‑model
                expanded = []
                for cfg in original_list:
                    new = cfg.copy()
                    new["model_name"] = model
                    expanded.append(new)

                run_cycle(expanded, suite_name, cycle, done_map, model)

            # after completing this cycle for all suites, bump and save
            next_cycle = cycle + 1
            save_cycle_id(next_cycle)
            logging.info("Persisted next cycle as %s", next_cycle)

    logging.info("All experimental suites have been executed.")


if __name__ == "__main__":
    main()
