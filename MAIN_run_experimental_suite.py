#!/usr/bin/env python
"""
run_experimental_suite.py

Orchestrates large‑scale experimental suites.
"""

import os
import random
import logging
import time
import json
from tqdm import tqdm

from experiment_orchestration_utils.c_launcher_utils import launch_config_accelerate_cli

from configs.b_models_config import huggingface_models
from configs.c_controlled_configs import controlled_config_list
from configs.d_scenario_configs import scenario_config_list
from configs.e_grid_configs import grid_config_list

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(process)d] - %(asctime)s - %(levelname)s - %(message)s"
)

CYCLES_OF_FULL_SUITE = 3
SINGLE_EXP_SCRIPT = os.path.abspath("MAIN_a_single_experiment.py")
PROGRESS_FILE = "run_progress.json"


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


def run_cycle(config_list, suite_name, cycle_num, done_map, model_name):
    logging.info("Starting Cycle %s for suite '%s' (model=%s)", cycle_num, suite_name, model_name)

    randomized = config_list.copy()
    random.shuffle(randomized)

    for cfg in tqdm(randomized, desc=f"{suite_name} cycle {cycle_num}", unit="config"):
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


def run_suite(config_list, suite_name, done_map, model_name):
    logging.info("Starting Experimental Suite: '%s' for model=%s", suite_name, model_name)
    for cycle in range(1, CYCLES_OF_FULL_SUITE + 1):
        run_cycle(config_list, suite_name, cycle, done_map, model_name)
    logging.info("Completed Experimental Suite: '%s'", suite_name)


def main():
    done_map = load_progress()

    suites = [
        ("Scenario", scenario_config_list),
        ("Controlled", controlled_config_list),
        # ("GridSearch", grid_config_list),
    ]

    for model in huggingface_models:
        logging.info("=== Model: %s ===", model)

        for suite_name, original_list in tqdm(suites, desc="Suites", unit="suite"):
            # expand per‑model
            expanded = []
            for cfg in original_list:
                new = cfg.copy()
                new["model_name"] = model
                # cached_flops logic …
                expanded.append(new)

            run_suite(expanded, suite_name, done_map, model)

    logging.info("All experimental suites have been executed.")


if __name__ == "__main__":
    main()