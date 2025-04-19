import time
import logging
import torch
import torch.distributed as dist
import sys

logger = logging.getLogger(__name__)

import os
import sys
import time
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def run_single_configuration(runner, max_retries=1, retry_delay=5):
    """
    Attempts to run a single experiment using runner.run_torch() up to max_retries times.
    Ensures that any partial distributed state is torn down on failure so we don't hang.
    """
    # 1) Set fragmentation avoidance early
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    for attempt in range(1, max_retries + 1):
        logger.info(f"Starting configuration run – attempt {attempt}/{max_retries}")
        # ensure we start from a clean slate each time
        try:
            runner.run_setup()
        except Exception as setup_err:
            logger.error(f"run_setup() failed: {setup_err}", exc_info=True)
            # nothing to teardown here, but clear any stray CUDA
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(retry_delay)
            continue

        # === STEP 1: initialize accelerator & get shared experiment_id ===
        try:
            runner.unique_id()
        except Exception as uid_err:
            logger.error(f"unique_id() failed: {uid_err}", exc_info=True)
            # teardown everything
            try: runner.teardown()
            except: pass
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(retry_delay)
            continue

        # === STEP 2: actually run the experiment ===
        try:
            result = runner.run_torch()
        except Exception as run_err:
            logger.error(f"run_torch() failed: {run_err}", exc_info=True)
            # teardown everything
            try: runner.teardown()
            except: pass
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(retry_delay)
            continue

        # === STEP 3: success path ===
        if runner.accelerator.is_main_process:
            # wait for all shards
            if dist.is_available() and dist.is_initialized():
                try: dist.barrier()
                except Exception as b: logger.warning(f"Pre-agg barrier failed: {b}")

            # aggregate & save
            runner.aggregate_results()
            runner.save_configuration_run_results_json()
            runner.save_configuration_run_results_tabular()

            # final teardown & sync
            try: runner.teardown()
            except: pass
            if dist.is_available() and dist.is_initialized():
                try: dist.barrier()
                except Exception as b: logger.warning(f"Post-teardown barrier failed: {b}")

            logger.info(f"Experiment #{runner.experiment_id} run succeeded.")
            return True, result

        else:
            # non‐main ranks just teardown & exit
            try: runner.teardown()
            except: pass
            sys.exit(0)

    # all retries exhausted
    logger.error("Experiment run failed after maximum attempts.")
    return False, None