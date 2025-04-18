import time
import logging
import torch.distributed as dist
import sys

logger = logging.getLogger(__name__)

def run_single_configuration(runner, max_retries=3, retry_delay=5):
    """
    Attempts to run a single experiment using runner.run_torch() up to max_retries times.
    Only the main process returns a success/failure result. Non-main processes simply teardown and exit.
    """
    attempt = 0
    result = None
    while attempt < max_retries:
        try:
            logger.info(f"Starting configuration run - run attempt {attempt+1}/{max_retries}")
            runner.run_setup()
            runner.unique_id()
            result = runner.run_torch()
            
            if runner.accelerator.is_main_process:
                # Main process: wait for all processes to finish run_torch()
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                except Exception as barrier_err:
                    logger.warning(f"Pre-aggregation barrier failed: {barrier_err}")
                
                # Main process: aggregate results and save.
                runner.aggregate_results()
                runner.save_configuration_run_results_json()
                runner.save_configuration_run_results_tabular()
                runner.teardown()  # Teardown for the main process
                
                try:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                except Exception as barrier_err:
                    logger.warning(f"Post-teardown barrier failed: {barrier_err}")
                
                logger.info(f"Experiment #{runner.experiment_id} run succeeded.")
                return True, result
            else:
                # Non-main processes: perform teardown and exit quietly.
                runner.teardown()
                sys.exit(0)
                
        except Exception as e:
            attempt += 1
            logger.error(f"Experiment run failed on attempt {attempt}: {e}", exc_info=True)
            time.sleep(retry_delay)
    
    logger.error(f"Experiment run failed after maximum attempts.")
    return False, None
