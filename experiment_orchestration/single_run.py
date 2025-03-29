import time
import logging

logger = logging.getLogger(__name__)

def run_single_experiment_with_retries(runner, max_retries=3, retry_delay=5):
    """
    Attempts to run a single experiment (i.e. one configuration) using runner.run_torch() 
    up to max_retries times.
    """
    attempt = 0
    result = None
    while attempt < max_retries:
        try:
            logger.info(f"Starting experiment run attempt {attempt+1}/{max_retries}")
            result = runner.run_torch()
            # If this is not the main process, return immediately.
            if not runner.accelerator.is_main_process:
                return True, result
            # Only main process continues aggregation and saving.
            runner.aggregate_results()
            runner.save_experiment_results()
            runner.teardown()
            logger.info("Experiment run succeeded.")
            return True, result
        except Exception as e:
            attempt += 1
            logger.error(f"Experiment run failed on attempt {attempt}: {e}", exc_info=True)
            time.sleep(retry_delay)
    logger.error("Experiment run failed after maximum attempts.")
    return False, None
