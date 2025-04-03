import time
import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)

def run_single_experiment_with_retries(runner, max_retries=3, retry_delay=5):
    """
    Attempts to run a single experiment using runner.run_torch() up to max_retries times.
    """
    attempt = 0
    result = None
    while attempt < max_retries:
        try:
            logger.info(f"Starting experiment - run attempt {attempt+1}/{max_retries}")
            
            runner.run_setup()
            runner.unique_id()
            result = runner.run_torch()
            
            # If not the main process, do local cleanup and return.
            if not runner.accelerator.is_main_process:
                runner.teardown()
                return True, result

            # Safeguard barrier: wait for all processes to finish run_torch().
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            except Exception as barrier_err:
                logger.warning(f"Pre-aggregation barrier failed: {barrier_err}")
            
            # Main process: aggregate results and save.
            runner.aggregate_results()
            runner.save_experiment_results()
            runner.teardown()  # Call with parentheses.

            # Final barrier to ensure all processes have cleaned up.
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            except Exception as barrier_err:
                logger.warning(f"Post-teardown barrier failed: {barrier_err}")
            
            logger.info(f"Experiment #{runner.experiment_id} run succeeded.")
            return True, result
        except Exception as e:
            attempt += 1
            logger.error(f"Experiment #{runner.experiment_id} run failed on attempt {attempt}: {e}", exc_info=True)
            time.sleep(retry_delay)
    logger.error(f"Experiment #{runner.experiment_id} run failed after maximum attempts.")
    return False, None

