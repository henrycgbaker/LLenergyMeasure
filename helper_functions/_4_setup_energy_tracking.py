from codecarbon import EmissionsTracker
import logging
logger = logging.getLogger(__name__)

def start_energy_tracking():
    tracker = EmissionsTracker(
        measure_power_secs=1, 
        allow_multiple_runs=True,
        tracking_mode="process", # try this with "machine"
        log_level=logging.ERROR
    )
    tracker.start()
    return tracker

def stop_energy_tracking(tracker):
    try:
        tracker.stop()
        return tracker._prepare_emissions_data()
    except AttributeError as e:
        logger.error(f"Failed to get emissions data: {e}")
        return {}