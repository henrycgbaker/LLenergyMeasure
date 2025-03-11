import logging
from codecarbon import EmissionsTracker

def start_energy_tracking():
    tracker = EmissionsTracker(
        measure_power_secs=1, 
        allow_multiple_runs=True,
        tracking_mode="machine",
        log_level=logging.ERROR
    )
    tracker.start()
    return tracker

def stop_energy_tracking(tracker):
    tracker.stop()
    return tracker.final_emissions_data
