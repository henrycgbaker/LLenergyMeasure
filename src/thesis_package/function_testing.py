import unittest
from codecarbon import EmissionsTracker


#ALSO USE ASSERT STATEMENTS
class TestEmissionsTracker(unittest.TestCase):
    def test_energy_tracking(self):
        tracker = EmissionsTracker(measure_power_secs=1, allow_multiple_runs=True, tracking_mode="machine")
        tracker.start()
        import time
        time.sleep(2)  # Simulate workload
        tracker.stop()

        emissions_data = tracker.final_emissions_data
        self.assertIsNotNone(emissions_data)
        self.assertGreater(emissions_data.energy_consumed, 0)  # Ensure energy was consumed
        print("Test Passed! Emissions Data:", emissions_data)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)  # <-- Fix for Jupyter
