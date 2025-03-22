from helper_functions._9_get_compute_info import get_flops, get_memory, get_gpu_cpu_utilisation
import os

def combine_energy_metrics(codecarbon_data, accelerator):
    """
    per process energy metrics
    """
    energy_kwh = getattr(codecarbon_data, "energy_consumed", 0)
    energy_joules = energy_kwh * 3.6e6  # 1 kWh = 3.6e6 joules
        
    process_id = os.getpid()
    local_process_index = accelerator.local_process_index

    return {
        "process_id": process_id,
        "local_process_index": local_process_index,
        "energy_results": {
            "cpu_power": getattr(codecarbon_data, "cpu_power", None),
            "gpu_power": getattr(codecarbon_data, "gpu_power", None),
            "ram_power": getattr(codecarbon_data, "ram_power", None),
            "cpu_energy": getattr(codecarbon_data, "cpu_energy", None),
            "gpu_energy": getattr(codecarbon_data, "gpu_energy", None),
            "ram_energy": getattr(codecarbon_data, "ram_energy", None),
            "total_energy_kwh": energy_kwh,
            "total_energy_joules": energy_joules,
            "final_emissions": getattr(codecarbon_data, "emissions", None),
        }
    }
