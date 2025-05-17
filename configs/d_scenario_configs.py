import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.a_default_config import base_config
from configs.config_utils import update_nested_dict, update_multiple_config, generate_config_name_from_variation

# ----------------------------
# Scenario Configurations
# ----------------------------

scenario_config_list = []
"""
# IDEAL BASED ON INITAL FINDINGS FROM ANALYSIS
updates_a0 = {
    "config_name": "A0_platonic_ideal",
    "batching_options.batch_size___fixed_batching": 64,
    "fp_precision": "float16",
    "quantization_config.quantization": True, 
    "quantization_config.load_in_4bit": True, 
    "decoder_config.decoding_mode": "greedy",
    "decoder_config.decoder_temperature": 0,
    "latency_simulation.simulate": False,
}
scenario_a0 = update_multiple_config(base_config, updates_a0)
scenario_a0["gpu_list"] = [0, 1, 2, 3]
scenario_a0["num_processes"] = 1
scenario_a0["scenario_info"] = {"name": "A0_platonic_ideal", "realistic": False}
scenario_a0["suite"] = "scenarios" 
scenario_config_list.append(scenario_a0)

# WORST BASED ON INITAL FINDINGS FROM ANALYSIS

updates_r0 = {
    "config_name": "R7_anti_platonic_ideal",
    "batching_options.batch_size___fixed_batching": 1,
    "fp_precision": "float32",
    "quantization_config.quantization": False,
    "decoder_config.decoding_mode": "greedy",
    "decoder_config.decoder_temperature": 0.8,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.4,
    "latency_simulation.delay_max": 0.5,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0, 
    "latency_simulation.burst_size": 8, 
}
scenario_r0 = update_multiple_config(base_config, updates_r0) # include this because empirically int8 is worst (although theoretically it shouldn't be)
scenario_r0["gpu_list"] = [0, 1, 2, 3]
scenario_r0["num_processes"] = 4
scenario_r0["scenario_info"] = {"name": "R7_anti_platonic_ideal", "realistic": True}
scenario_r0["suite"] = "scenarios" 
scenario_config_list.append(scenario_r0)

# ------------------------------------------------------------------------------------------
"""
# A1: 
updates_a1 = {
    "config_name": "A1_ideal",
    "batching_options.batch_size___fixed_batching": 128,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": False,
}
scenario_a1 = update_multiple_config(base_config, updates_a1)
scenario_a1["gpu_list"] = [0, 1, 2, 3]
scenario_a1["num_processes"] = 1
scenario_a1["scenario_info"] = {"name": "A1_Single_GPU", "realistic": False}
scenario_a1["suite"] = "scenarios" 
scenario_config_list.append(scenario_a1)

"""
# A2: Precision Minimalist
updates_a2 = {
    "config_name": "A2_2_GPU",
    "batching_options.batch_size___fixed_batching": 64,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": True,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": False,
}
scenario_a2 = update_multiple_config(base_config, updates_a2)
scenario_a2["gpu_list"] = [0, 1]
scenario_a2["num_processes"] = 2
scenario_a2["scenario_info"] = {"name": "A2_Precision_Focus", "realistic": False}
scenario_a2["suite"] = "scenarios" 
scenario_config_list.append(scenario_a2)

# A3: Quantisation 
updates_a3 = {
    "config_name": "A3_Quantisation_Focus",
    "batching_options.batch_size___fixed_batching": 64,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": True,
    "decoder_config.decoding_mode": "top_k",
    "decoder_config.decoder_top_k": 50,
    "latency_simulation.simulate": False,
}
scenario_a3 = update_multiple_config(base_config, updates_a3)
scenario_a3["gpu_list"] = [0]
scenario_a3["num_processes"] = 2
scenario_a3["scenario_info"] = {"name": "A3_Quantisation_Focus", "realistic": False}
scenario_a3["suite"] = "scenarios" 
scenario_config_list.append(scenario_a3)

# A5: Parallel 
updates_a4 = {
    "config_name": "A4_Parallel_Exploit",
    "batching_options.batch_size___fixed_batching": 256,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": False,
}
scenario_a5 = update_multiple_config(base_config, updates_a4)
scenario_a5["gpu_list"] = [0, 1, 2, 3]
scenario_a5["num_processes"] = 4
scenario_a5["scenario_info"] = {"name": "A4_Parallel_Exploit", "realistic": False}
scenario_a5["suite"] = "scenarios" 
scenario_config_list.append(scenario_a5)
"""

# Realistic Scenarios
# 1) Ultra-low latency, single-GPU, greedy, no quant
updates_r1 = {
    "config_name": "R1_small_batch_single_gpu_low_latency",
    "batching_options.batch_size___fixed_batching": 4,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "decoder_config.decoder_temperature": 0.0,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.01,
    "latency_simulation.delay_max": 0.1,
    "latency_simulation.simulate_burst": False,
}
scenario_r1 = update_multiple_config(base_config, updates_r1)
scenario_r1.update({
    "gpu_list": [0],
    "num_processes": 1,
    "scenario_info": {"name": updates_r1["config_name"], "realistic": True},
    "suite": "scenarios",
})
scenario_config_list.append(scenario_r1)

# 2) Ultra-low latency, dual-GPU, sampling (top_p=0.9), no quant
updates_r2 = {
    "config_name": "R2_DualGPU_TopP_low_latency",
    "batching_options.batch_size___fixed_batching": 4,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "top_p",
    "decoder_config.decoder_top_p": 0.9,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.01,
    "latency_simulation.delay_max": 0.1,
    "latency_simulation.simulate_burst": False,
}
scenario_r2 = update_multiple_config(base_config, updates_r2)
scenario_r2.update({
    "gpu_list": [0, 1],
    "num_processes": 2,
    "scenario_info": {"name": updates_r2["config_name"], "realistic": True},
    "suite": "scenarios",
})
scenario_config_list.append(scenario_r2)

# 3) Standard latency, single-GPU, greedy, 4-bit quant
updates_r3 = {
    "config_name": "R3_Standard_SingleGPU_Greedy_4bit",
    "batching_options.batch_size___fixed_batching": 8,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_4bit": True,
    "decoder_config.decoding_mode": "greedy",
    "decoder_config.decoder_temperature": 0.0,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.5,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 3.0,
    "latency_simulation.burst_size": 6,
}
scenario_r3 = update_multiple_config(base_config, updates_r3)
scenario_r3.update({
    "gpu_list": [0],
    "num_processes": 1,
    "scenario_info": {"name": updates_r3["config_name"], "realistic": True},
    "suite": "scenarios",
})
scenario_config_list.append(scenario_r3)

# 4) Standard latency, dual-GPU, sampling (top_p=0.9), 4-bit quant
updates_r4 = {
    "config_name": "R4_Standard_DualGPU_TopP_4bit",
    "batching_options.batch_size___fixed_batching": 8,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_4bit": True,
    "decoder_config.decoding_mode": "top_p",
    "decoder_config.decoder_top_p": 0.9,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.5,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 3.0,
    "latency_simulation.burst_size": 6,
}
scenario_r4 = update_multiple_config(base_config, updates_r4)
scenario_r4.update({
    "gpu_list": [0, 1],
    "num_processes": 2,
    "scenario_info": {"name": updates_r4["config_name"], "realistic": True},
    "suite": "scenarios",
})
scenario_config_list.append(scenario_r4)

# 5) Tail-end latency, single-GPU, greedy, no quant
updates_r5 = {
    "config_name": "R5_Tail_SingleGPU_Greedy",
    "batching_options.batch_size___fixed_batching": 8,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "decoder_config.decoder_temperature": 0.0,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.3,
    "latency_simulation.delay_max": 0.8,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0,
    "latency_simulation.burst_size": 5,
}
scenario_r5 = update_multiple_config(base_config, updates_r5)
scenario_r5.update({
    "gpu_list": [0],
    "num_processes": 1,
    "scenario_info": {"name": updates_r5["config_name"], "realistic": True},
    "suite": "scenarios",
})
scenario_config_list.append(scenario_r5)

# 6) Tail-end latency, dual-GPU, sampling (top_p=0.9), no quant
updates_r6 = {
    "config_name": "R6_Tail_DualGPU_TopP",
    "batching_options.batch_size___fixed_batching": 8,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "top_p",
    "decoder_config.decoder_top_p": 0.9,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.3,
    "latency_simulation.delay_max": 0.8,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0,
    "latency_simulation.burst_size": 5,
}
scenario_r6 = update_multiple_config(base_config, updates_r6)
scenario_r6.update({
    "gpu_list": [0, 1],
    "num_processes": 2,
    "scenario_info": {"name": updates_r6["config_name"], "realistic": True},
    "suite": "scenarios",
})
scenario_config_list.append(scenario_r6)


# -----------------------------------------
# Combine All Scenarios
# -----------------------------------------

__all__ = ["scenario_config_list"]

# ----------------------------
# Scenario Configuration Validation
# ----------------------------
def validate_scenario_config(cfg):
    # Ensure required top-level keys are present
    required_keys = [
        "config_name", "suite", "scenario_info",
        "gpu_list", "num_processes", "batching_options",
        "decoder_config", "quantization_config", "latency_simulation", "fp_precision"
    ]
    for key in required_keys:
        assert key in cfg, f"Missing required key '{key}' in scenario config: {cfg}"
    
    # Validate suite is 'scenarios'
    assert cfg["suite"] == "scenarios", f"Suite must be 'scenarios', got {cfg['suite']}"
    
    # Validate scenario_info structure
    scenario_info = cfg["scenario_info"]
    assert isinstance(scenario_info, dict), f"'scenario_info' must be a dict, got {type(scenario_info)}"
    for info_key in ["name", "realistic"]:
        assert info_key in scenario_info, f"Missing '{info_key}' in scenario_info: {scenario_info}"
    
    # Validate batching_options exists 
    assert isinstance(cfg["batching_options"], dict), "'batching_options' must be a dictionary"
    
    # Validate decoder_config
    decoder_cfg = cfg["decoder_config"]
    for d_key in ["decoding_mode", "decoder_temperature", "decoder_top_k", "decoder_top_p"]:
        assert d_key in decoder_cfg, f"Missing '{d_key}' in decoder_config"
    if decoder_cfg["decoding_mode"] != None:
        valid_modes = ["greedy", "top_k", "top_p"]
        mode = decoder_cfg["decoding_mode"]
        assert mode in valid_modes, f"Invalid 'decoder_config.decoding_mode': {mode}"
        if mode == "top_k":
            assert decoder_cfg["decoder_top_k"] != None, "For top_k sampling, decoder_top_k must be set"
        if mode == "top_p":
            assert decoder_cfg["decoder_top_p"] != None, "For top_p sampling, decoder_top_p must be set"
    
    # Validate quantization_config
    quant_cfg = cfg["quantization_config"]
    for q_key in ["quantization", "load_in_8bit", "load_in_4bit"]:
        assert q_key in quant_cfg, f"Missing '{q_key}' in quantization_config"
    
    # Validate latency_simulation
    latency_cfg = cfg["latency_simulation"]
    for l_key in ["simulate", "delay_min", "delay_max", "simulate_burst", "burst_interval", "burst_size"]:
        assert l_key in latency_cfg, f"Missing '{l_key}' in latency_simulation"
    if latency_cfg["simulate"] != None and latency_cfg["simulate"] is True:
        # When simulation is active, delay_min and delay_max should be meaningful
        assert latency_cfg["delay_min"] != None, "latency_simulation.delay_min must be set if simulation is enabled"
        assert latency_cfg["delay_max"] != None, "latency_simulation.delay_max must be set if simulation is enabled"
    
    return True

__all__ = ["scenario_config_list"]

# ----------------------------
# Validation and (Optional) File Output
# ----------------------------
if __name__ == "__main__":
    # Validate each scenario configuration
    for i, cfg in enumerate(scenario_config_list):
        try:
            validate_scenario_config(cfg)
        except AssertionError as e:
            print(f"❌ Scenario Config {i} ({cfg.get('scenario_info', {}).get('name', 'NO_NAME')}) failed validation: {e}")
            raise
        else:
            print(f"✅ Scenario Config {i} ({cfg['scenario_info']['name']}) passed validation.")
