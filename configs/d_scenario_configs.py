# scenario_configs.py
from configs.a_default_config import base_config
from configs.config_utils import update_nested_dict, update_multiple_config, generate_config_name_from_variation

# -----------------------------------------
# Artificial (Non-realistic) Scenarios
# -----------------------------------------

# A1: Max Throughput Exploit
#  - Batch size 256, FP16, INT8 quantisation, Greedy, No latency, 4 GPUs.
updates_a1 = {
    "config_name": "A1_Max_Throughput_Exploit",
    "batching_options.batch_size___fixed_batching": 256,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",  # we add a key to indicate decoding mode
    "latency_simulation.simulate": False,
}
scenario_a1 = update_multiple_config(base_config, updates_a1)
scenario_a1["gpu_list"] = [0, 1, 2, 3]  # ensure 4 GPUs
scenario_a1["num_processes"] = 4
scenario_a1["scenario_info"] = {
    "name": "A1_Max_Throughput_Exploit",
    "realistic": False,
}
scenario_a1["suite"] = "scenarios" 

# A2: Precision Minimalist
#  - Batch size 128, FP16, INT4 quantisation, Greedy, No latency, 2 GPUs.
updates_a2 = {
    "config_name": "A2_Precision_Minimalist",
    "batching_options.batch_size___fixed_batching": 128,
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
scenario_a2["scenario_info"] = {
    "name": "A2_Precision_Minimalist",
    "realistic": False,
}
scenario_a2["suite"] = "scenarios" 

# A3: Quantisation Gaming
#  - Batch size 64, FP16, INT4 quantisation, Top-k sampling (k=50), No latency, 1 GPU.
updates_a3 = {
    "config_name": "A3_Quantisation_Gaming",
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
scenario_a3["num_processes"] = 1
scenario_a3["scenario_info"] = {
    "name": "A3_Quantisation_Gaming",
    "realistic": False,
}
scenario_a3["suite"] = "scenarios" 

# A4: Latency Ignorance Exploit
#  - Batch size 32, FP16, INT8 quantisation, Greedy, No latency, 1 GPU.
updates_a4 = {
    "config_name: A4_Latency_Ignorance_Exploit"
    "batching_options.batch_size___fixed_batching": 32,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": False,
}
scenario_a4 = update_multiple_config(base_config, updates_a4)
scenario_a4["gpu_list"] = [0]
scenario_a4["num_processes"] = 1
scenario_a4["scenario_info"] = {
    "name": "A4_Latency_Ignorance_Exploit",
    "realistic": False,
}
scenario_a4["suite"] = "scenarios" 

# A5: Parallel Overdrive
#  - Batch size 64, FP16, no quantisation, Greedy, No latency, 4 GPUs.
updates_a5 = {
    "config_name": "A5_Parallel_Overdrive",
    "batching_options.batch_size___fixed_batching": 64,
    "fp_precision": "float16",
    "quantization_config.quantization": False,   # no quantisation
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": False,
}
scenario_a5 = update_multiple_config(base_config, updates_a5)
scenario_a5["gpu_list"] = [0, 1, 2, 3]
scenario_a5["num_processes"] = 4
scenario_a5["scenario_info"] = {
    "name": "A5_Parallel_Overdrive",
    "realistic": False,
}
scenario_a5["suite"] = "scenarios" 

# -----------------------------------------
# Realistic Scenarios
# -----------------------------------------

# R1: Standard Production Config
#  - Batch size 16, FP32, no quantisation, Greedy,
#    Latency: simulate True, delay_min=0.5, delay_max=1.5, simulate_burst True, burst_interval=4.0, burst_size=5,
#    2 GPUs.
updates_r1 = {
    "config_name": "R1_Standard_Production_Config",
    "batching_options.batch_size___fixed_batching": 16,
    "fp_precision": "float32",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.5,
    "latency_simulation.delay_max": 1.5,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0,
    "latency_simulation.burst_size": 5,
}
scenario_r1 = update_multiple_config(base_config, updates_r1)
scenario_r1["gpu_list"] = [0, 1]
scenario_r1["num_processes"] = 2
scenario_r1["scenario_info"] = {
    "name": "R1_Standard_Production_Config",
    "realistic": True,
}
scenario_r1["suite"] = "scenarios" 

# R2: Low-Latency Chatbot Deployment
#  - Batch size 4, FP32, no quantisation, Top-p sampling (p=0.9),
#    Latency: simulate True, delay_min=0.01, delay_max=0.05, simulate_burst False,
#    1 GPU.
updates_r2 = {
    "config_name": "R2_Low_Latency_Chatbot_Deployment",
    "batching_options.batch_size___fixed_batching": 4,
    "fp_precision": "float32",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "top_p",
    "decoder_config.decoder_top_p": 0.9,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.01,
    "latency_simulation.delay_max": 0.05,
    "latency_simulation.simulate_burst": False,
}
scenario_r2 = update_multiple_config(base_config, updates_r2)
scenario_r2["gpu_list"] = [0]
scenario_r2["num_processes"] = 1
scenario_r2["scenario_info"] = {
    "name": "R2_Low_Latency_Chatbot_Deployment",
    "realistic": True,
}
scenario_r2["suite"] = "scenarios" 

# R3: Balanced Enterprise Service
#  - Batch size 32, FP16, INT8 quantisation, Top-k sampling (k=50),
#    Latency: simulate True, delay_min=0.5, delay_max=1.5, simulate_burst True, burst_interval=4.0, burst_size=5,
#    2 GPUs.
updates_r3 = {
    "config_name": "R3_Balanced_Enterprise_Service",
    "batching_options.batch_size___fixed_batching": 32,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "top_k",
    "decoder_config.decoder_top_k": 50,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.5,
    "latency_simulation.delay_max": 1.5,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 4.0,
    "latency_simulation.burst_size": 5,
}
scenario_r3 = update_multiple_config(base_config, updates_r3)
scenario_r3["gpu_list"] = [0, 1]
scenario_r3["num_processes"] = 2
scenario_r3["scenario_info"] = {
    "name": "R3_Balanced_Enterprise_Service",
    "realistic": True,
}
scenario_r3["suite"] = "scenarios" 

# R4: High-Load Cloud API Deployment
#  - Batch size 8, FP16, no quantisation, Greedy,
#    Latency: simulate True, delay_min=0.05, delay_max=0.2, simulate_burst True, burst_interval=2.0, burst_size=5,
#    1 GPU.
updates_r4 = {
    "config_name": "R4_High_Load_Cloud_API_Deployment",
    "batching_options.batch_size___fixed_batching": 8,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.05,
    "latency_simulation.delay_max": 0.2,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 2.0,
    "latency_simulation.burst_size": 5,
}
scenario_r4 = update_multiple_config(base_config, updates_r4)
scenario_r4["gpu_list"] = [0]
scenario_r4["num_processes"] = 1
scenario_r4["scenario_info"] = {
    "name": "R4_High_Load_Cloud_API_Deployment",
    "realistic": True,
}
scenario_r4["suite"] = "scenarios" 

# R5: Real-Time Mobile Inference
#  - Batch size 1, FP16, INT8 quantisation, Top-p sampling (p=0.9),
#    Latency: simulate True, delay_min=0.2, delay_max=0.6, simulate_burst True, burst_interval=5.0, burst_size=8,
#    1 GPU.
updates_r5 = {
    "config_name": "R5_Real_Time_Mobile_Inference",
    "batching_options.batch_size___fixed_batching": 1,
    "fp_precision": "float16",
    "quantization_config.quantization": True,
    "quantization_config.load_in_8bit": True,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "top_p",
    "decoder_config.decoder_top_p": 0.9,
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.2,
    "latency_simulation.delay_max": 0.6,
    "latency_simulation.simulate_burst": True,
    "latency_simulation.burst_interval": 5.0,
    "latency_simulation.burst_size": 8,
}
scenario_r5 = update_multiple_config(base_config, updates_r5)
scenario_r5["gpu_list"] = [0]
scenario_r5["num_processes"] = 1
scenario_r5["scenario_info"] = {
    "name": "R5_Real_Time_Mobile_Inference",
    "realistic": True,
}
scenario_r5["suite"] = "scenarios" 

# R6: Medium-Scale Language Model Serving
#  - Batch size 32, FP16, no quantisation, Greedy,
#    Latency: simulate True, delay_min=0.01, delay_max=0.1, simulate_burst False,
#    4 GPUs.
updates_r6 = {
    "config_name": "R6_Medium_Scale_Language_Model_Serving",
    "batching_options.batch_size___fixed_batching": 32,
    "fp_precision": "float16",
    "quantization_config.quantization": False,
    "quantization_config.load_in_8bit": False,
    "quantization_config.load_in_4bit": False,
    "decoder_config.decoding_mode": "greedy",
    "latency_simulation.simulate": True,
    "latency_simulation.delay_min": 0.01,
    "latency_simulation.delay_max": 0.1,
    "latency_simulation.simulate_burst": False,
}
scenario_r6 = update_multiple_config(base_config, updates_r6)
scenario_r6["gpu_list"] = [0, 1, 2, 3]
scenario_r6["num_processes"] = 4
scenario_r6["scenario_info"] = {
    "name": "R6_Medium_Scale_Language_Model_Serving",
    "realistic": True,
}
scenario_r6["suite"] = "scenarios" 

# -----------------------------------------
# Combine All Scenarios
# -----------------------------------------

scenario_config_list = [
    scenario_a1, scenario_a2, scenario_a3, scenario_a4, scenario_a5,
    scenario_r1, scenario_r2, scenario_r3, scenario_r4, scenario_r5, scenario_r6
]

__all__ = ["scenario_config_list"]

if __name__ == "__main__":
    for cfg in scenario_config_list:
        print(cfg["scenario_info"])
