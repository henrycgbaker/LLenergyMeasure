from codecarbon import EmissionsTracker
import logging
import torch

logger = logging.getLogger(__name__)

def warm_up(model, tokenizer, config, num_warmup_runs=3):
    """
    Run a number of dummy forward passes to warm up the model,
    ensuring that GPUs, CPU caches, and any lazy initialization are ready.
    
    Parameters:
        model: The loaded model.
        tokenizer: The corresponding tokenizer.
        config: The experiment configuration object (which should have max_input_tokens, etc.).
        num_warmup_runs (int): The number of warm-up iterations.
    """
    dummy_prompt = "This is a warm-up run."
    for i in range(num_warmup_runs):
        # Tokenize the dummy prompt.
        dummy_input = tokenizer(dummy_prompt, return_tensors="pt", truncation=True, max_length=config.max_input_tokens)
        # Move the input to the model's device.
        dummy_input = {key: value.to(model.device) for key, value in dummy_input.items()}
        with torch.no_grad():
            _ = model(**dummy_input)
        print(f"Warm-up iteration {i+1} completed.")


def start_energy_tracking():
    tracker = EmissionsTracker(
        measure_power_secs=1, 
        allow_multiple_runs=True,
        tracking_mode="process", # to do - try this with "machine"?
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