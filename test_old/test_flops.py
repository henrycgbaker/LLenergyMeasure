import argparse
import torch
import logging
import json
from experiment_config_class import ExperimentConfig
from experiment_core_utils.hf_model_loading import load_model_tokenizer  # adjust if your path differs
from your_flops_module import get_flops, maybe_convert_to_float  # use your own defined FLOPs helpers

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def load_config_from_json(path):
    with open(path, "r") as f:
        config_dict = json.load(f)
    return ExperimentConfig(**config_dict)


def test_flops_from_config(config_path, batch_size=1):
    # Load experiment config from JSON
    config = load_config_from_json(config_path)
    logging.info(f"Loaded config: {config.config_name}")

    # Load model & tokenizer using your own loader
    model, tokenizer = load_model_tokenizer(config)
    model.eval()

    # Put on GPU if not already
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Handle quantised models (convert to float if needed for FLOPs)
    model = maybe_convert_to_float(model)

    # Prepare dummy tokenised input
    max_len = config.max_input_tokens
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long).to(device)

    # Compute FLOPs
    logging.info(f"Computing FLOPs for input of shape {input_ids.shape}")
    flops = get_flops(model, input_ids)

    if flops is not None:
        logging.info(f"[✓] Estimated total FLOPs: {flops:,.0f}")
    else:
        logging.warning("[!] FLOPs computation failed.")

    return flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to ExperimentConfig JSON file.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for dummy FLOPs test.")
    args = parser.parse_args()

    test_flops_from_config(args.config, args.batch_size)
