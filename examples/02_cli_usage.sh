#!/bin/bash
#
# CLI Usage Examples
# ==================
#
# This script demonstrates all available CLI commands for the LLM Efficiency
# Measurement Tool. The CLI provides a user-friendly interface for running
# experiments, viewing results, and exporting data.
#
# Installation:
#   pip install llm-efficiency
#
# Or from source:
#   pip install -e .


echo "=========================================="
echo "LLM Efficiency CLI - Usage Examples"
echo "=========================================="

# 1. Interactive Configuration Wizard
# Creates a configuration file interactively
echo -e "\n--- 1. Create Configuration (Interactive) ---"
echo "Command: llm-efficiency init"
echo "This launches an interactive wizard to create experiment_config.yaml"

# 2. Run Experiment from Config File
# Run an experiment using a YAML configuration file
echo -e "\n--- 2. Run Experiment from Config ---"
echo "Command: llm-efficiency run experiment_config.yaml"
echo "Example config file:"
cat << EOF
---
model_name: "gpt2"
precision: "float16"
batch_size: 4
num_batches: 10
max_length: 128
dataset_name: "wikitext"
dataset_config: "wikitext-2-raw-v1"
output_dir: "./results"
quantization:
  enabled: false
EOF

# 3. Run Experiment with CLI Arguments
# Override config with command-line arguments
echo -e "\n--- 3. Run with CLI Overrides ---"
echo "Command:"
echo "  llm-efficiency run config.yaml \\"
echo "    --model gpt2-medium \\"
echo "    --batch-size 8 \\"
echo "    --num-batches 50 \\"
echo "    --precision float16"

# 4. List All Experiments
# Display all experiments in a formatted table
echo -e "\n--- 4. List Experiments ---"
echo "Command: llm-efficiency list"
echo "Shows: ID, Model, Timestamp, Tokens/sec, Energy (kWh), CO2 (kg)"

# 5. List with Filtering
echo -e "\n--- 5. List with Filters ---"
echo "Command: llm-efficiency list --model gpt2 --limit 10"
echo "Filter by model name and limit results"

# 6. Show Detailed Results
# Display comprehensive results for a specific experiment
echo -e "\n--- 6. Show Experiment Details ---"
echo "Command: llm-efficiency show <experiment-id>"
echo "Displays:"
echo "  - Configuration details"
echo "  - Performance metrics"
echo "  - Energy consumption"
echo "  - Compute metrics"
echo "  - Timing breakdown"

# 7. Export Results to CSV
# Export all or filtered experiments to CSV format
echo -e "\n--- 7. Export to CSV ---"
echo "Command: llm-efficiency export results.csv"
echo "Options:"
echo "  --format csv      # Default"
echo "  --format json     # JSON array"
echo "  --format pickle   # Python pickle (fastest)"

# 8. Export Specific Experiments
echo -e "\n--- 8. Export Filtered Results ---"
echo "Command:"
echo "  llm-efficiency export gpt2_results.csv \\"
echo "    --ids exp1,exp2,exp3"

# 9. Generate Summary Statistics
# Calculate aggregate statistics across experiments
echo -e "\n--- 9. Summary Statistics ---"
echo "Command: llm-efficiency summary"
echo "Shows:"
echo "  - Total experiments"
echo "  - Models tested"
echo "  - Average throughput"
echo "  - Total energy consumed"
echo "  - Total CO2 emissions"

# 10. Summary with Grouping
echo -e "\n--- 10. Summary by Model ---"
echo "Command: llm-efficiency summary --group-by model"
echo "Group statistics by model name or precision"

# 11. Help and Documentation
echo -e "\n--- 11. Get Help ---"
echo "Command: llm-efficiency --help"
echo "Command: llm-efficiency run --help"
echo "Command: llm-efficiency export --help"

# Real-world workflow example
echo -e "\n=========================================="
echo "COMPLETE WORKFLOW EXAMPLE"
echo "=========================================="

cat << 'EOF'

# Step 1: Create configuration
llm-efficiency init
# Follow prompts to create experiment_config.yaml

# Step 2: Run experiment
llm-efficiency run experiment_config.yaml

# Step 3: List results
llm-efficiency list

# Step 4: View specific result
llm-efficiency show exp_20240115_123456_abc123

# Step 5: Export all results
llm-efficiency export all_experiments.csv

# Step 6: Get summary statistics
llm-efficiency summary --group-by model

# Advanced: Compare multiple models
for model in gpt2 gpt2-medium gpt2-large; do
    llm-efficiency run config.yaml --model $model --output-dir results/$model
done

# Then export and analyze
llm-efficiency export comparison.csv
llm-efficiency summary --group-by model

EOF

echo -e "\n=========================================="
echo "For more information:"
echo "  Documentation: https://github.com/henrycgbaker/llm-efficiency-measurement-tool"
echo "  Issues: https://github.com/henrycgbaker/llm-efficiency-measurement-tool/issues"
echo "=========================================="
