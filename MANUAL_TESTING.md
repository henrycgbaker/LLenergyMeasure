# Manual Testing Guide

This guide provides step-by-step instructions for manually testing all functionality of the LLM Efficiency Measurement Tool. Follow this checklist to verify that everything works correctly after installation or before a release.

## Prerequisites

- Python 3.11 or 3.12 installed
- 4GB+ RAM available
- Internet connection (for downloading models)
- Optional: NVIDIA GPU with CUDA for GPU tests

## Test Environment Setup

### 1. Fresh Installation Test

```bash
# Create a fresh test directory
mkdir ~/llm-efficiency-test
cd ~/llm-efficiency-test

# Create a virtual environment
python3.11 -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Clone the repository
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool

# Install PyTorch (platform-specific - see INSTALLATION.md)
# For macOS:
pip install torch torchvision torchaudio

# For Linux CPU-only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the package
pip install -e .
```

**✅ Verification:**
- [ ] No installation errors
- [ ] All dependencies installed successfully
- [ ] Virtual environment activated

### 2. Version Check

```bash
llm-efficiency --version
```

**Expected Output:**
```
LLM Efficiency Measurement Tool version 2.0.0
```

**✅ Verification:**
- [ ] Version displays correctly
- [ ] No errors or warnings
- [ ] PyTorch warning about pynvml is suppressed

---

## Core Functionality Tests

### Test 1: Help Command

```bash
llm-efficiency --help
```

**Expected Output:**
- List of available commands: `init`, `run`, `list`, `show`, `export`, `summary`
- Brief description of each command
- Usage information

**✅ Verification:**
- [ ] Help text displays
- [ ] All commands listed
- [ ] Formatting is clean and readable

---

### Test 2: Init Command - Default Config

```bash
# Test creating default config
llm-efficiency init
```

**Interactive Prompts (use these test values):**
1. Model name: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
2. Precision: `float16` (default)
3. Batch size: `4`
4. Number of prompts: `5`
5. Max input tokens: `128`
6. Max output tokens: `32`
7. Enable quantization: `n`

**Expected Output:**
```
✓ Configuration saved to config.json
```

**✅ Verification:**
- [ ] Config wizard runs without errors
- [ ] `config.json` file created in current directory
- [ ] File contains valid JSON
- [ ] Values match what you entered

**Verify Config Contents:**
```bash
cat config.json | python -m json.tool
```

Check that the JSON is valid and contains your settings.

---

### Test 3: Init Command - Custom Config Name

```bash
llm-efficiency init my-experiment.json
```

**Use the same test values as Test 2**

**✅ Verification:**
- [ ] `my-experiment.json` created (not `config.json`)
- [ ] File contains valid JSON
- [ ] No extra files created

```bash
ls -la *.json
```

Should show both `config.json` and `my-experiment.json`.

---

### Test 4: Run Command - With Config File

```bash
llm-efficiency run --config config.json --num-prompts 3
```

**This will:**
1. Download the TinyLlama model (~2GB, first time only)
2. Load the model
3. Run inference on 3 prompts
4. Save results

**Expected Output:**
```
╭────────────────────────────────────╮
│ LLM Efficiency Measurement Tool    │
│ Version 2.0.0                      │
╰────────────────────────────────────╯

✓ Configuration loaded
  Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Precision: float16
  Batch size: 4
  Prompts: 3

✓ Experiment ID: [some-id]
✓ Model loaded on [cpu/cuda:0]
✓ Loaded 3 prompts
✓ Inference complete
  Throughput: [X.XX] tokens/s
  Latency: [X.XX] ms/query
✓ FLOPs: [X,XXX]
✓ GPU Memory: [X.XX] GB (or skipped on CPU)
✓ Results saved to results/[experiment-id].json

════════════════════════════════════════
Experiment Complete!
Experiment ID: [some-id]
Throughput: [X.XX] tokens/s
Energy: [X.XX] Wh
Emissions: [X.XX] g CO2
════════════════════════════════════════
```

**⏱️ Expected Duration:** 2-5 minutes (first run), 30-60 seconds (subsequent runs)

**✅ Verification:**
- [ ] Model downloads successfully (first time)
- [ ] No errors during model loading
- [ ] Inference completes successfully
- [ ] Throughput > 0 tokens/s
- [ ] Results file created in `results/` directory
- [ ] No PyTorch pynvml warning displayed

**Note the Experiment ID** for later tests.

---

### Test 5: Run Command - CLI Arguments Only

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --precision float16 \
  --batch-size 2 \
  --num-prompts 3 \
  --max-input 128 \
  --max-output 32
```

**✅ Verification:**
- [ ] Runs without config file
- [ ] Uses CLI arguments for configuration
- [ ] Completes successfully
- [ ] Results saved

---

### Test 6: Run Command - Config Override

```bash
llm-efficiency run --config config.json --batch-size 2 --num-prompts 2
```

**✅ Verification:**
- [ ] Loads config from file
- [ ] Overrides batch_size to 2 (not 4 from config)
- [ ] Overrides num_prompts to 2 (not 5 from config)
- [ ] Other settings from config file used
- [ ] Completes successfully

**Check the output shows:**
```
Batch size: 2
Prompts: 2
```

---

### Test 7: List Command

```bash
llm-efficiency list
```

**Expected Output:**
```
Experiments (3 total)
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ ID              ┃ Model                           ┃ Throughput  ┃ Energy   ┃ Timestamp  ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ [exp-id-1]      │ TinyLlama/TinyLlama-1.1B-...   │ X.XX tok/s  │ X.XX Wh  │ 2024-XX-XX │
│ [exp-id-2]      │ TinyLlama/TinyLlama-1.1B-...   │ X.XX tok/s  │ X.XX Wh  │ 2024-XX-XX │
│ [exp-id-3]      │ TinyLlama/TinyLlama-1.1B-...   │ X.XX tok/s  │ X.XX Wh  │ 2024-XX-XX │
└─────────────────┴─────────────────────────────────┴─────────────┴──────────┴────────────┘
```

**✅ Verification:**
- [ ] Shows all 3 experiments from previous tests
- [ ] Table formatted correctly
- [ ] IDs, models, throughput, energy, and timestamps shown
- [ ] No errors

---

### Test 8: Show Command

```bash
# Use the experiment ID from Test 4
llm-efficiency show [experiment-id]
```

**Expected Output:**
```
╭─────────────────────────────────────╮
│ Experiment [experiment-id]          │
│ 2024-XX-XXTXX:XX:XX                 │
╰─────────────────────────────────────╯

Model Information
  Name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  Parameters: X,XXX,XXX
  Precision: float16

Inference Metrics
  Throughput: X.XX tokens/s
  Queries/s: X.XX
  Avg latency: X.XX ms
  Total tokens: XXX
  Prompts: 3

Compute Metrics
  FLOPs: X,XXX,XXX
  GPU Memory: X.XX MB
  GPU Peak: X.XX MB

Energy Metrics
  Duration: X.XX s
  Total energy: X.XX Wh
  CPU energy: X.XX Wh
  GPU energy: X.XX Wh
  RAM energy: X.XX Wh
  Emissions: X.XX g CO2
```

**✅ Verification:**
- [ ] Shows detailed metrics for experiment
- [ ] All sections present (Model, Inference, Compute, Energy)
- [ ] Numbers are reasonable (no NaN, no zeros where unexpected)
- [ ] Formatting is clean

---

### Test 9: Export Command - CSV

```bash
llm-efficiency export results.csv --format csv
```

**Expected Output:**
```
✓ Exported to results.csv (CSV format)
```

**Verify Export:**
```bash
head -5 results.csv
```

**✅ Verification:**
- [ ] CSV file created
- [ ] Contains header row
- [ ] Contains data rows for all experiments
- [ ] Columns include: experiment_id, model_name, throughput, energy, etc.

---

### Test 10: Export Command - JSON

```bash
llm-efficiency export results.json --format json
```

**Verify Export:**
```bash
cat results.json | python -m json.tool | head -20
```

**✅ Verification:**
- [ ] JSON file created
- [ ] Valid JSON format
- [ ] Contains array of experiment objects

---

### Test 11: Export Command - Pickle

```bash
llm-efficiency export results.pkl --format pickle
```

**Verify Export:**
```bash
python -c "import pickle; data = pickle.load(open('results.pkl', 'rb')); print(f'Loaded {len(data)} experiments')"
```

**✅ Verification:**
- [ ] Pickle file created
- [ ] Can be loaded without errors
- [ ] Contains all experiments

---

### Test 12: Summary Command

```bash
llm-efficiency summary
```

**Expected Output:**
```
╭─────────────────────────────╮
│ Summary Statistics          │
│ Total experiments: 3        │
╰─────────────────────────────╯

Throughput
  Mean: X.XX tokens/s
  Max: X.XX tokens/s
  Min: X.XX tokens/s

Energy
  Total: X.XX Wh
  Mean: X.XX Wh
```

**✅ Verification:**
- [ ] Shows aggregate statistics
- [ ] Mean values are reasonable (between min and max)
- [ ] Total energy is sum of all experiments

---

## Error Handling Tests

### Test 13: Missing Model Argument

```bash
llm-efficiency run
```

**Expected Output:**
```
Error: --model is required when not using --config
```

**✅ Verification:**
- [ ] Error message displayed
- [ ] Exit code is 1 (error)
- [ ] No crash or traceback

---

### Test 14: Nonexistent Config File

```bash
llm-efficiency run --config nonexistent.json
```

**Expected Output:**
```
Error: Config file nonexistent.json not found
```

**✅ Verification:**
- [ ] Clear error message
- [ ] No crash or traceback
- [ ] Exit code is 1

---

### Test 15: Invalid Experiment ID

```bash
llm-efficiency show invalid-experiment-id
```

**Expected Output:**
```
Error: Experiment invalid-experiment-id not found
```

**✅ Verification:**
- [ ] Clear error message
- [ ] No crash or traceback

---

### Test 16: Invalid Export Format

```bash
llm-efficiency export results.txt --format invalid
```

**Expected Output:**
```
Error: Invalid format 'invalid'. Must be one of: csv, pickle, json
```

**✅ Verification:**
- [ ] Clear error message listing valid formats
- [ ] No file created

---

## Advanced Configuration Tests

### Test 17: Quantization (4-bit)

Create a quantization config:

```bash
llm-efficiency init quant-config.json
```

**Settings:**
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Precision: `float16`
- Batch size: `2`
- Prompts: `3`
- Max input: `128`
- Max output: `32`
- **Quantization: `y` (YES)**

**Run with quantization:**
```bash
llm-efficiency run --config quant-config.json
```

**✅ Verification:**
- [ ] Output shows "Quantization: 4-bit"
- [ ] Model loads successfully
- [ ] Inference completes
- [ ] Memory usage lower than non-quantized (if GPU available)

**Note:** On CPU-only systems, quantization may fail or be slow. This is expected.

---

### Test 18: Different Precision (bfloat16)

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --precision bfloat16 \
  --batch-size 2 \
  --num-prompts 2
```

**✅ Verification:**
- [ ] Output shows "Precision: bfloat16"
- [ ] Runs successfully
- [ ] Results saved

**Note:** bfloat16 requires specific hardware support. May fall back to float16 on some systems.

---

### Test 19: Larger Batch Size

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --batch-size 16 \
  --num-prompts 32 \
  --max-input 256 \
  --max-output 64
```

**✅ Verification:**
- [ ] Handles larger batch size
- [ ] Throughput increases with larger batch
- [ ] No out-of-memory errors (if sufficient RAM/VRAM)

---

### Test 20: Custom Dataset

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset AIEnergyScore/text_generation \
  --num-prompts 5
```

**✅ Verification:**
- [ ] Downloads/loads custom dataset
- [ ] Inference runs successfully

---

### Test 21: Disable Energy Tracking

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-prompts 2 \
  --no-energy
```

**✅ Verification:**
- [ ] Runs without energy tracking
- [ ] Final output doesn't show energy/emissions
- [ ] Results file may have null/missing energy fields

---

### Test 22: Verbose Logging

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-prompts 2 \
  --verbose
```

**✅ Verification:**
- [ ] Additional debug output displayed
- [ ] Shows detailed progress information
- [ ] No errors in verbose output

---

## Platform-Specific Tests

### Test 23: macOS Specific

**Only run on macOS:**

```bash
# Verify no installation issues
pip install -e .

# Run a simple experiment
llm-efficiency run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num-prompts 2
```

**✅ Verification:**
- [ ] Installation succeeds on macOS
- [ ] No bitsandbytes warnings (if not using quantization)
- [ ] CPU inference works
- [ ] Energy tracking works (should track CPU/RAM)

---

### Test 24: Linux Specific (CUDA)

**Only run on Linux with NVIDIA GPU:**

```bash
# Verify CUDA available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run with GPU
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-prompts 5
```

**✅ Verification:**
- [ ] CUDA is available and detected
- [ ] Model loads on GPU (output shows "cuda:0")
- [ ] GPU memory stats displayed
- [ ] GPU energy tracking works
- [ ] Throughput higher than CPU

---

### Test 25: Linux Specific (CPU-only)

**Run on Linux without GPU:**

```bash
llm-efficiency run \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-prompts 3
```

**✅ Verification:**
- [ ] Runs on CPU (output shows "cpu")
- [ ] No GPU memory stats
- [ ] Energy tracking still works (CPU/RAM only)

---

## Integration Tests

### Test 26: Multiple Sequential Experiments

Run 3 experiments back-to-back:

```bash
llm-efficiency run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num-prompts 2
llm-efficiency run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num-prompts 3 --batch-size 2
llm-efficiency run --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --num-prompts 4 --batch-size 4
```

Then verify:

```bash
llm-efficiency list
llm-efficiency summary
```

**✅ Verification:**
- [ ] All 3 experiments complete successfully
- [ ] Each gets unique experiment ID
- [ ] List shows all experiments
- [ ] Summary aggregates correctly
- [ ] Results files all created

---

### Test 27: Export All Results

```bash
llm-efficiency export all-results.csv
llm-efficiency export all-results.json
llm-efficiency export all-results.pkl
```

**✅ Verification:**
- [ ] All export formats work
- [ ] All experiments included in exports
- [ ] Files can be loaded/opened

---

### Test 28: Full Workflow Test

Complete workflow from scratch:

```bash
# 1. Initialize
llm-efficiency init workflow-test.json
# Use: batch_size=2, num_prompts=3

# 2. Run experiment
llm-efficiency run --config workflow-test.json

# 3. View results
llm-efficiency list

# 4. Show detailed results (use ID from step 2)
llm-efficiency show [experiment-id]

# 5. Export
llm-efficiency export workflow-results.csv

# 6. Summary
llm-efficiency summary
```

**✅ Verification:**
- [ ] All steps complete without errors
- [ ] Data flows through entire pipeline
- [ ] Final CSV contains the experiment

---

## Cleanup

After testing:

```bash
# Remove test results
rm -rf results/
rm *.json *.csv *.pkl

# Deactivate and remove virtual environment
deactivate
cd ../..
rm -rf llm-efficiency-test/
```

---

## Test Results Summary

Use this checklist to track your testing progress:

### Core Functionality
- [ ] Test 1: Help Command
- [ ] Test 2: Init (default)
- [ ] Test 3: Init (custom name)
- [ ] Test 4: Run with config
- [ ] Test 5: Run with CLI args
- [ ] Test 6: Config override
- [ ] Test 7: List
- [ ] Test 8: Show
- [ ] Test 9: Export CSV
- [ ] Test 10: Export JSON
- [ ] Test 11: Export Pickle
- [ ] Test 12: Summary

### Error Handling
- [ ] Test 13: Missing model
- [ ] Test 14: Nonexistent config
- [ ] Test 15: Invalid experiment ID
- [ ] Test 16: Invalid export format

### Advanced Configuration
- [ ] Test 17: Quantization
- [ ] Test 18: Different precision
- [ ] Test 19: Larger batch size
- [ ] Test 20: Custom dataset
- [ ] Test 21: No energy tracking
- [ ] Test 22: Verbose logging

### Platform-Specific
- [ ] Test 23: macOS
- [ ] Test 24: Linux with GPU
- [ ] Test 25: Linux CPU-only

### Integration
- [ ] Test 26: Multiple experiments
- [ ] Test 27: Export all
- [ ] Test 28: Full workflow

---

## Issue Reporting

If any test fails, report the issue with:

1. **Test number and name**
2. **Platform** (OS, Python version)
3. **Command executed**
4. **Expected vs actual output**
5. **Full error message/traceback**
6. **Steps to reproduce**

Example:

```
Test 4 Failed: Run Command - With Config File

Platform: macOS 14.0, Python 3.11.7
Command: llm-efficiency run --config config.json --num-prompts 3

Expected: Model loads and inference runs
Actual: AttributeError: 'ExperimentConfig' object has no attribute 'from_dict'

Error:
[full traceback here]

Steps to reproduce:
1. Run llm-efficiency init
2. Accept all defaults
3. Run llm-efficiency run --config config.json --num-prompts 3
```

---

## Notes

- **First run**: Expect 2-5 minutes for model download
- **Subsequent runs**: Should be much faster (30-60 seconds)
- **CPU vs GPU**: GPU tests require NVIDIA GPU with CUDA
- **Memory**: TinyLlama requires ~2GB RAM/VRAM
- **Network**: Required for first model download only

## Questions?

If you encounter issues or have questions:
1. Check `INSTALLATION.md` for platform-specific setup
2. Check `TESTING.md` for automated test information
3. Review error messages carefully
4. Open an issue with details from "Issue Reporting" section above
