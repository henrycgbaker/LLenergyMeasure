# Research: EleutherAI lm-evaluation-harness

**Date:** 2026-02-17
**Source:** Research agent transcript (agent-a5c2a1f.jsonl, 376KB)
**Status:** Agent stopped before final synthesis; findings compiled from web research and GitHub analysis

---

## Summary

The lm-evaluation-harness is EleutherAI's open-source framework for few-shot evaluation of language models. It serves as the backend for Hugging Face's Open LLM Leaderboard, is used internally by NVIDIA, Cohere, BigScience, BigCode, Nous Research, Mosaic ML, and has been cited in hundreds of papers. It is a library-first tool with a CLI wrapper and no built-in web UI.

---

## 1. Project Metadata

| Field | Value |
|-------|-------|
| Name | `lm_eval` |
| Current version | 0.4.11 (released 13 Feb 2026) |
| Dev version | 0.4.12.dev0 |
| Author | EleutherAI (contact@eleuther.ai) |
| Licence | MIT |
| Python | >= 3.10 |
| Repository | https://github.com/EleutherAI/lm-evaluation-harness |
| GitHub stars | ~11,434 |
| Development status | Alpha (3) |
| Active since | September 2021 |

---

## 2. Architecture

### 2.1 Product Layers

The tool operates as a **library + CLI** only. There is no built-in web UI or web server mode.

- **Layer 1 -- Python Library**: `lm_eval` package with `simple_evaluate()`, `EvaluatorConfig`, and `evaluate()` entry points
- **Layer 2 -- CLI**: `lm-eval` command with three subcommands (`run`, `ls`, `validate`)
- **No Layer 3**: No web server, no REST API, no dashboard

The CLI was refactored around v0.4.10 to use explicit subcommands. Backward compatibility is maintained: `lm-eval --model hf --tasks hellaswag` still works without the `run` subcommand.

### 2.2 Core Components

Three-part architecture: **models**, **tasks**, **metrics**.

Models inherit from `lm_eval.api.model.LM` with three required methods:
- `generate_until()` -- sample text until stopping criteria
- `loglikelihood()` -- compute log probability of target conditioned on input, returns `(log_prob, is_greedy)`
- `loglikelihood_rolling()` -- compute full-text log probability (for perplexity)

Models register via `@register_model()` decorator.

### 2.3 Codebase Size

| Category | Count |
|----------|-------|
| Core Python files (excl. tasks) | 71 |
| Task Python files | 651 |
| Task YAML configs | 13,374 |

Core package structure:

```
lm_eval/
  __init__.py, __main__.py
  _cli/          # CLI: harness.py, ls.py, run.py, validate.py, subcommand.py, utils.py
  api/           # Core abstractions: model.py, task.py, metrics.py, filter.py, group.py, instance.py, registry.py, samplers.py
  caching/       # Request caching
  config/        # evaluate_config.py, group.py, task.py
  decontamination/  # Data decontamination
  evaluator.py   # Main evaluation logic
  evaluator_utils.py
  filters/       # Post-processing: extraction, selection, transformation, custom, decontamination
  loggers/       # evaluation_tracker.py, wandb_logger.py
  models/        # 28+ model backends
  prompts/       # Prompt templates
  result_schema.py
  utils.py
```

---

## 3. CLI Reference

### 3.1 Subcommands

| Command | Purpose |
|---------|---------|
| `lm-eval run` | Run evaluations on language models |
| `lm-eval ls` | List available tasks, groups, subtasks, or tags |
| `lm-eval validate` | Validate task configurations |

### 3.2 Key Arguments for `lm-eval run`

**Model and task selection:**
- `--model/-M`: Model type/provider (default: `hf`)
- `--model_args/-a`: Constructor arguments as key=value pairs
- `--tasks/-t`: Space or comma-separated task names
- `--apply_chat_template`: Apply chat formatting to prompts

**Evaluation control:**
- `--num_fewshot/-f`: Few-shot example count
- `--batch_size/-b`: Batch processing size (integer, `auto`, or `auto:N`)
- `--max_batch_size`: Upper limit for auto-tuning
- `--device`: Hardware target (`cuda`, `cpu`, `mps`; default: `cuda`)
- `--gen_kwargs`: Generation parameters (temperature, top_p, etc.)
- `--limit/-L`: Restrict examples per task

**Output and logging:**
- `--output_path/-o`: Results directory or JSON file
- `--log_samples/-s`: Save all model inputs/outputs for post-hoc analysis
- `--predict_only/-x`: Generate predictions without computing metrics

**Configuration:**
- `--config/-C`: YAML configuration file path
- `--include_path`: External task directory
- `--system_instruction`: Custom system prompt
- `--seed`: Random seed specification

**Integrations:**
- `--wandb_args`: Weights & Biases configuration
- `--hf_hub_log_args`: HuggingFace Hub logging settings

**Listing:**
```bash
lm-eval ls [tasks|groups|subtasks|tags] [--include_path DIR]
```

**Validation:**
```bash
lm-eval validate --tasks <task1,task2> [--include_path DIR]
```

---

## 4. Configuration System

### 4.1 YAML Config Files

Instead of passing many CLI arguments, parameters can be defined in a YAML configuration file:

```bash
lm-eval run --config eval_config.yaml
```

**Key sections:**
- Model setup: `model`, `model_args`
- Task definition: `tasks`, `num_fewshot`, `limit`
- Execution: `batch_size`, `device`, `seed`
- Output: `output_path`, `log_samples`, `wandb_args`, `hf_hub_log_args`
- Advanced: `gen_kwargs`, `apply_chat_template`, `system_instruction`, `cache_requests`

**Priority: CLI arguments override config file values.**

### 4.2 Task YAML Schema

Tasks use declarative YAML configurations:

```yaml
task: <unique_identifier>
tag: [optional_categories]
dataset_path: <hf_hub_name>
dataset_name: <config_name>
training_split: train
validation_split: validation
test_split: test
fewshot_split: train

doc_to_text: "{{passage}}\nQuestion: {{question}}"  # Jinja2 templating
doc_to_target: "{{answer}}"
doc_to_choice: ["A", "B", "C", "D"]

metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true

metadata:
  version: 1.0
```

Tasks are organised hierarchically into families, with support for:
- Jinja2 prompt templating
- Python function-based prompts via `\!function` operator
- PromptSource integration
- Multi-language support
- Group aggregation (e.g., MMLU's 57 subtasks)
- Configurable few-shot examples with sampler strategies

---

## 5. Model Backends

### 5.1 Registered Backends

28+ model backends registered via `@register_model()`:

**Local inference:**
- `huggingface.py` -- HuggingFace Transformers (primary)
- `vllm_causallms.py` -- vLLM (fast, memory-efficient, continuous batching)
- `sglang_causallms.py`, `sglang_generate_API.py` -- SGLang (tensor parallelism)
- `gguf.py` -- GGML/GGUF models
- `mamba_lm.py` -- Mamba models
- `nemo_lm.py`, `megatron_lm.py` -- NeMo/Megatron-LM (enterprise)
- `neuron_optimum.py` -- AWS Neuron
- `optimum_lm.py`, `optimum_ipex.py` -- HuggingFace Optimum
- `hf_vlms.py` -- Vision-language models
- `hf_audiolm.py` -- Audio language models
- `hf_steered.py` -- Steering vector evaluation
- `mistral3.py` -- Mistral-specific
- `winml.py` -- Windows ML

**API-based:**
- `openai_completions.py` -- OpenAI
- `anthropic_llms.py` -- Anthropic
- `textsynth.py` -- TextSynth
- `api_models.py` -- Generic API (TemplateAPI superclass)
- `ibm_watsonx_ai.py` -- IBM WatsonX
- `dummy.py` -- Testing

### 5.2 Installation via pip extras

Since December 2025, the base `lm-eval` package no longer includes `transformers` or `torch`. Backends are installed via extras:

```bash
pip install lm-eval          # Base only (no backends)
pip install "lm_eval[hf]"    # HuggingFace (torch, transformers, accelerate, peft)
pip install "lm_eval[vllm]"  # vLLM
pip install "lm_eval[api]"   # API models (requests, aiohttp, tiktoken)
pip install "lm_eval[gptq]"  # GPTQ quantization
pip install "lm_eval[hf,vllm,api]"  # Multiple backends
```

**Task-specific extras:** `tasks`, `acpbench`, `ifeval`, `japanese_leaderboard`, `longbench`, `math`, `multilingual`, `ruler`

**Development:** `pip install -e ".[dev]"`

**Conflict note:** `acpbench` is incompatible with `math/tasks`; `gptq` is incompatible with `vllm`.

### 5.3 API Model Configuration

API models extend `TemplateAPI` superclass:
- `num_concurrent`: Parallel requests (default: 1)
- `tokenizer_backend`: `tiktoken`, `huggingface`, or None
- `batch_size`, `timeout` (30s), `max_retries` (3)
- Completion endpoints support loglikelihood tasks; chat-completion endpoints are limited to generation only

---

## 6. Python API

### 6.1 Three Entry Points

**`simple_evaluate()`** -- recommended for most use cases:
```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=gpt2",
    tasks=["hellaswag"],
    batch_size=8,
    device="cuda:0",
)
print(results["results"])
```

**`EvaluatorConfig`** -- config-based approach for structured management.

**`evaluate()`** -- low-level, direct access to task dictionaries.

### 6.2 `simple_evaluate()` Parameters

- `model`: String identifier ("hf", "vllm") or pre-initialised `LM` instance
- `model_args`: Constructor parameters (string or dict)
- `tasks`: List of task names
- `num_fewshot`: Integer for few-shot count
- `batch_size`: Integer or "auto"
- `device`: "cuda", "cpu", "mps"
- `limit`: Max examples per task (int or float)
- `log_samples`: Capture model inputs/outputs
- `gen_kwargs`: Generation parameters dict
- `apply_chat_template`: Boolean or string
- `system_instruction`: Custom system prompt
- `task_manager`: TaskManager instance for custom tasks

### 6.3 Custom Model Implementation

Extend `lm_eval.api.model.LM` with three methods plus a `batch_size` property:

```python
class MyModel(lm_eval.api.model.LM):
    def loglikelihood(self, requests): ...
    def generate_until(self, requests): ...
    def loglikelihood_rolling(self, requests): ...

    @property
    def batch_size(self): return 8
```

Pass the instance directly to `simple_evaluate(model=MyModel())`.

---

## 7. Results Schema

### 7.1 Output Files

Results saved to:
- `{output_path}/{model_name}/results_YYYY-MM-DDTHH-MM-SS.xxxxx.json` -- aggregated metrics
- `{output_path}/{model_name}/samples_{task_name}_YYYY-MM-DDTHH-MM-SS.xxxxx.jsonl` -- per-sample (when `--log_samples`)

### 7.2 EvalResults TypedDict

```
EvalResults
  results: dict[task_name -> _TaskMetrics]
  groups: dict (aggregated group-level metrics)
  group_subtasks: dict[group -> list[subtask]]
  configs: dict (full YAML task configurations)
  versions: dict (task versions)
  n-shot: dict (few-shot counts)
  higher_is_better: dict (metric direction)
  n-samples: dict[task -> _SampleCount(original, effective)]
  samples: dict[task -> list[SampleResult]]  # if log_samples=True
  config: _EvalConfig (model, args, batch_size, device, seeds, git_hash, date)
  Environment info (pretty_env_info, transformers_version, lm_eval_version)
  Tokenizer info (pad_token, eos_token, bos_token, eot_token_id, max_length)
  Model identity (model_source, model_name, model_name_sanitized)
  Chat fields (system_instruction, chat_template, fewshot_as_multiturn)
  task_hashes, total_evaluation_time_seconds
```

**Per-task metrics (_TaskMetrics):** `name`, `alias`, `sample_len`, dynamic metric keys like `"acc,none"`.

**Per-sample results (SampleResult):** `doc_id`, `doc`, `target`, `arguments`, `resps`, `filtered_resps`, `filter`, `metrics`, `doc_hash`, `prompt_hash`, `target_hash`.

### 7.3 EvaluationTracker

The `EvaluationTracker` handles result persistence:

| Method | Purpose |
|--------|---------|
| `save_results_aggregated()` | JSON with timestamps and task hashes |
| `save_results_samples()` | JSONL per task with arguments, responses, targets |
| `log_experiment_args()` | Records model parameters and configuration |
| `log_end_time()` | Captures evaluation completion time |
| `recreate_metadata_card()` | Generates HF Hub dataset documentation |

**HuggingFace Hub upload:** Pushes aggregated results via `api.upload_file()`, sample results via `api.upload_folder()`. Supports public/private repositories with optional auto-access gating.

---

## 8. Multi-GPU Evaluation

Three options:
1. **Data parallelism** (each GPU loads full model): `accelerate launch -m lm_eval ...`
2. **Model sharding** (split across GPUs): `parallelize=True` flag
3. **Combined approach** for very large models

---

## 9. Docker and Deployment

- **Primary distribution**: PyPI (`pip install lm-eval`)
- **Docker**: NVIDIA provides official NeMo Evaluator-compatible container
- **CPU Docker**: `docker build -f Dockerfile.cpu -t opea/lm-eval:latest .`
- **From source**: `git clone --depth 1 ... && pip install -e .`

No official Docker images from EleutherAI; Docker usage is primarily through NVIDIA's NeMo integration or community builds.

---

## 10. No Web UI

**lm-evaluation-harness does not have a built-in web UI or web server mode.** It is purely a CLI/library tool. The web presence comes through:
- HuggingFace Open LLM Leaderboard (Gradio Space consuming results)
- Integration with W&B, Zeno for visualisation
- Third-party wrappers (Lemonade Server)

---

## 11. Batch Evaluation

The tool supports evaluating multiple tasks in a single run via `--tasks task1,task2,task3`. Multiple model evaluations require separate runs with `--output_path` organising results by model. Auto batch sizing available with `--batch_size auto` or `auto:N`.

No built-in "campaign sweep" or "grid search" functionality. Multi-model sweeps require external scripting.

---

## 12. Relevance to LLenergyMeasure

| Aspect | lm-eval-harness Pattern | LLenergyMeasure Implication |
|--------|------------------------|----------------------------|
| Package structure | Library + CLI, no web | Validates library-first approach |
| Backend extras | `pip install lm_eval[vllm]` | Identical pattern already used |
| Config | YAML + CLI, CLI overrides | Same layered precedence design |
| Results | JSON + JSONL, timestamped | Similar but LLenergyMeasure uses Pydantic models |
| Task system | 13,374 YAML configs | Shows declarative task definition scales |
| No web UI | Purely CLI/library | Web layer is separate concern (leaderboard) |
| Model abstraction | `LM` base class, `@register_model` | Similar to our backend protocol pattern |
| HF Hub integration | EvaluationTracker uploads | Model for future result sharing |

**Key takeaway:** lm-eval-harness validates the library+CLI architecture without a built-in web layer. The web presentation (HF Leaderboard) is a completely separate system that consumes the results. This supports LLenergyMeasure's planned separation of CLI tool (v2.0) from web platform (v4.0).

---

## Sources

- [GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness)
- [PyPI package](https://pypi.org/project/lm-eval/)
- [Architecture blog post](https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html)
- [CLI reference (interface.md)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
- [Python API (python-api.md)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)
- [Model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
- [Task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md)
- [Config files guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md)
- [HF Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
