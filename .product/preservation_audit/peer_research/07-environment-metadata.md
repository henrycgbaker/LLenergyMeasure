# Peer Research: Environment Metadata for Reproducibility

> Generated 2026-02-26. Peer evidence for preservation audit item N-C01.

---

## Evidence Per Tool

### 1. MLflow

**Source**: [MLflow Tracking docs](https://mlflow.org/docs/latest/tracking/), [System Metrics docs](https://mlflow.org/docs/latest/ml/tracking/system-metrics/), [mlflow_tags.py](https://github.com/mlflow/mlflow/blob/master/mlflow/utils/mlflow_tags.py)

**Approach**: MLflow does *not* auto-capture hardware metadata as first-class fields. Environment
data is split across two mechanisms: **automatic run tags** and **optional system metrics**.

**Automatic run tags** (set at `mlflow.start_run()`):

| Tag | Content |
|-----|---------|
| `mlflow.user` | OS username |
| `mlflow.source.type` | `NOTEBOOK`, `JOB`, `PROJECT`, `LOCAL` |
| `mlflow.source.name` | Script path or notebook name |
| `mlflow.source.git.commit` | Git SHA |
| `mlflow.source.git.branch` | Git branch name |
| `mlflow.source.git.repoURL` | Git remote URL |
| `mlflow.source.git.dirty` | Uncommitted changes flag |

MLflow does **not** auto-tag GPU name, OS version, Python version, CPU model, or CUDA version.
These are user-logged or handled by a third-party plugin ([mlflow-sysmetrics](https://github.com/hugodscarvalho/mlflow-sysmetrics)).

The mlflow-sysmetrics plugin adds: `sys.cpu` (model), `sys.cpu_cores`, `sys.memory_gb`,
`sys.disk_free_gb`, `sys.platform` (OS + kernel), `sys.gpu` (via nvidia-smi on Linux).

**System metrics** (time-series, sampled every 10s, opt-in via `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING`):

| Metric | Name |
|--------|------|
| CPU utilisation | `system/cpu_utilization_percentage` |
| System memory (MB) | `system/system_memory_usage_megabytes` |
| System memory (%) | `system/system_memory_usage_percentage` |
| GPU utilisation (%) | `system/gpu_utilization_percentage` |
| GPU memory (MB) | `system/gpu_memory_usage_megabytes` |
| GPU memory (%) | `system/gpu_memory_usage_percentage` |
| GPU power (W) | `system/gpu_power_usage_watts` |
| GPU power (%) | `system/gpu_power_usage_percentage` |
| Network rx/tx (MB) | `system/network_receive_megabytes`, `system/network_transmit_megabytes` |
| Disk (MB) | `system/disk_usage_megabytes`, `system/disk_available_megabytes` |

GPU metrics require `nvidia-ml-py` (pynvml).

**Storage**: Tags stored in run metadata. System metrics stored as time-series alongside
run metrics. No separate environment file.

**Key fields captured**: Git commit, branch, user, script path. GPU power, utilisation, memory
(time-series only). No GPU model name, no CUDA version, no Python version natively.

---

### 2. Weights & Biases (W&B)

**Source**: [System Metrics Reference](https://docs.wandb.ai/models/ref/python/experiments/system-metrics), [wandb-metadata.json discussion](https://github.com/wandb/wandb/issues/5256)

**Approach**: W&B auto-captures extensive metadata at `wandb.init()`, stored in a
`wandb-metadata.json` file per run. System metrics are sampled every 15 seconds.

**wandb-metadata.json fields** (auto-captured, no user action needed):

| Field | Content |
|-------|---------|
| `host` | Hostname |
| `os` | Full OS string (e.g. `Linux-5.15.0-1019-aws-x86_64-with-glibc2.29`) |
| `python` | Python version (e.g. `3.8.10`) |
| `username` | OS user |
| `executable` | Python interpreter path |
| `cpu_count` | Physical CPU cores |
| `cpu_count_logical` | Logical CPU cores |
| `cpu_freq` | CPU frequency |
| `cpu_freq_per_core` | Per-core frequency |
| `gpu` | GPU model name(s) |
| `gpu_count` | Number of GPUs |
| `gpu_devices` | Per-device memory details |
| `memory` | Total RAM (GB) |
| `disk` | Total and used disk space |
| `program` | Script path |
| `codePath` | Code path |
| `args` | Command-line arguments |
| `commit` | Git commit hash |
| `remote` | Git remote URL |
| `startedAt` | Run start timestamp |

**System metrics** (time-series, per GPU index):

| Category | Metrics |
|----------|---------|
| GPU | Memory utilisation (%), memory allocated (%), utilisation (%), temperature (C), power usage (W/%), SM clock, memory clock, graphics clock, encoder utilisation (%) |
| CPU | Process CPU %, process CPU threads |
| Memory | Process RSS (MB), process memory (%), system memory (%), available memory (MB) |
| Disk | Usage (%), usage (GB), read (MB), write (MB) |
| Network | Bytes sent, bytes received |

Supports NVIDIA, AMD, Apple ARM, Graphcore IPU, Google TPU, AWS Trainium.

**Storage**: Metadata JSON stored alongside run artifacts. Accessible via API:
`run.file("wandb-metadata.json").download()`.

**Key fields captured**: GPU name, GPU count, GPU memory, OS, Python version, hostname, CPU count,
RAM, disk, git commit. The most comprehensive auto-capture of any tool surveyed.

---

### 3. lm-eval (EleutherAI lm-evaluation-harness)

**Source**: [evaluator.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py), [loggers/utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/loggers/utils.py)

**Approach**: Environment info is appended to the results dict via `add_env_info()`.
Minimal — focused on software reproducibility, not hardware.

**Fields stored in results JSON**:

| Field | Content |
|-------|---------|
| `pretty_env_info` | Output of `torch.utils.collect_env.get_pretty_env_info()` (includes PyTorch version, CUDA version, cuDNN version, GPU name, OS, GCC version, Clang version, CMake version, libc version, Python version, NumPy version) |
| `transformers_version` | Transformers library version |
| `lm_eval_version` | lm-eval package version (tool's own version) |
| `upper_git_hash` | Git commit hash of the parent repo |
| `git_hash` | Git commit hash of lm-eval |
| `date` | Execution timestamp |

Also stores full config: `model`, `model_args`, `batch_size`, `device`, seeds, etc.

**Key observation**: `pretty_env_info` is a human-readable multi-line string from PyTorch,
not structured data. It contains GPU name, CUDA version, cuDNN, OS, Python version, but as
a blob rather than individually queryable fields. This is a pragmatic but lossy approach.

**Storage**: Embedded in the results JSON file alongside task scores.

---

### 4. optimum-benchmark (Hugging Face)

**Source**: [system_utils.py](https://github.com/huggingface/optimum-benchmark/blob/main/optimum_benchmark/system_utils.py)

**Approach**: `get_system_info()` returns a flat dict of system metadata, stored in
benchmark config/report.

**Fields from `get_system_info()`**:

| Field | Content |
|-------|---------|
| `cpu` | CPU model name |
| `cpu_count` | Number of CPU cores |
| `cpu_ram_mb` | Total RAM in MB |
| `system` | OS name (Windows, Darwin, Linux) |
| `machine` | Architecture (e.g. `x86_64`) |
| `platform` | Full platform string |
| `processor` | Processor identifier |
| `python_version` | Python version |
| `gpu` | List of GPU device names (conditional on NVIDIA/ROCm) |
| `gpu_count` | Number of GPUs (conditional) |
| `gpu_vram_mb` | Total GPU VRAM in MB (conditional) |

GPU detection uses `is_nvidia_system()` (via nvidia-smi + pynvml) and `is_rocm_system()`
(via rocm-smi + amdsmi). GPU fields are absent if no GPU detected.

**Storage**: Stored in `benchmark_config.json` alongside benchmark configuration.

**Key observation**: Clean, minimal, machine-parseable. No CUDA version, no driver version,
no thermal data, no container info. Trade-off: simpler but less reproducibility-relevant
for energy measurement.

---

### 5. MLPerf

**Source**: [submission_rules.adoc](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc), [actual submission JSON](https://github.com/mlcommons/inference_results_v5.0/tree/main/closed/NVIDIA/systems)

**Approach**: The gold standard. Every submission requires a `<system_desc_id>.json` with
mandatory fields. The submission checker validates field presence (though not content quality).

**Mandatory fields from actual NVIDIA B200 submission** (v5.0):

| Category | Fields |
|----------|--------|
| **Organisation** | `submitter`, `division`, `status` |
| **System** | `system_name`, `system_type` (datacenter/edge), `system_type_detail`, `number_of_nodes` |
| **Host CPU** | `host_processor_model_name`, `host_processors_per_node`, `host_processor_core_count`, `host_processor_frequency`, `host_processor_caches`, `host_processor_interconnect` |
| **Host Memory** | `host_memory_capacity`, `host_memory_configuration` |
| **Host Storage** | `host_storage_capacity`, `host_storage_type` |
| **Accelerator** | `accelerator_model_name`, `accelerators_per_node`, `accelerator_frequency`, `accelerator_memory_capacity`, `accelerator_memory_configuration`, `accelerator_on-chip_memories` |
| **Interconnect** | `accelerator_host_interconnect` (e.g. `PCIe Gen5 x16`), `accelerator_interconnect` (e.g. `18x 4th Gen NVLink, 900GB/s`), `accelerator_interconnect_topology` |
| **Software** | `framework` (e.g. `TensorRT 10.8, CUDA 12.8`), `operating_system`, `other_software_stack` (e.g. `TensorRT 10.8, CUDA 12.8, cuDNN 8.9.7, Driver 550.90`), `sw_notes` |
| **Physical** | `cooling`, `disk_controllers`, `disk_drives`, `power_management`, `power_supply_details`, `power_supply_quantity_and_rating_watts` |
| **Networking** | `host_network_card_count`, `host_networking`, `host_networking_topology`, `network_speed_mbit`, `nics_enabled_connected`, `nics_enabled_firmware`, `nics_enabled_os`, `number_of_type_nics_installed` |
| **Firmware** | `boot_firmware_version`, `management_firmware_version` |
| **Notes** | `hw_notes` (e.g. `B200 TGP 1000W`), `other_hardware` |

Total: **~50 fields**. Many are blank strings (`""`) in practice but must be present.

**Storage**: Separate JSON file per system, alongside results.

**Key observation**: MLPerf is exhaustive because it serves *cross-organisation comparison*.
Most fields (networking, firmware, NICs, cooling) are irrelevant for a single-machine
measurement tool. The relevant subset for our tool is: accelerator name/count/memory,
host CPU, host memory, framework + driver + CUDA version, OS, PCIe generation,
interconnect, power supply rating.

---

### 6. Zeus

**Source**: [GPU reference](https://ml.energy/zeus/reference/device/gpu/nvidia/), [Getting Started](https://ml.energy/zeus/getting_started/)

**Approach**: Zeus is an energy measurement library, not an experiment tracker. It captures
GPU metadata only through its `NVIDIAGPU` class methods — focused on power/energy, not
general environment.

**Available GPU queries**:

| Method | Returns |
|--------|---------|
| `get_name()` | GPU model name |
| `get_power_management_limit()` | Current power limit (mW) |
| `get_power_management_limit_constraints()` | Min/max power limits (mW) |
| `get_instant_power_usage()` | Instantaneous power draw (mW) |
| `get_average_power_usage()` | Average power draw (mW) |
| `get_average_memory_power_usage()` | Memory subsystem power (mW) |
| `get_total_energy_consumption()` | Cumulative energy since driver load (mJ) — Volta+ only |
| `get_supported_memory_clocks()` | Available memory frequencies (MHz) |
| `get_supported_graphics_clocks()` | Available GPU core frequencies (MHz) |
| `get_gpu_temperature()` | Current temperature (C) |
| `supports_get_total_energy_consumption()` | Whether Volta+ energy API is available |

**Not exposed**: driver version, CUDA version, compute capability, VRAM total,
PCIe generation, Python version, OS, pip packages. Zeus delegates these to user code.

**Storage**: Zeus returns `Measurement` dataclasses with `time`, `gpu_energy`, `cpu_energy`.
No environment metadata in measurement output.

**Key observation**: Zeus captures power/thermal state that most other tools ignore.
Power limit, power limit constraints, supported clock frequencies, and memory power are
all relevant for energy reproducibility. Our tool should capture these via Zeus/NVML.

---

### 7. CodeCarbon

**Source**: [Output docs](http://docs.codecarbon.io/output.html), [EmissionsData source](https://github.com/mlco2/codecarbon/blob/master/codecarbon/output_methods/emissions_data.py)

**Approach**: CodeCarbon stores a flat `EmissionsData` dataclass with 38 fields. Every
tracking session appends one row to `emissions.csv`. Strong on hardware identity, weak on
software environment detail.

**`EmissionsData` fields** (all stored per row in emissions.csv):

| Category | Fields |
|----------|--------|
| **Identity** | `timestamp`, `project_name`, `run_id`, `experiment_id` |
| **Energy** | `emissions` (kg CO2), `emissions_rate`, `energy_consumed` (kWh) |
| **Power** | `cpu_power`, `gpu_power`, `ram_power` (W) |
| **Energy breakdown** | `cpu_energy`, `gpu_energy`, `ram_energy` (kWh) |
| **Hardware** | `cpu_count`, `cpu_model`, `gpu_count`, `gpu_model`, `ram_total_size` (GB) |
| **Utilisation** | `cpu_utilization_percent`, `gpu_utilization_percent`, `ram_utilization_percent`, `ram_used_gb` |
| **Software** | `os` (e.g. `Linux-5.15.0-113-generic-x86_64-with-glibc2.35`), `python_version`, `codecarbon_version` |
| **Location** | `country_name`, `country_iso_code`, `region`, `longitude`, `latitude` |
| **Cloud** | `on_cloud`, `cloud_provider`, `cloud_region` |
| **Tracking** | `tracking_mode`, `duration` (seconds) |
| **Sustainability** | `water_consumed`, `pue` (power usage effectiveness), `wue` |

**Storage**: Appended to `emissions.csv` (one row per tracking session). Also available via
API dashboard and custom `BaseOutput` extensions.

**Key observation**: CodeCarbon captures `cpu_model`, `gpu_model`, `gpu_count`, `os`,
`python_version`, and its own version — a solid baseline set. It also captures `pue`
(power usage effectiveness) which is relevant for energy measurement context. Missing:
CUDA version, driver version, GPU VRAM, compute capability, container info, pip packages.

---

### 8. PyTorch (`torch.cuda.get_device_properties`)

**Source**: [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_device_properties.html)

**Approach**: Not a tracking tool, but the standard GPU introspection API in the PyTorch
ecosystem. Returns a `_CudaDeviceProperties` namedtuple.

**Available fields**:

| Field | Content |
|-------|---------|
| `name` | GPU model name (e.g. `NVIDIA A100-PCIE-40GB`) |
| `major` | Compute capability major version |
| `minor` | Compute capability minor version |
| `total_memory` | Total VRAM in MB |
| `multi_processor_count` | Number of SMs |

Additional PyTorch GPU queries (separate API calls):

| Function | Returns |
|----------|---------|
| `torch.version.cuda` | CUDA version string (e.g. `12.4`) |
| `torch.backends.cudnn.version()` | cuDNN version |
| `torch.cuda.device_count()` | Number of GPUs |
| `torch.cuda.get_device_name()` | GPU name |
| `torch.cuda.get_arch_list()` | Supported CUDA architectures |

`torch.utils.collect_env.get_pretty_env_info()` aggregates: PyTorch version, CUDA version,
cuDNN version, GPU name, OS, GCC version, CMake version, libc version, Python version,
NumPy version. This is what lm-eval uses. It returns a multi-line string, not structured data.

**Key observation**: Compute capability (`major.minor`) and multi-processor count are
unique to PyTorch introspection. Other tools skip these. Compute capability is relevant
for understanding which GPU features are available (e.g., FP8 support, tensor core generation).

---

### 9. nvidia-smi queryable fields

**Source**: [nvidia-smi docs](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [nvidia-smi --help-query-gpu](https://gist.github.com/sansmoraxz/8a98d987f12d7edc983d611b8326fc67)

**Approach**: nvidia-smi exposes ~60+ queryable fields via `--query-gpu`. This is the
most comprehensive GPU metadata source available.

**Field categories** (subset relevant to reproducibility):

| Category | Fields |
|----------|--------|
| **Identity** | `name`/`gpu_name`, `serial`/`gpu_serial`, `uuid`/`gpu_uuid`, `index`, `count` |
| **Memory** | `memory.total`, `memory.reserved`, `memory.used`, `memory.free` |
| **Performance** | `clocks.current.graphics`, `clocks.current.sm`, `clocks.current.memory`, `clocks.max.graphics`, `clocks.max.sm`, `clocks.max.memory`, `pstate` (P0-P12), `utilization.gpu`, `utilization.memory` |
| **Power** | `power.draw`, `power.limit`, `enforced.power.limit`, `power.default_limit`, `power.min_limit`, `power.max_limit` |
| **Temperature** | `temperature.gpu`, `temperature.memory` |
| **PCIe** | `pci.bus_id`, `pcie.link.gen.current`, `pcie.link.gen.max`, `pcie.link.width.current`, `pcie.link.width.max` |
| **Driver** | `driver_version`, `vbios_version` |
| **InfoROM** | `inforom.img`, `inforom.oem`, `inforom.ecc`, `inforom.pwr` |

Also available: `persistence_mode`, `compute_mode`, `mig.mode.current`,
`ecc.mode.current`, `retired_pages.count`.

**Key observation**: nvidia-smi is the superset source. All other tools ultimately query a
subset of these fields (usually via pynvml, the Python bindings to the same NVML library).
Fields like `power.limit` vs `power.default_limit`, `pcie.link.gen.current` vs `max`,
and `persistence_mode` are directly relevant to energy measurement reproducibility.

---

### 10. pip freeze / conda environment capture

**Source**: [pip freeze docs](https://pip.pypa.io/en/stable/cli/pip_freeze/), [PEP 665 discussion](https://discuss.python.org/t/pep-665-take-2-a-file-format-to-list-python-dependencies-for-reproducibility-of-an-application/11736)

**Approach**: Standard Python practice for software environment reproducibility.

**Patterns observed across tools**:

| Tool | Method | Output |
|------|--------|--------|
| lm-eval | `torch.utils.collect_env.get_pretty_env_info()` | Human-readable blob (not parseable) |
| W&B | Not captured in metadata (relies on git + requirements.txt logging) | N/A |
| CodeCarbon | `python_version` only; no package list | Single field |
| optimum-benchmark | `python_version` only; no package list | Single field |
| MLPerf | `other_software_stack` free-text field | e.g. `TensorRT 10.8, CUDA 12.8, cuDNN 8.9.7, Driver 550.90` |
| MLflow | Not captured natively; user-logged artifact | Optional |

**Best practice** (from PEP 665, ML reproducibility literature):
- `pip list --format=freeze` (sorted) is the standard for pip environments
- `pip freeze` excludes pip/wheel/setuptools but handles editable installs differently
- `pip list --format=freeze` is more complete and predictable
- For conda: `conda env export --from-history` (cross-platform) or `conda list --export` (exact)
- Some tools store `requirements.txt` as an artifact rather than inline in results

**Key observation**: Only 1/7 tools (our v2.0 design) includes full `installed_packages` inline
in results. This is uncommon but defensible for energy measurement: the specific versions of
PyTorch, vLLM, CUDA toolkit, etc., directly affect inference performance and energy consumption.
The alternative (artifact-based) requires a separate file management system.

---

## Summary Table

| Field | MLflow | W&B | lm-eval | optimum-bench | MLPerf | Zeus | CodeCarbon | Our v1.x | Our v2.0 design |
|-------|--------|-----|---------|---------------|--------|------|------------|----------|-----------------|
| **GPU name** | - | Y | blob | Y | Y | Y | Y | Y | Y |
| **GPU VRAM** | - | Y | blob | Y | Y | - | - | Y | Y |
| **GPU count** | - | Y | - | Y | Y | - | Y | - | Y |
| **Compute capability** | - | - | blob | - | - | - | - | Y | - |
| **CUDA version** | - | - | blob | - | Y* | - | - | Y | Y |
| **Driver version** | - | - | blob | - | Y* | - | - | Y | Y |
| **cuDNN version** | - | - | blob | - | Y* | - | - | Y | - |
| **PCIe gen** | - | - | - | - | Y | - | - | Y | - |
| **MIG enabled** | - | - | - | - | - | - | - | Y | - |
| **Power limit (W)** | time-series | time-series | - | - | Y** | Y | - | Y | - |
| **Default power limit** | - | - | - | - | - | Y | - | Y | - |
| **Temperature (C)** | - | time-series | - | - | - | Y | - | Y | - |
| **Fan speed (%)** | - | - | - | - | - | - | - | Y | - |
| **CPU model** | plugin | Y | blob | Y | Y | - | Y | Y | - |
| **CPU count** | plugin | Y | - | Y | Y | - | Y | - | - |
| **CPU governor** | - | - | - | - | - | - | - | Y | - |
| **OS** | plugin | Y | blob | Y | Y | - | Y | Y | - |
| **RAM total** | plugin | Y | - | Y | Y | - | Y | - | - |
| **Python version** | - | Y | blob | Y | - | - | Y | - | Y |
| **Tool version** | - | - | Y | - | - | - | Y | - | Y |
| **pip packages** | artifact | - | - | - | - | - | - | - | Y |
| **Git commit** | Y | Y | Y | - | - | - | - | - | - |
| **Container detected** | - | - | - | - | - | - | - | Y | - |
| **Container runtime** | - | - | - | - | - | - | - | Y | - |
| **Hostname** | - | Y | - | - | - | - | - | - | - |
| **Timestamp** | Y | Y | Y | - | - | - | Y | Y | Y |

`blob` = embedded in `pretty_env_info` multi-line string, not individually addressable.
`Y*` = stored in free-text `other_software_stack` field, not structured.
`Y**` = stored in `hw_notes` free-text (e.g. `B200 TGP 1000W`).

---

## Recommended Field Set

Based on the peer evidence, fields fall into three tiers:

### Tier 1 — Universal (captured by 4+ tools; essential for any reproducibility claim)

| Field | v1.x | v2.0 design | Recommendation |
|-------|------|-------------|----------------|
| GPU name(s) | Y | Y | Keep. Use `list[str]` for multi-GPU. |
| GPU VRAM | Y | Y | Keep. Per-device `list[float]` in GB. |
| GPU count | - | Y | Keep. `len(gpu_names)` or explicit. |
| CUDA version | Y | Y | Keep. |
| Driver version | Y | Y | Keep. |
| OS | Y (platform) | - | **Add.** Present in 6/7 tools. Use `platform.platform()`. |
| CPU model | Y | - | **Add.** Present in 5/7 tools. |
| Python version | - | Y | Keep. |
| Tool version | - | Y | Keep. |
| Timestamp | Y | Y | Keep. |

### Tier 2 — Energy-measurement-specific (critical for our domain; 2-3 tools capture these)

| Field | v1.x | v2.0 design | Recommendation |
|-------|------|-------------|----------------|
| Power limit (W) | Y | - | **Add.** Zeus + nvidia-smi + our v1.x. Directly affects energy readings. |
| Default power limit (W) | Y | - | **Add.** Comparison shows whether limit was modified. |
| Temperature at start (C) | Y | - | **Add.** Thermal state is the primary uncontrolled variable. |
| Compute capability | Y | - | **Add.** Determines available instructions (FP8, tensor cores). PyTorch provides this. |
| cuDNN version | Y | - | **Add.** Affects inference kernel selection. |
| MIG enabled | Y | - | **Add.** MIG fundamentally changes GPU resource partitioning and power behaviour. |
| Container detected + runtime | Y | - | **Add.** Docker/podman affects NVML access and resource isolation. |

### Tier 3 — Valuable but lower priority (1-2 tools; nice-to-have)

| Field | v1.x | v2.0 design | Recommendation |
|-------|------|-------------|----------------|
| PCIe generation | Y | - | **Add.** Affects data transfer energy. nvidia-smi queryable. |
| CPU governor | Y | - | **Add.** `performance` vs `powersave` affects CPU energy. Linux-only. |
| CPU count | - | - | **Add.** 4 tools capture this. Trivial via `os.cpu_count()`. |
| RAM total | - | - | **Add.** 4 tools capture this. Trivial via `psutil` or `/proc/meminfo`. |
| Fan speed (%) | Y | - | Defer. Only our v1.x captures this. Low reproducibility value. |
| Installed packages | - | Y | **Keep but reconsider storage.** Only our v2.0 design does this inline. Consider: store hash of sorted pip freeze for comparison, full list as optional detail. |
| Hostname | - | - | Defer. Only W&B captures this. Privacy concern for published results. |
| Git commit | - | - | Defer. MLflow + W&B + lm-eval capture this, but it's session/development metadata, not measurement environment. |

---

## Recommendation

### Structure: Composed sub-models (v1.x approach) over flat (v2.0 design)

The v1.x code's five-sub-model composition (`GPUEnvironment`, `CUDAEnvironment`,
`ThermalEnvironment`, `CPUEnvironment`, `ContainerEnvironment`) is the better structure.

**Evidence**: MLPerf uses hierarchical categories (host CPU, accelerator, interconnect, software).
W&B groups metadata into hardware, system, execution sections. Only CodeCarbon uses a
fully flat structure, and it has 38 fields — demonstrating why flat becomes unwieldy.

optimum-benchmark uses a flat dict but with only 11 fields; it would not scale if they added
thermal/power fields.

### Merge: Take the union of v1.x and v2.0 design

The v2.0 `EnvironmentSnapshot` adds fields the v1.x model lacks (`python_version`,
`llenergymeasure_version`, `installed_packages`). The v1.x `EnvironmentMetadata` adds fields
the v2.0 design lacks (`compute_capability`, `pcie_gen`, `mig_enabled`, `cudnn_version`,
`power_limit_w`, `default_power_limit_w`, `temperature_c`, `fan_speed_pct`, `cpu_governor`,
`cpu_model`, `container.*`). The correct v2.0 model is the union, not either alone.

### Additions beyond both (peer-evidenced)

Based on the summary table, add:
- `os: str` (full platform string) — 6/7 tools capture this
- `cpu_count: int` — 4/7 tools capture this
- `ram_total_gb: float` — 4/7 tools capture this

### Defer

- `hostname` — privacy concern for published results
- `git_commit` — development metadata, not measurement environment
- `fan_speed_pct` — only v1.x captures; low reproducibility value
- Networking, firmware, storage fields from MLPerf — HPC/datacenter scope, not single-machine

### Package list strategy

Only our tool's v2.0 design includes full `installed_packages` inline. This is unusual but
justified: for energy measurement, the specific PyTorch/vLLM/TensorRT version directly affects
kernel selection and thus energy consumption. Store the sorted `pip list --format=freeze` output.
Consider also storing a SHA-256 hash of the sorted package list for quick comparison without
transmitting the full list.
