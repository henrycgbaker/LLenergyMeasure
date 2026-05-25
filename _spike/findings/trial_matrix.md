# Empirical trial matrix

_generated at 2026-05-25T14:46:07.642363+00:00_

_score files aggregated: 51_

## Per-cell matrix

| strategy | engine | version | bump | schema_recall | schema_prec | inv_recall | inv_prec | sev_acc | wall_s | energy_wh | failure_modes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | tensorrt | v0_19_0 | v-2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.2 | 0.00 | none |
| a | tensorrt | v0_20_0 | v-1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.1 | 0.00 | none |
| a | tensorrt | v0_21_0 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | tensorrt | v1_0_0 | v+1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.1 | 0.00 | none |
| a | tensorrt | v1_2_1 | v+major | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.1 | 0.00 | none |
| a | transformers | v4_55_4 | v-2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.7 | 0.08 | detectable |
| a | transformers | v4_56_2 | v-1 | 100.0% | 100.0% | 48.7% | 54.3% | 94.7% | 7.0 | 0.08 | none |
| a | transformers | v4_57_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | transformers | v4_57_6 | v+1 | 100.0% | 100.0% | 43.6% | 60.7% | 100.0% | 2.9 | 0.03 | none |
| a | transformers | v5_9_0 | v+major | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.8 | 0.01 | detectable |
| a | vllm | v0_19_1 | v+major | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.7 | 0.03 | detectable |
| a | vllm | v0_6_0 | v-2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.7 | 0.03 | detectable |
| a | vllm | v0_6_6_post1 | v-1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.9 | 0.03 | detectable |
| a | vllm | v0_7_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | vllm | v0_9_2 | v+1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.9 | 0.03 | detectable |
| b | tensorrt | v0_19_0 | v-2 | 4.7% | 6.9% | 16.1% | 13.5% | 100.0% | 1666.6 | 77.86 | silent |
| b | tensorrt | v0_20_0 | v-1 | 4.7% | 6.4% | 16.1% | 15.2% | 100.0% | 1678.0 | 78.55 | silent |
| b | tensorrt | v0_21_0 | active | 56.1% | 46.5% | 25.8% | 20.5% | 75.0% | 1372.2 | 66.44 | none |
| b | tensorrt | v1_0_0 | v+1 | 52.3% | 43.4% | 22.6% | 18.4% | 100.0% | 2248.0 | 101.85 | none |
| b | tensorrt | v1_2_1 | v+major | 50.5% | 36.7% | 19.4% | 15.0% | 100.0% | 2260.2 | 102.65 | none;silent |
| b | transformers | v4_55_4 | v-2 | 88.4% | 93.4% | 59.0% | 31.1% | 78.3% | 6526.6 | 361.09 | none |
| b | transformers | v4_56_2 | v-1 | 86.6% | 93.3% | 59.0% | 31.5% | 78.3% | 6516.0 | 360.71 | none |
| b | transformers | v4_57_3 | active | 83.0% | 93.9% | 56.4% | 43.1% | 77.3% | 1649.2 | 81.31 | none |
| b | transformers | v4_57_6 | v+1 | 83.0% | 93.9% | 59.0% | 45.1% | 60.9% | 6502.0 | 359.92 | none |
| b | transformers | v5_9_0 | v+major | 81.2% | 91.0% | 43.6% | 23.9% | 76.5% | 6319.2 | 349.00 | none |
| b | vllm | v0_19_1 | v+major | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 875.3 | 41.08 | silent |
| b | vllm | v0_6_0 | v-2 | 75.6% | 82.9% | 34.6% | 14.1% | 100.0% | 3969.0 | 208.96 | none |
| b | vllm | v0_6_6_post1 | v-1 | 57.8% | 77.2% | 34.6% | 15.5% | 100.0% | 4167.5 | 220.49 | none |
| b | vllm | v0_7_3 | active | 97.0% | 85.1% | 38.5% | 15.2% | 100.0% | 1414.3 | 67.93 | none |
| b | vllm | v0_9_2 | v+1 | 87.4% | 63.4% | 30.8% | 13.3% | 100.0% | 4006.4 | 210.95 | none |
| b_8b | transformers | v4_57_3 | active | 85.7% | 93.2% | 35.7% | 16.1% | 100.0% | 412.6 | 4.93 | none |
| c | transformers | v4_57_3 | active | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | key_absent |
| d-ab | tensorrt | v0_19_0 | v-2 | 100.0% | 100.0% | 100.0% | 91.2% | 100.0% | 19.7 | 1.09 | none |
| d-ab | tensorrt | v0_20_0 | v-1 | 100.0% | 100.0% | 100.0% | 91.2% | 100.0% | 63.5 | 3.65 | none |
| d-ab | tensorrt | v0_21_0 | active | 100.0% | 100.0% | 100.0% | 79.5% | 100.0% | 207.5 | 10.94 | none |
| d-ab | tensorrt | v1_0_0 | v+1 | 100.0% | 100.0% | 100.0% | 79.5% | 100.0% | 97.5 | 5.64 | none |
| d-ab | tensorrt | v1_2_1 | v+major | 100.0% | 100.0% | 100.0% | 88.6% | 100.0% | 45.0 | 2.56 | none |
| d-ab | transformers | v4_55_4 | v-2 | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 455.9 | 21.24 | none |
| d-ab | transformers | v4_56_2 | v-1 | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 473.7 | 22.25 | none |
| d-ab | transformers | v4_57_3 | active | 100.0% | 100.0% | 100.0% | 93.3% | 100.0% | 20.1 | 0.84 | none |
| d-ab | transformers | v4_57_6 | v+1 | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 490.5 | 23.32 | none |
| d-ab | transformers | v5_9_0 | v+major | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 507.7 | 24.29 | none |
| d-ab | vllm | v0_19_1 | v+major | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 101.4 | 5.91 | none |
| d-ab | vllm | v0_6_0 | v-2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 343.8 | 20.20 | none |
| d-ab | vllm | v0_6_6_post1 | v-1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 417.3 | 24.52 | none |
| d-ab | vllm | v0_7_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 433.6 | 19.38 | none |
| d-ab | vllm | v0_9_2 | v+1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 144.4 | 8.48 | none |
| e6 | transformers | v4_57_3 | active | 83.0% | 98.9% | 56.4% | 38.6% | 90.9% | 1256.0 | 0.00 | none |
| e6 | vllm | v0_7_3 | active | 97.0% | 85.1% | 30.8% | 17.4% | 100.0% | 1048.5 | 0.00 | none |
| e9 | transformers | v4_57_3 | active | 83.0% | 98.9% | 33.3% | 40.6% | 92.3% | 901.8 | 0.00 | none |
| h6 | transformers | v4_57_3 | active | 75.0% | 94.4% | 12.8% | 31.2% | 60.0% | 526.4 | 0.00 | none;silent |

## Per-strategy aggregates

| strategy | cells | schema_recall_mean | schema_recall_median | inv_recall_mean | inv_recall_median | wall_mean_s | energy_mean_wh | crashes |
|---|---|---|---|---|---|---|---|---|
| a | 15 | 60.0% | 100.0% | 52.8% | 48.7% | 1.9 | 0.02 | 0 |
| b | 15 | 60.6% | 75.6% | 34.4% | 34.6% | 3411.4 | 179.25 | 0 |
| b_8b | 1 | 85.7% | 85.7% | 35.7% | 35.7% | 412.6 | 4.93 | 0 |
| c | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | 0 |
| d-ab | 15 | 100.0% | 100.0% | 100.0% | 100.0% | 254.8 | 12.96 | 0 |
| e6 | 2 | 90.0% | 90.0% | 43.6% | 43.6% | 1152.3 | 0.00 | 0 |
| e9 | 1 | 83.0% | 83.0% | 33.3% | 33.3% | 901.8 | 0.00 | 0 |
| h6 | 1 | 75.0% | 75.0% | 12.8% | 12.8% | 526.4 | 0.00 | 0 |

## Per-engine aggregates

| engine | cells | schema_recall_mean | inv_recall_mean | wall_mean_s |
|---|---|---|---|---|
| tensorrt | 15 | 77.9% | 73.3% | 643.9 |
| transformers | 20 | 77.5% | 55.4% | 1628.8 |
| vllm | 16 | 63.4% | 48.1% | 1058.3 |

## Per-bump-distance aggregates

| bump | cells | schema_recall_mean | inv_recall_mean | pass_through_mean |
|---|---|---|---|---|
| v-2 | 9 | 63.2% | 56.6% | - |
| v-1 | 9 | 72.1% | 62.0% | - |
| active | 15 | 84.0% | 59.3% | - |
| v+1 | 9 | 80.3% | 61.8% | - |
| v+major | 9 | 59.1% | 51.4% | - |

## Adjacent observations (deduped per strategy)

### strategy a

- strategy_a bumped (tensorrt): stdout=wrote 38 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/tensorrt/v0_19_0/invariants.proposed.yaml
- CAVEAT (post-trial audit): the tensorrt active miner reads from a hardcoded _DEFAULT_SOURCE_ROOT (=/tmp/trt-llm-0.21.0/tensorrt_llm), NOT from the source-only venv passed via PYTHONPATH. The walker is AST-only (no import-time landmark check), so PYTHONPATH override has NO effect. Result: this 'bumped' (a) cell actually re-extracted invariants from the ACTIVE source - identical 31 candidates to v0_21_0. The 1.00 recall/precision is an artefact of substrate wiring, not honest bumped-cell performance. Brittleness mode: MINER_VERSION_BLIND - the tensorrt miner CANNOT be steered to bumped source via the current dispatcher pattern. Same class as vllm's msgspec-import brittleness, expressed differently.
- strategy_a bumped (tensorrt): stdout=wrote 38 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/tensorrt/v0_20_0/invariants.proposed.yaml
- strategy_a: reusing canonical engine_versions outputs for a/tensorrt/v0_21_0
- strategy_a bumped (tensorrt): stdout=wrote 38 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/tensorrt/v1_0_0/invariants.proposed.yaml
- strategy_a bumped (tensorrt): stdout=wrote 38 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/tensorrt/v1_2_1/invariants.proposed.yaml
- strategy_a bumped: miner subprocess returncode=1; stderr_tail=Traceback (most recent call last):
  File "<string>", line 7, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py", line 1462, in walk_transformers
    import transformers.generation.configuration_utils as gen_mod  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/__init__.py", line 27, in <module>
    from . import dependency_versions_check
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/dependency_versions_check.py", line 57, in <module>
    require_version_core(deps[pkg])
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/utils/versions.py", line 117, in require_version_core
    return require_version(requirement, hint)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/utils/versions.py", line 111, in require_version
    _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/utils/versions.py", line 44, in _compare_versions
    raise ImportError(
ImportError: tokenizers>=0.21,<0.22 is required for a normal functioning of this module, but found tokenizers==0.22.2.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main

- strategy_a bumped: failure_mode=miner_runtime_error
- strategy_a bumped: subprocess stdout=wrote 38 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/transformers/v4_56_2/invariants.proposed.yaml
- strategy_a: reusing canonical engine_versions outputs for a/transformers/v4_57_3
- strategy_a bumped: subprocess stdout=wrote 28 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/transformers/v4_57_6/invariants.proposed.yaml
- strategy_a bumped: miner subprocess returncode=1; stderr_tail=Traceback (most recent call last):
  File "<string>", line 7, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py", line 1462, in walk_transformers
    import transformers.generation.configuration_utils as gen_mod  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/dependency_versions_check.py", line 16, in <module>
    from .utils.versions import require_version, require_version_core
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/utils/__init__.py", line 76, in <module>
    from .hub import (
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/utils/hub.py", line 29, in <module>
    from huggingface_hub import (
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub' (/home/h.baker@hertie-school.lan/miniforge3/lib/python3.12/site-packages/huggingface_hub/__init__.py)

- strategy_a bumped (vllm): miner subprocess returncode=1; stderr_tail=MINER_CRASH:MinerLandmarkMissingError:Miner landmark missing: vllm.sampling_params (module not importable: No module named 'msgspec')
Traceback (most recent call last):
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 974, in _check_landmarks
    module = __import__(target.module_path, fromlist=[target.class_name])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_vllm_v0_19_1_venv/src/vllm/sampling_params.py", line 12, in <module>
    import msgspec
ModuleNotFoundError: No module named 'msgspec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<string>", line 8, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 1017, in walk_vllm_static
    installed_version, abs_paths = _check_landmarks()
                                   ^^^^^^^^^^^^^^^^^^
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 976, in _check_landmarks
    raise MinerLandmarkMissingError(
scripts.engine_producers._base.MinerLandmarkMissingError: Miner landmark missing: vllm.sampling_params (module not importable: No module named 'msgspec')

- strategy_a bumped: failure_mode=detectable
- strategy_a bumped (vllm): miner subprocess returncode=1; stderr_tail=lm not importable)
Traceback (most recent call last):
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 967, in _check_landmarks
    import vllm  # type: ignore
    ^^^^^^^^^^^
  File "/tmp/trial_vllm_v0_6_0_venv/src/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/tmp/trial_vllm_v0_6_0_venv/src/vllm/engine/arg_utils.py", line 11, in <module>
    from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig,
  File "/tmp/trial_vllm_v0_6_0_venv/src/vllm/config.py", line 12, in <module>
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
  File "/tmp/trial_vllm_v0_6_0_venv/src/vllm/model_executor/__init__.py", line 3, in <module>
    from vllm.model_executor.sampling_metadata import (SamplingMetadata,
  File "/tmp/trial_vllm_v0_6_0_venv/src/vllm/model_executor/sampling_metadata.py", line 8, in <module>
    from vllm.sampling_params import SamplingParams, SamplingType
  File "/tmp/trial_vllm_v0_6_0_venv/src/vllm/sampling_params.py", line 7, in <module>
    import msgspec
ModuleNotFoundError: No module named 'msgspec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<string>", line 8, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 1017, in walk_vllm_static
    installed_version, abs_paths = _check_landmarks()
                                   ^^^^^^^^^^^^^^^^^^
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 969, in _check_landmarks
    raise MinerLandmarkMissingError("vllm.__init__", detail="vllm not importable") from exc
scripts.engine_producers._base.MinerLandmarkMissingError: Miner landmark missing: vllm.__init__ (vllm not importable)

- strategy_a bumped (vllm): miner subprocess returncode=1; stderr_tail= line 7, in <module>
    from vllm.distributed import get_tensor_model_parallel_rank
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/distributed/__init__.py", line 1, in <module>
    from .communication_op import *
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/distributed/communication_op.py", line 6, in <module>
    from .parallel_state import get_tp_group
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/distributed/parallel_state.py", line 38, in <module>
    import vllm.distributed.kv_transfer.kv_transfer_agent as kv_transfer
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/distributed/kv_transfer/kv_transfer_agent.py", line 15, in <module>
    from vllm.distributed.kv_transfer.kv_connector.factory import (
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/distributed/kv_transfer/kv_connector/factory.py", line 3, in <module>
    from .base import KVConnectorBase
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/distributed/kv_transfer/kv_connector/base.py", line 14, in <module>
    from vllm.sequence import IntermediateTensors
  File "/tmp/trial_vllm_v0_6_6_post1_venv/src/vllm/sequence.py", line 13, in <module>
    import msgspec
ModuleNotFoundError: No module named 'msgspec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<string>", line 8, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 1017, in walk_vllm_static
    installed_version, abs_paths = _check_landmarks()
                                   ^^^^^^^^^^^^^^^^^^
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 969, in _check_landmarks
    raise MinerLandmarkMissingError("vllm.__init__", detail="vllm not importable") from exc
scripts.engine_producers._base.MinerLandmarkMissingError: Miner landmark missing: vllm.__init__ (vllm not importable)

- strategy_a: reusing canonical engine_versions outputs for a/vllm/v0_7_3
- strategy_a bumped (vllm): miner subprocess returncode=1; stderr_tail=MINER_CRASH:MinerLandmarkMissingError:Miner landmark missing: vllm.sampling_params (module not importable: No module named 'msgspec')
Traceback (most recent call last):
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 974, in _check_landmarks
    module = __import__(target.module_path, fromlist=[target.class_name])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_vllm_v0_9_2_venv/src/vllm/sampling_params.py", line 10, in <module>
    import msgspec
ModuleNotFoundError: No module named 'msgspec'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<string>", line 8, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 1017, in walk_vllm_static
    installed_version, abs_paths = _check_landmarks()
                                   ^^^^^^^^^^^^^^^^^^
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/vllm/v0_7_3/producers/static_invariant_miner.py", line 976, in _check_landmarks
    raise MinerLandmarkMissingError(
scripts.engine_producers._base.MinerLandmarkMissingError: Miner landmark missing: vllm.sampling_params (module not importable: No module named 'msgspec')


### strategy b

- multipass summary: pass2_dropped=5, pass3_added=18, total_invariants=37
- strategy_b: engine='tensorrt', schema_chunks=7, invariants_chunks=7, schema_wall=596.3s, invariants_wall=1069.7s, multipass=True
- multipass summary: pass2_dropped=5, pass3_added=15, total_invariants=33
- strategy_b: engine='tensorrt', schema_chunks=7, invariants_chunks=7, schema_wall=679.4s, invariants_wall=998.1s, multipass=True
- chunk 'base_llm_args_validators_bottom' pass2 flag (non-applied): id='tensorrt_llm_enable_lora_ignored_when_lora_config_provided_for_pytorch_backend' reason="Source has a more specific condition `self.enable_lora and self.lora_config is not None and self.backend in ['pytorch', '_autodeploy']` which is not fully captured by the invariant." fix='correct_predicate:exact'
- chunk 'base_llm_args_validators_bottom' pass2 flag (non-applied): id='tensorrt_llm_both_lora_dir_and_lora_target_modules_empty' reason='Source has a more specific condition `len(self.lora_config.lora_dir) == 0 and len(self.lora_config.lora_target_modules) == 0` which is not fully captured by the invariant.' fix='correct_predicate:exact'
- multipass summary: pass2_dropped=0, pass3_added=18, total_invariants=39
- strategy_b: engine='tensorrt', schema_chunks=7, invariants_chunks=7, schema_wall=791.1s, invariants_wall=580.5s, multipass=True
- Phase 2.6 RUBRIC CORRECTION: namespace canonicalisation (tensorrt_llm.X -> tensorrt.X) applied at identity-extraction time. Re-scored against existing trial_runs output (no LLM re-extraction).
- chunk 'base_llm_args_validators_top' pass2 flag (non-applied): id='tensorrt_llm_max_batch_size_gt_build_config_max_batch_size' reason='Source has `if self.max_batch_size > self.build_config.max_batch_size` but invariant says `> 1`' fix='correct_predicate:cross-field check'
- chunk 'base_llm_args_validators_top' pass2 flag (non-applied): id='tensorrt_llm_max_num_tokens_gt_build_config_max_num_tokens' reason='Same as above, source has `if self.max_num_tokens > self.build_config.max_num_tokens` but invariant says `> 1`' fix='correct_predicate:cross-field check'
- multipass summary: pass2_dropped=1, pass3_added=19, total_invariants=38
- strategy_b: engine='tensorrt', schema_chunks=7, invariants_chunks=7, schema_wall=971.6s, invariants_wall=1275.8s, multipass=True
- chunk 'base_llm_args_validators_bottom' pass2 flag (non-applied): id='tensorrt_lora_prefetch_dir_set_while_not_supported' reason='Source raises ValueError, but invariant has severity warning.' fix='correct_severity:error'
- chunk 'trt_llm_args_validators' pass2 flag (non-applied): id='tensorrt_max_seq_len_overridden_by_build_config' reason="Source checks for inequality, but invariant emits '!=' which is incorrect." fix='correct_predicate:not_equal'
- chunk 'trt_llm_args_validators' pass2 flag (non-applied): id='tensorrt_max_beam_width_overridden_by_build_config' reason="Source checks for inequality, but invariant emits '!=' which is incorrect." fix='correct_predicate:not_equal'
- chunk 'trt_llm_args_validators' pass2 flag (non-applied): id='tensorrt_max_input_len_overridden_by_build_config' reason="Source checks for inequality, but invariant emits '!=' which is incorrect." fix='correct_predicate:not_equal'
- multipass summary: pass2_dropped=0, pass3_added=16, total_invariants=40
- strategy_b: engine='tensorrt', schema_chunks=7, invariants_chunks=7, schema_wall=1056.8s, invariants_wall=1202.7s, multipass=True
- chunk 'bitsandbytes_config_invariants' pass2 flag (non-applied): id='transformers_bnb_4bit_quant_storage_not_string_or_torch_dtype' reason="Source allows `bnb_4bit_quant_storage` to be a string or torch.dtype, but invariant says it's not." fix='correct_predicate:type_is'
- chunk 'validate_section_00_1._Validation_of_individual_attributes' pass2 flag (non-applied): id='transformers_early_stopping_not_in_allowlist' reason="Source does not validate early_stopping against an allowlist with 'never' as a valid value." fix='correct_kwargs_positive'
- chunk 'validate_section_01_1.1._Decoding_attributes' pass2 flag (non-applied): id='transformers_pad_token_id_lt_zero' reason='Source raises minor issue for pad_token_id < 0, but invariant severity is warning.' fix='correct_severity:error'
- invariants chunk 'validate_section_01_1.1._Decoding_attributes' pass3_extend: extraction failed; modes=['parse_failure_after_retries']
- chunk 'validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode' pass2 flag (non-applied): id='transformers_repetition_penalty_set_when_dola_layers_and_num_beams_one' reason='Source checks for `repetition_penalty < 1.2`, but invariant has no such condition.' fix='correct_predicate:range'
- chunk 'validate_section_08_2.3._detect_incorrect_parameterization_specific_to_advanced_' pass2 flag (non-applied): id='transformers_num_beams_not_divisible_by_num_beam_groups_when_group_beam_search' reason='source code requires `num_beams` to be divisible by `num_beam_groups`, but invariant says the opposite' fix='correct_predicate:not_divisible_by -> divisible_by'
- multipass summary: pass2_dropped=6, pass3_added=27, total_invariants=74
- strategy_b: schema_chunks=5, invariants_chunks=15, schema_wall=1579.9s, invariants_wall=4945.8s
- chunk 'validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode' pass2 flag (non-applied): id='transformers_num_beam_groups_set_when_num_beams_one' reason='source checks for self.num_beam_groups != 1, not just presence' fix='correct_predicate:not_equal'
- chunk 'validate_section_09_2.4._check_num_return_sequences' pass2 flag (non-applied): id='transformers_num_return_sequences_gt_num_beams' reason='Source checks `num_return_sequences > num_beams` but invariant says `> 1`.' fix='correct_predicate:gt'
- multipass summary: pass2_dropped=6, pass3_added=28, total_invariants=73
- strategy_b: schema_chunks=5, invariants_chunks=15, schema_wall=1314.5s, invariants_wall=5200.5s
- chunk 'bitsandbytes_config_invariants' pass2 flag (non-applied): id='transformers_bnb_4bit_compute_dtype_not_string_or_torch_dtype' reason='Source allows bnb_4bit_compute_dtype to be a string or torch.dtype, but invariant only checks for not being a string.' fix='correct_predicate:type_is_not_str_or_torch_dtype'
- multipass summary: pass2_dropped=0, pass3_added=23, total_invariants=51
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=517.3s, invariants_wall=1125.5s
- multipass summary: pass2_dropped=0, pass3_added=26, total_invariants=51
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=1772.3s, invariants_wall=4728.9s
- chunk 'validate_section_08_2.4._check_num_return_sequences' pass2 flag (non-applied): id='transformers_num_return_sequences_needs_beam_search_or_sampling' reason='Source checks for `num_return_sequences > 1` without beam search or sampling, but also requires `do_sample != True`, which is not captured by the invariant.' fix='correct_predicate:range'
- multipass summary: pass2_dropped=11, pass3_added=27, total_invariants=71
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=1407.5s, invariants_wall=4910.7s
- multipass summary: pass2_dropped=1, pass3_added=2, total_invariants=4
- strategy_b: engine='vllm', schema_chunks=1, invariants_chunks=1, schema_wall=13.6s, invariants_wall=861.4s, multipass=True
- chunk 'model_config_verify_quantization' pass2 flag (non-applied): id='vllm_awq_quantization_not_optimized' reason='Source raises a warning for non-optimized quantization methods, but the invariant does not match the exact list of optimized methods.' fix='correct_predicate:not_in'
- invariants chunk 'cache_config_invariants' pass2_verify: extraction failed; modes=['parse_failure_after_retries']; pass1 unchanged
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_num_lookahead_slots_lt_zero' reason='Source checks for `num_lookahead_slots < 0`, but invariant has `< 1` in the match.' fix='correct_predicate:ge'
- chunk 'parallel_config_invariants' pass2 flag (non-applied): id='vllm_distributed_executor_backend_not_in_allowlist' reason='Source allows custom ExecutorBase subclass, but invariant does not.' fix='correct_predicate:not_in_or_subclass'
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_max_prompt_adapters_lt_1' reason='Source checks for `max_prompt_adapters < 1` but allows 0, invariant should be `== 0`' fix='correct_predicate:not_equal_to_zero'
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_max_prompt_adapter_token_eq_0' reason='Invariant is too strict, source only raises if `max_prompt_adapter_token == 0`, not `< 0`' fix='correct_predicate:not_equal_to_zero'
- multipass summary: pass2_dropped=1, pass3_added=22, total_invariants=64
- strategy_b: engine='vllm', schema_chunks=7, invariants_chunks=9, schema_wall=1309.7s, invariants_wall=2658.6s, multipass=True
- chunk 'sampling_params_invariants' pass2 flag (non-applied): id='vllm_stop_cannot_contain_empty_string' reason='Source checks for `any(not stop_str for stop_str in self.stop)`, not just presence of empty string.' fix='correct_predicate:not_in'
- chunk 'sampling_params_invariants' pass2 flag (non-applied): id='vllm_best_of_must_equal_n_to_use_output_kind_delta' reason='Source checks for `self.best_of != self._real_n` and `self.output_kind == RequestOutputKind.DELTA`, but invariant does not account for `_real_n`.' fix='correct_predicate:not_equal'
- invariants chunk 'model_config_verify_quantization' pass2_verify: extraction failed; modes=['parse_failure_after_retries']; pass1 unchanged
- chunk 'cache_config_invariants' pass2 flag (non-applied): id='vllm_prefix_caching_with_sliding_window' reason='Source raises NotImplementedError, not ValueError.' fix='correct_severity:error'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_num_lookahead_slots_lt_zero' reason='Source checks for `num_lookahead_slots < 0` but invariant has `< 1`.' fix='correct_predicate:range'
- invariants chunk 'parallel_config_invariants' pass2_verify: extraction failed; modes=['parse_failure_after_retries']; pass1 unchanged
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_pool_type_not_in_allowlist' reason='Source allows pool_type to be a type, not just "ray".' fix='correct_predicate:not_in_or_isinstance'
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_extra_config_not_dict' reason='Source checks for isinstance(self.extra_config, dict), not just type.' fix='correct_predicate:type_is'
- multipass summary: pass2_dropped=0, pass3_added=12, total_invariants=58
- strategy_b: engine='vllm', schema_chunks=7, invariants_chunks=10, schema_wall=1351.9s, invariants_wall=2814.8s, multipass=True
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_num_partial_prefills_lt_1' reason='Source checks for `max_num_partial_prefills < 1` but allows it to be equal to 1.' fix='correct_predicate:not_equal_or_less_than'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills' reason='Source checks for `max_long_partial_prefills < 1 or max_long_partial_prefills > max_num_partial_prefills` but allows it to be equal to max_num_partial_prefills.' fix='correct_predicate:not_equal_or_less_than_and_not_greater_than'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled' reason='Source checks for `max_num_partial_prefills > 1 and not chunked_prefill_enabled` but allows it to be equal to 1 without chunked_prefill_enabled.' fix='correct_predicate:not_equal_or_greater_than_and_not_equal'
- chunk 'parallel_config_invariants' pass2 flag (non-applied): id='vllm_tpu_backend_not_ray_for_distributed_inference' reason='Source sets distributed_executor_backend to "ray" when current_platform.device_type is "tpu" and world_size > 1, but does not raise an error for other backends.' fix='correct_predicate:not_equal'
- multipass summary: pass2_dropped=1, pass3_added=18, total_invariants=66
- strategy_b: engine='vllm', schema_chunks=7, invariants_chunks=10, schema_wall=516.6s, invariants_wall=896.8s, multipass=True
- chunk 'sampling_params_invariants' pass2 flag (non-applied): id='vllm_top_k_must_be_non_negative_integer' reason='source allows -1 as disabled, but invariant does not' fix='correct_predicate:range'
- chunk 'sampling_params_invariants' pass2 flag (non-applied): id='vllm_stop_strings_are_only_supported_when_detokenize_is_true' reason='source also checks for presence of stop strings, but invariant does not' fix='correct_kwargs_positive'
- chunk 'model_config_verify_quantization' pass2 flag (non-applied): id='vllm_quantization_method_override_not_in_allowlist' reason='Source checks for overrides in a specific order, but invariant does not reflect this.' fix='correct_predicate:not_equal'
- chunk 'cache_config_invariants' pass2 flag (non-applied): id='vllm_cache_dtype_not_in_allowlist' reason='Source checks for `self.cache_dtype in get_args(CacheDType)` but also allows `"auto"`, which is not accounted for in the invariant.' fix='correct_predicate:not_in_or_equal_to_auto'
- chunk 'cache_config_invariants' pass2 flag (non-applied): id='vllm_sliding_window_not_supported_with_prefix_caching' reason="Source raises `NotImplementedError` instead of `ValueError` when sliding window is used with prefix caching, contradicting the invariant's severity." fix='correct_severity:dormant'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills' reason='Source checks for `< 1 or > max_num_partial_prefills`, but invariant only checks `> 2`.' fix='correct_predicate:range'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled' reason='Severity is error, but source raises ValueError with a different message.' fix='correct_severity:warning'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_long_prefill_token_threshold_gt_max_model_len' reason='Source checks for `> max_model_len`, but invariant only checks `> 8192`.' fix='correct_predicate:exact'
- chunk 'parallel_config_invariants' pass2 flag (non-applied): id='vllm_num_redundant_experts_neq_0_when_not_enable_eplb' reason='Source raises ValueError when num_redundant_experts != 0 and enable_eplb is False, but invariant says != 0 is incorrect.' fix='correct_predicate:not_equal'
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_max_cpu_loras_lt_max_loras' reason='Source checks for `max_cpu_loras < self.max_loras`, but invariant only checks for `< 1`.' fix='correct_predicate:not_less_than'
- multipass summary: pass2_dropped=1, pass3_added=9, total_invariants=60
- strategy_b: engine='vllm', schema_chunks=7, invariants_chunks=10, schema_wall=1441.7s, invariants_wall=2563.8s, multipass=True

### strategy b_8b

- invariants chunk 'validate_section_03_1.3._Performance_attributes': extraction failed; modes=['parse_failure_after_retries']
- invariants chunk 'validate_section_08_2.4._check_num_return_sequences': parsed but yielded 0 unique invariants
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=228.3s, invariants_wall=178.3s

### strategy c

- cell crashed: KeyAbsentError: ANTHROPIC_API_KEY not set; strategy (c) cells are skipped.

### strategy d-ab

- strategy_d_ab on tensorrt: extension=3, flagged_spurious=2, merged_total=38, elapsed=19.3s
- strategy_d_ab on tensorrt: extension=3, flagged_spurious=1, merged_total=38, elapsed=63.1s
- strategy_d_ab on tensorrt: extension=8, flagged_spurious=1, merged_total=43, elapsed=207.1s
- Phase 2.6 RUBRIC CORRECTION: namespace canonicalisation (tensorrt_llm.X -> tensorrt.X) applied at identity-extraction time. Re-scored against existing trial_runs output (no LLM re-extraction).
- strategy_d_ab on tensorrt: extension=8, flagged_spurious=1, merged_total=43, elapsed=97.1s
- strategy_d_ab on tensorrt: extension=4, flagged_spurious=1, merged_total=39, elapsed=44.6s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=455.4s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=473.2s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=13.9s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=490.0s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=507.2s
- strategy_d_ab on vllm: extension=0, flagged_spurious=0, merged_total=26, elapsed=101.1s
- hybrid (d-ab) for vllm: extraction failed; modes=['parse_failure_after_retries']
- strategy_d_ab on vllm: extension=0, flagged_spurious=0, merged_total=26, elapsed=343.4s
- strategy_d_ab on vllm: extension=0, flagged_spurious=0, merged_total=26, elapsed=416.9s
- strategy_d_ab on vllm: extension=0, flagged_spurious=0, merged_total=26, elapsed=433.2s
- strategy_d_ab on vllm: extension=2, flagged_spurious=1, merged_total=28, elapsed=143.9s

### strategy e6

- pattern=e6 engine=transformers schema_wall=497.5s invariants_wall=758.0s total_wall=1256.0s
- e6_field_anchor: 3 classes, 82 declared fields
- invariants chunk 'generation_config_init_invariants': emitted 21 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'bitsandbytes_config_invariants': emitted 12 unique; anchor_classes=['BitsAndBytesConfig']
- invariants chunk 'validate_section_00_1._Validation_of_individual_attributes': emitted 8 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_01_1.1._Decoding_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_02_1.2._Cache_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_03_1.3._Performance_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_04_1.4._Watermarking_attributes': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_05_2._Validation_of_attribute_combinations': emitted 2 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl': emitted 5 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_08_2.4._check_num_return_sequences': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_09_2.5._check_cache_related_arguments': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_10_2.6._other_incorrect_combinations': emitted 1 unique; anchor_classes=['GenerationConfig']
- invariants chunk 'validate_section_11_3._Check_common_issue_passing_generate_arguments_inside_the_': emitted 1 unique; anchor_classes=['GenerationConfig']
- pattern=e6 engine=vllm schema_wall=516.9s invariants_wall=531.2s total_wall=1048.5s
- e6_field_anchor: 15 classes, 249 declared fields
- invariants chunk 'sampling_params_invariants': emitted 18 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'guided_decoding_params_invariants': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_tokenizer_mode': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_quantization': emitted 2 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_cuda_graph': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'model_config_verify_bnb_config': emitted 1 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'cache_config_invariants': emitted 3 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'scheduler_config_invariants': emitted 8 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'parallel_config_invariants': emitted 3 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']
- invariants chunk 'lora_prompt_adapter_invariants': emitted 8 unique; anchor_classes=['ModelConfig', 'CacheConfig', 'ParallelConfig', 'SchedulerConfig', 'EngineArgs', 'LoRAConfig', 'DeviceConfig', 'DecodingConfig', 'ObservabilityConfig', 'LoadConfig', 'PromptAdapterConfig', 'TokenizerPoolConfig', 'SamplingParams', 'BeamSearchParams', 'GuidedDecodingParams']

### strategy e9

- pattern=e9 engine=transformers schema_wall=497.4s invariants_wall=403.9s total_wall=901.8s
- invariants chunk 'generation_config_init_invariants': emitted 0 unique (cumulative total so far: 0)
- invariants chunk 'bitsandbytes_config_invariants': emitted 12 unique (cumulative total so far: 12)
- invariants chunk 'validate_section_00_1._Validation_of_individual_attributes': emitted 1 unique (cumulative total so far: 13)
- invariants chunk 'validate_section_01_1.1._Decoding_attributes': emitted 3 unique (cumulative total so far: 16)
- invariants chunk 'validate_section_02_1.2._Cache_attributes': emitted 1 unique (cumulative total so far: 17)
- invariants chunk 'validate_section_03_1.3._Performance_attributes': emitted 1 unique (cumulative total so far: 18)
- invariants chunk 'validate_section_04_1.4._Watermarking_attributes': emitted 0 unique (cumulative total so far: 18)
- invariants chunk 'validate_section_05_2._Validation_of_attribute_combinations': emitted 6 unique (cumulative total so far: 24)
- invariants chunk 'validate_section_06_2.1._detect_sampling_only_parameterization_when_not_in_sampl': emitted 1 unique (cumulative total so far: 25)
- invariants chunk 'validate_section_07_2.2._detect_beam_only_parameterization_when_not_in_beam_mode': emitted 1 unique (cumulative total so far: 26)
- invariants chunk 'validate_section_08_2.4._check_num_return_sequences': emitted 1 unique (cumulative total so far: 27)
- invariants chunk 'validate_section_09_2.5._check_cache_related_arguments': emitted 3 unique (cumulative total so far: 30)
- invariants chunk 'validate_section_10_2.6._other_incorrect_combinations': emitted 1 unique (cumulative total so far: 31)
- invariants chunk 'validate_section_11_3._Check_common_issue_passing_generate_arguments_inside_the_': emitted 1 unique (cumulative total so far: 32)

### strategy h6

- pattern=h6 engine=transformers schema_wall=257.4s invariants_wall=268.8s total_wall=526.4s
- h6_schema_source_chars=46594
- h6_invariants_source_chars=33440
- h6 invariants: emitted 16 unique entries
