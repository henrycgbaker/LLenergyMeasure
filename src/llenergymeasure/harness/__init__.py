"""MeasurementHarness - owns the measurement lifecycle for any EnginePlugin.

The harness extracts the ~600 lines of identical measurement infrastructure
duplicated across transformers.py and vllm.py into a single location. Engines
become thin plugins implementing the 4-method EnginePlugin protocol.

The facade lives in ``llenergymeasure.harness.measurement``; the lifecycle is
split across phase modules - ``lifecycle`` (per-model-load phases), ``window``
(the measured window over ``bracket``), ``result_assembly`` (source-branched
result assembly), ``measurement_warnings`` (warnings generation +
orchestration), and ``staging`` (temp-dir sidecar writes). This package init
re-exports the public surface plus the module-level helpers that tests patch.
Note: patching a name on this package does NOT affect the phase modules'
globals - tests that patch module globals must target the module that owns the
name (see each phase module's docstring for its use site).
"""

from __future__ import annotations

from llenergymeasure.harness.bracket import PowerThermalSampler

# Private helpers are re-exported with redundant aliases so legacy
# ``from llenergymeasure.harness import _cuda_sync`` keeps working. They are
# intentionally absent from ``__all__`` (private surface). Tests that PATCH these
# as module globals must target the OWNING module (``bracket`` for _cuda_sync,
# ``window`` for the capture helper, ``measurement_warnings`` for
# _check_persistence_mode).
from llenergymeasure.harness.bracket import (
    _cuda_sync as _cuda_sync,
)
from llenergymeasure.harness.lifecycle import (
    collect_environment_snapshot,
    collect_environment_snapshot_async,
    measure_baseline_power,
)
from llenergymeasure.harness.measurement import MeasurementHarness
from llenergymeasure.harness.measurement_warnings import (
    _check_persistence_mode as _check_persistence_mode,
)
from llenergymeasure.harness.measurement_warnings import (
    collect_measurement_warnings,
)
from llenergymeasure.harness.staging import write_timeseries_parquet
from llenergymeasure.harness.traffic import (
    ArrivalSchedule,
    IssuerReport,
    OpenLoopPoissonSource,
    RequestRecord,
    ShapeSource,
    TrafficSource,
    Transport,
    build_schedule,
)
from llenergymeasure.harness.window import (
    _capture_observed_params_into_output as _capture_observed_params_into_output,
)
from llenergymeasure.harness.window import (
    estimate_flops_palm,
    estimate_flops_palm_from_config,
)
from llenergymeasure.serving.transport import RequestShape, require_httpx

__all__ = [
    "ArrivalSchedule",
    "IssuerReport",
    "MeasurementHarness",
    "OpenLoopPoissonSource",
    "PowerThermalSampler",
    "RequestRecord",
    "RequestShape",
    "ShapeSource",
    "TrafficSource",
    "Transport",
    "build_schedule",
    "collect_environment_snapshot",
    "collect_environment_snapshot_async",
    "collect_measurement_warnings",
    "estimate_flops_palm",
    "estimate_flops_palm_from_config",
    "measure_baseline_power",
    "require_httpx",
    "write_timeseries_parquet",
]
