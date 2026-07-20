"""MeasurementHarness - owns the measurement lifecycle for any EnginePlugin.

The harness extracts the ~600 lines of identical measurement infrastructure
duplicated across transformers.py and vllm.py into a single location.
Engines become thin plugins implementing the 4-method EnginePlugin protocol.

The implementation lives in ``llenergymeasure.harness.measurement``; the
measured-window mechanics live in ``llenergymeasure.harness.bracket``. This
package init re-exports the public surface plus the module-level helpers that
tests patch. Note: patching a name on this package does NOT affect the
implementation modules' globals - tests that patch module globals must target
the module that owns the name (``measurement`` or ``bracket``).
"""

from __future__ import annotations

from llenergymeasure.harness.bracket import PowerThermalSampler

# Private helpers are re-exported with redundant aliases so legacy
# ``from llenergymeasure.harness import _cuda_sync`` keeps working. They are
# intentionally absent from ``__all__`` (private surface). Tests that PATCH these
# as module globals must target the owning module
# (``measurement`` for the capture/persistence helpers, ``bracket`` for _cuda_sync).
from llenergymeasure.harness.bracket import (
    _cuda_sync as _cuda_sync,
)
from llenergymeasure.harness.measurement import (
    MeasurementHarness,
    collect_environment_snapshot,
    collect_environment_snapshot_async,
    collect_measurement_warnings,
    estimate_flops_palm,
    estimate_flops_palm_from_config,
    measure_baseline_power,
    write_timeseries_parquet,
)
from llenergymeasure.harness.measurement import (
    _capture_observed_params_into_output as _capture_observed_params_into_output,
)
from llenergymeasure.harness.measurement import (
    _check_persistence_mode as _check_persistence_mode,
)

__all__ = [
    "MeasurementHarness",
    "PowerThermalSampler",
    "collect_environment_snapshot",
    "collect_environment_snapshot_async",
    "collect_measurement_warnings",
    "estimate_flops_palm",
    "estimate_flops_palm_from_config",
    "measure_baseline_power",
    "write_timeseries_parquet",
]
