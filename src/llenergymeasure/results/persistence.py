"""v3.0 results persistence - save, load, atomic writes.

Handles directory lifecycle, collision avoidance,
JSON serialisation (primary), and Parquet sidecar management.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from llenergymeasure.domain.bundle_artefacts import (
    BUNDLE_VERSION,
    CONFIG_SIDECAR_FILENAME,
    ENVIRONMENT_FILENAME,
    RESULT_FILENAME,
    TIMESERIES_FILENAME,
)

if TYPE_CHECKING:
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.experiment import ExperimentResult

logger = logging.getLogger(__name__)


def _experiment_dir_name(
    result: ExperimentResult,
    *,
    model_name: str,
    engine: str,
    experiment_index: int | None = None,
    cycle: int = 1,
) -> str:
    """Generate a human-readable directory name for an experiment result.

    Format: ``[{index:03d}_]c{cycle}_{model_short}-{engine}_{hash[:8]}``

    ``model_name`` and ``engine`` are the identity of what was measured. Their
    authoritative home is the ``config.json`` sidecar (``result.json`` carries
    convenience copies only), so the caller supplies them explicitly.

    When ``experiment_index`` is provided (study context), the directory is
    prefixed with a zero-padded index for natural sort ordering.

    Examples:
        ``001_c1_Qwen2.5-0.5B-transformers_abcdef01``
        ``c1_gpt2-vllm_fedcba98``  (single experiment, no index)
    """
    from llenergymeasure.utils.formatting import model_short_name

    model_short = model_short_name(model_name)
    config_hash = result.declared_config_hash[:8]

    # Build slug: model_short-engine
    slug = f"{model_short}-{engine}"
    # Sanitise for filesystem: replace spaces, slashes, special chars
    slug = slug.replace(" ", "_").replace("/", "-").replace(":", "-")
    # Truncate overly long slugs (filesystem limits)
    if len(slug) > 120:
        slug = slug[:120]

    if experiment_index is not None:
        return f"{experiment_index:03d}_c{cycle}_{slug}_{config_hash}"
    return f"c{cycle}_{slug}_{config_hash}"


def _find_collision_free_dir(base: Path) -> Path:
    """Return base or base_1, base_2, etc. - never overwrites.

    Creates the directory atomically to avoid race conditions.
    """
    target = base
    counter = 0
    while target.exists():
        counter += 1
        target = Path(f"{base}_{counter}")
    target.mkdir(parents=True)
    return target


def _atomic_write(content: str, path: Path) -> None:
    """Write content to path atomically via temp file + os.replace().

    Uses POSIX rename semantics - atomic on same filesystem.
    Calls fsync before replace to ensure durability on power loss.
    Cleans up temp file on failure.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.stem)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        # tempfile.mkstemp() creates the file mode 0600 regardless of umask.
        # Every caller writes a results artefact (result.json, config/environment
        # sidecars, manifest.json, equivalence groups), all of which must be
        # world-readable so a non-root host can read a sidecar written by a root
        # container during docker-dispatch rescue - matching the 0644 that
        # result.json and timeseries.parquet already get. Without this, the host
        # rescue hits PermissionError and the sidecar is silently dropped.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def save_config_sidecar(
    experiment_dir: Path,
    *,
    experiment_id: str,
    config_hash: str,
    engine: str,
    engine_version: str,
    model_name: str,
    measurement_methodology: str,
    steady_state_window: tuple[float, float] | None = None,
    measurement_window_discard_fraction: float | None = None,
    steady_state_not_detected: bool = False,
    observed_engine_params: dict[str, object] | None = None,
    observed_sampling_params: dict[str, object] | None = None,
    resolved_config_hash: str | None = None,
    observed_config_hash: str | None = None,
    config_validation_observations: list[dict[str, object]] | None = None,
    declared_config: dict[str, object] | None = None,
) -> Path:
    """Write the per-experiment ``config.json`` sidecar with resolved/observed config-hash payload.

    Schema lives in ``.product/designs/config-deduplication-dormancy/sweep-dedup.md``
    §3.3. Fields:

    - ``engine`` / ``engine_version`` - the inference engine and its library
      version. This sidecar is the authoritative home for engine identity;
      ``result.json`` keeps ``engine`` as a convenience copy only, and
      ``engine_version`` lives here exclusively.
    - ``model_name`` - the model name/path that was measured (a configuration
      input, not a measurement output). Authoritative here; ``result.json``
      keeps a convenience copy.
    - ``measurement_methodology`` / ``steady_state_window`` /
      ``measurement_window_discard_fraction`` / ``steady_state_not_detected`` -
      how the measurement window was set up. Methodology choices are configuration,
      so they live here rather than alongside the metrics in ``result.json``.
      ``steady_state_window`` and ``measurement_window_discard_fraction`` are
      omitted when None (total/windowed methodologies).
    - ``observed_engine_params`` / ``observed_sampling_params`` - authoritative
      post-construction library state (populated by
      :func:`llenergymeasure.engines._observed.extract_observed_params`).
    - ``resolved_config_hash`` - library-resolution mechanism-output hash, carried forward from sweep
      expansion via ``StudyConfig.declared_resolved_config_hashes``.
    - ``observed_config_hash`` - library-observation hash computed from the effective
      params at sidecar-write time.
    - ``config_validation_observations`` - DormantField entries that
      ``_apply_rules`` attached at load time.
    - ``declared_config`` - the full user-declared ``ExperimentConfig``
      (JSON model dump). Every other config field in this sidecar is a hash;
      this is the only place the declared state is recorded in full. Consumed
      by observed-collision detection
      (:func:`llenergymeasure.study.equivalence_groups.find_observed_collisions`),
      which groups experiments by ``observed_config_hash`` and diffs their
      declared configs to find fields whose variation left the engine-effective
      state identical.

    Two further fields are patched into this sidecar later by the study layer
    (``llenergymeasure.study.runner._save_and_record``) rather than written
    here, because both need study-level context the harness subprocess lacks:

    - ``resolved_config_hash`` - carried forward from sweep expansion.
    - ``provenance`` - the per-field resolution log (``{source, effective,
      default}`` per non-default field) built by
      :func:`llenergymeasure.config.resolution.build_resolution_log`, whose
      ``cli_flag``/``sweep``/``yaml`` source labels are only known in the
      parent process. This replaces the retired ``_resolution.json`` sidecar.

    Any missing optional field is omitted from the sidecar (not written as
    null) so downstream consumers distinguish "not available" from
    "explicitly null". The file is small (< 4 KB typical) and atomically
    written.
    """
    payload: dict[str, object] = {
        "bundle_version": BUNDLE_VERSION,
        "experiment_id": experiment_id,
        "declared_config_hash": config_hash,
        "engine": engine,
        "engine_version": engine_version,
        "model_name": model_name,
        "measurement_methodology": measurement_methodology,
        "steady_state_not_detected": steady_state_not_detected,
    }
    if steady_state_window is not None:
        payload["steady_state_window"] = list(steady_state_window)
    if measurement_window_discard_fraction is not None:
        payload["measurement_window_discard_fraction"] = measurement_window_discard_fraction
    if observed_engine_params is not None:
        payload["observed_engine_params"] = observed_engine_params
    if observed_sampling_params is not None:
        payload["observed_sampling_params"] = observed_sampling_params
    if resolved_config_hash is not None:
        payload["resolved_config_hash"] = resolved_config_hash
    if observed_config_hash is not None:
        payload["observed_config_hash"] = observed_config_hash
    if config_validation_observations is not None:
        payload["config_validation_observations"] = config_validation_observations
    if declared_config is not None:
        payload["declared_config"] = declared_config

    path = experiment_dir / CONFIG_SIDECAR_FILENAME
    _atomic_write(json.dumps(payload, indent=2, default=str), path)
    logger.debug("Saved config sidecar to %s", path)
    return path


def save_environment(
    snapshot: EnvironmentSnapshot,
    experiment_id: str,
    declared_config_hash: str,
    experiment_dir: Path,
) -> Path:
    """Write per-experiment environment.json sidecar.

    Contains hardware/runtime metadata for the experiment. Software package
    listings live in the study-level environment.json instead.

    Args:
        snapshot: EnvironmentSnapshot with hardware/runtime metadata.
        experiment_id: Unique experiment identifier.
        declared_config_hash: Config hash for orphan attribution.
        experiment_dir: Experiment result directory (must already exist).

    Returns:
        Path to the written environment.json file.
    """
    env_data: dict[str, object] = {
        "bundle_version": BUNDLE_VERSION,
        "experiment_id": experiment_id,
        "declared_config_hash": declared_config_hash,
    }
    snapshot_dict = snapshot.model_dump()
    env_data["hardware"] = snapshot_dict["hardware"]
    env_data["python_version"] = snapshot_dict["python_version"]
    env_data["tool_version"] = snapshot_dict["tool_version"]
    env_data["cuda_version"] = snapshot_dict.get("cuda_version")
    env_data["cuda_version_source"] = snapshot_dict.get("cuda_version_source")
    # Runner provenance block (docker vs local, image + registry digest,
    # precedence source). None when the snapshot carries no runner block (e.g.
    # the in-container snapshot, whose runner facts the host patches in later).
    env_data["runner"] = snapshot_dict.get("runner")

    path = experiment_dir / ENVIRONMENT_FILENAME
    _atomic_write(json.dumps(env_data, indent=2, default=str), path)
    logger.debug("Saved environment to %s", path)
    return path


def save_result(
    result: ExperimentResult,
    output_dir: Path,
    *,
    model_name: str,
    engine: str,
    timeseries_source: Path | None = None,
    experiment_index: int | None = None,
    cycle: int = 1,
) -> Path:
    """Save ExperimentResult to a collision-safe subdirectory of output_dir.

    Creates: ``{output_dir}/[{index}_]c{cycle}_{model}-{engine}_{hash}/result.json``
    If timeseries_source provided: copies to ``{dir}/timeseries.parquet``.

    Per-field config provenance is no longer written here. It is folded into the
    ``config.json`` sidecar's ``provenance`` section by the study layer.

    Args:
        result: The experiment result to persist.
        output_dir: Parent directory. Created if missing.
        model_name: Model name/path (used for the directory slug). Authoritative
            home is the ``config.json`` sidecar; the result carries a convenience copy.
        engine: Inference engine name (used for the directory slug). Authoritative
            home is the ``config.json`` sidecar; the result carries a convenience copy.
        timeseries_source: Optional path to existing .parquet file to copy in.
        experiment_index: Optional 1-based experiment index for directory prefix
            (used in study context for natural sort ordering).
        cycle: Cycle number (1-based). Embedded in directory name.

    Returns:
        Path to the result.json file (usable with load_result() directly).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_name = _experiment_dir_name(
        result,
        model_name=model_name,
        engine=engine,
        experiment_index=experiment_index,
        cycle=cycle,
    )
    base_dir = output_dir / dir_name
    target_dir = _find_collision_free_dir(base_dir)

    result_path = target_dir / RESULT_FILENAME
    _atomic_write(result.model_dump_json(indent=2), result_path)
    logger.debug("Saved result to %s", result_path)

    if timeseries_source is not None:
        timeseries_source = Path(timeseries_source)
        if timeseries_source.exists():
            dest = target_dir / TIMESERIES_FILENAME
            shutil.copy2(timeseries_source, dest)
            logger.debug("Copied timeseries sidecar to %s", dest)
        else:
            logger.warning("timeseries_source %s does not exist - skipping copy", timeseries_source)

    return result_path


def load_result(path: Path) -> ExperimentResult:
    """Load ExperimentResult from a result.json path.

    Thin wrapper over :meth:`llenergymeasure.results.bundle.BundleReader.read`,
    kept as public API for stability. The reader owns the read policy: it
    auto-discovers the timeseries.parquet and environment.json sidecars in the
    same directory, parses result.json (dropping the retired ``schema_version``
    key on legacy bundles), attaches the environment snapshot to
    ``result.environment`` when present, and emits a UserWarning when the result
    references a timeseries whose parquet is missing (graceful degradation).

    Args:
        path: Path to result.json (as returned by save_result()).

    Returns:
        ExperimentResult loaded from disk, with ``environment`` populated
        from the environment.json sidecar when one is present.
    """
    from llenergymeasure.results.bundle import BundleReader

    return BundleReader.read(Path(path).parent).result
