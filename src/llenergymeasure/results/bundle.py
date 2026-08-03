"""Per-experiment results-bundle owner (writer + reader).

``BundleWriter`` is the single home for the assembly policy that decides how a
completed experiment becomes an on-disk bundle: the collision-free experiment
directory, result.json (with runner provenance folded in), system.json
(host snapshot vs the preferred in-container rescue, plus the host-only runner
block), the config.json sidecar move + patch, the timeseries attach, and the
loudness backstops that make a silently-missing artefact visible.

``BundleReader`` is its read-side counterpart: given a bundle directory it
discovers the artefacts via the same ``ARTEFACTS`` registry, parses result.json
(the one required artefact), attaches the system.json snapshot, and returns
a :class:`LoadedBundle` (result + environment + config payload + the discovered
paths). ``persistence.load_result`` is a thin wrapper over it, kept for API
stability. A registry-driven single-artefact accessor (:meth:`BundleReader.read_sidecar`)
serves consumers that need one JSON sidecar (e.g. ``report-gaps`` reading config
provenance) without materialising the whole bundle.

Previously this policy was smeared across ``results.persistence`` (atomic writes
+ dir naming) and ``study.runner._save_and_record`` (~215 lines of assembly). It
now lives here, driven by the ``domain.bundle_artefacts.ARTEFACTS`` registry, so
a future artefact type (e.g. a server-mode per-request series) is added with one
registry entry plus one writer method - ``finalize()`` sweeps it automatically
and ``read()`` surfaces it in ``LoadedBundle.paths``.

The low-level mechanics (collision-free dir + atomic result/timeseries write,
the host environment write) stay in ``results.persistence`` and are delegated to:
``persistence.save_result`` is public API, and keeping the primitive there
preserves the atomic-write + chmod semantics and the existing regression tests.
BundleWriter owns the write policy; BundleReader owns the read policy;
persistence owns the primitives.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import RUNNER_CONTAINER
from llenergymeasure.domain.bundle_artefacts import (
    ARTEFACTS,
    BUNDLE_VERSION,
    CONFIG_SIDECAR_FILENAME,
    REQUESTS_FILENAME,
    SYSTEM_FILENAME,
    TIMESERIES_FILENAME,
)
from llenergymeasure.results import persistence
from llenergymeasure.utils.io import load_json

if TYPE_CHECKING:
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.experiment import ExperimentResult
    from llenergymeasure.domain.provenance import RunnerProvenance
    from llenergymeasure.domain.session import SessionBlock
    from llenergymeasure.results.request_log import RequestLogRow

logger = logging.getLogger(__name__)


class BundleWriter:
    """Assemble one per-experiment results bundle under ``study_dir``.

    Usage mirrors the natural write order::

        writer = BundleWriter(study_dir, model_name=..., engine=..., config_hash=...)
        result_path = writer.write_result(result, runner_provenance=...)
        writer.write_system(host_snapshot=..., runner=...)
        writer.move_config_sidecar(resolved_config_hash=..., resolution_log=...)
        writer.finalize()

    ``write_result`` must run first: it creates the experiment directory that the
    later steps write into (available afterwards as :pyattr:`bundle_dir`).
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        model_name: str,
        engine: str,
        config_hash: str,
        cycle: int = 1,
        experiment_index: int | None = None,
        ts_source_dir: Path | None = None,
    ) -> None:
        self._study_dir = Path(study_dir)
        self._model_name = model_name
        self._engine = engine
        self._config_hash = config_hash
        self._cycle = cycle
        self._experiment_index = experiment_index
        # Directory the harness/container staged its artefacts in. Under docker
        # dispatch this is the rescue dir, which also carries the accurate
        # in-container system.json and config.json.
        self._ts_source_dir = ts_source_dir
        self._dir: Path | None = None
        self._experiment_id: str | None = None
        # Registry keys whose finalize backstop is not expected this run (e.g. an
        # experiment that produced no timeseries).
        self._skip_backstops: set[str] = set()

    @property
    def bundle_dir(self) -> Path:
        """Directory holding the bundle. Only valid after :pymeth:`write_result`."""
        if self._dir is None:
            raise RuntimeError("write_result() must be called before accessing bundle_dir")
        return self._dir

    def write_result(
        self,
        result: ExperimentResult,
        *,
        runner_provenance: RunnerProvenance | None = None,
        session: SessionBlock | None = None,
    ) -> Path:
        """Create the experiment dir and write result.json (+ timeseries copy).

        Attaches ``runner_provenance`` and the ``session`` facts block to the
        frozen result before serialising (both ride in result.json, unlike the
        environment sidecar). Resolves the timeseries parquet from the result's
        ``timeseries`` field inside ``ts_source_dir`` and hands it to the atomic
        writer, then removes the stale staged copy. Returns the path to result.json.

        ``session`` is a passthrough data block (the writer never learns that
        sessions exist): the offline caller stamps ``{session_id, window_count=1}``
        with null raws, the server caller stamps the full launch/warmup/drain facts.
        """
        updates: dict[str, Any] = {}
        if runner_provenance is not None:
            updates["runner_provenance"] = runner_provenance
        if session is not None:
            updates["session"] = session
        if updates and hasattr(result, "model_copy"):
            result = result.model_copy(update=updates)

        ts_filename = getattr(result, "timeseries", None)
        if not ts_filename:
            # No timeseries declared: its finalize backstop does not apply.
            self._skip_backstops.add("timeseries")
        # requests.parquet is a server-mode artefact (one row per issued request
        # per window). Offline and any non-server bundle produces none, so its
        # finalize backstop is skipped here (the single opt-out point, mirroring
        # the timeseries skip - no per-call-site hand-glue). Server bundles keep
        # the backstop so a missing request log stays a loud warning.
        if getattr(result, "serving_mode", "offline") != "server":
            self._skip_backstops.add("requests")
        ts_source: Path | None = None
        if ts_filename and self._ts_source_dir is not None:
            candidate = self._ts_source_dir / ts_filename
            if candidate.exists():
                ts_source = candidate

        result_path = persistence.save_result(
            result,
            self._study_dir,
            model_name=self._model_name,
            engine=self._engine,
            timeseries_source=ts_source,
            experiment_index=self._experiment_index,
            cycle=self._cycle,
        )
        self._dir = result_path.parent
        self._experiment_id = getattr(result, "experiment_id", None)

        # Remove the stale staged parquet now that it is copied into the bundle.
        if ts_source is not None:
            ts_source.unlink(missing_ok=True)
        return result_path

    def write_system(
        self,
        *,
        host_snapshot: EnvironmentSnapshot | None,
        runner: RunnerProvenance | None = None,
        session: SessionBlock | None = None,
    ) -> None:
        """Write system.json with the rescue-preference policy.

        The host snapshot (patched with the host-only runner block) is written
        first. Under docker dispatch the accurate snapshot is collected INSIDE
        the container and rescued to ``ts_source_dir``; that file is preferred
        and overwrites the host-written one, with the runner block re-applied.
        A docker run that lands without a rescued snapshot warns loudly: the
        persisted system.json would then describe the dispatching host, not
        the container the experiment actually ran in.

        ``runner`` is the unified runner provenance: it is both the
        system.json ``runner`` block (image + digest + source) and the
        mode discriminator for the missing-rescue docker warning. ``session`` is
        the session-facts data block dual-serialised alongside the result.json copy.
        """
        if self._dir is None:
            raise RuntimeError("write_result() must be called before write_system()")

        # Attach the host-only runner block and the session block to the host
        # snapshot so the host write carries them (local path, and the docker
        # no-rescue fallback).
        if host_snapshot is not None and (runner is not None or session is not None):
            updates: dict[str, Any] = {}
            if runner is not None:
                updates["runner"] = runner
            if session is not None:
                updates["session"] = session
            host_snapshot = host_snapshot.model_copy(update=updates)

        if host_snapshot is not None:
            persistence.save_system(
                host_snapshot,
                self._experiment_id or "",
                self._config_hash,
                self._dir,
            )

        rescued = self._ts_source_dir / SYSTEM_FILENAME if self._ts_source_dir is not None else None
        if rescued is not None and rescued.exists():
            self._rescue_system(rescued, runner, session)
        elif runner is not None and runner.mode == RUNNER_CONTAINER:
            logger.warning(
                "No in-container system.json rescued for %s (cycle %d) at %s - "
                "system.json records the dispatching host, not the container "
                "the experiment ran in.",
                self._config_hash,
                self._cycle,
                self._dir,
            )

    def _rescue_system(
        self,
        rescued: Path,
        runner: RunnerProvenance | None,
        session: SessionBlock | None = None,
    ) -> None:
        """Prefer the rescued in-container system.json over the host write.

        Loads the rescued snapshot, patches the host-only runner block and the
        session-facts block into it, and atomically overwrites the host-written
        system.json. Best-effort: a read/write failure warns loudly and leaves the
        host-written file (which already carries both blocks) in place. The staged
        rescue file is always consumed.
        """
        self._stage_json_artefact(
            rescued,
            SYSTEM_FILENAME,
            lambda payload: self._patch_runner_block(payload, runner, session),
            failure_note=(
                "Failed to rescue in-container system.json - system.json will "
                "record the dispatching host, not the container the experiment ran in"
            ),
        )

    def _stage_json_artefact(
        self,
        src: Path,
        dest_filename: str,
        patch: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        failure_note: str,
    ) -> None:
        """Move a staged JSON artefact into the bundle, applying ``patch``.

        Shared scaffolding for the system-snapshot rescue and the config-sidecar
        move, which differ only in their patch callback and failure message: load the
        staged file, apply ``patch``, atomically write it into the bundle dir
        under ``dest_filename``. Best-effort - a read/write failure logs
        ``failure_note`` with context and never raises; the staged source is
        always consumed.
        """
        assert self._dir is not None
        try:
            payload = patch(load_json(src))
            persistence._atomic_write(
                json.dumps(payload, indent=2, default=str),
                self._dir / dest_filename,
            )
        except Exception as exc:
            logger.warning(
                "%s (%s, cycle %d, from %s): %s",
                failure_note,
                self._config_hash,
                self._cycle,
                src,
                exc,
            )
        finally:
            src.unlink(missing_ok=True)

    @staticmethod
    def _patch_runner_block(
        payload: dict[str, Any],
        runner: RunnerProvenance | None,
        session: SessionBlock | None = None,
    ) -> dict[str, Any]:
        """Patch the host-only runner + session blocks into a rescued system payload.

        The container writes system.json without facts only the host knows: the
        runner block (image ref, registry digest, precedence source) and the
        session-facts block. This patches them in and stamps ``bundle_version`` if
        the (older-image) payload omitted it. A no-op when neither block is available.
        """
        if runner is not None:
            payload["runner"] = runner.model_dump()
            payload.setdefault("bundle_version", BUNDLE_VERSION)
        if session is not None:
            payload["session"] = session.model_dump(mode="json")
            payload.setdefault("bundle_version", BUNDLE_VERSION)
        return payload

    def move_config_sidecar(
        self,
        *,
        resolved_config_hash: str | None = None,
        resolution_log: dict[str, Any] | None = None,
    ) -> None:
        """Move the harness-written config.json into the bundle, patched.

        The harness writes config.json to the staging dir; the host moves it and
        patches in two fields the harness subprocess cannot compute: the
        ``resolved_config_hash`` (from StudyConfig) and the per-field
        ``provenance`` log (whose source labels are only known in the parent).
        Best-effort: a read/write failure warns loudly and the staged file is
        always consumed. No-op when no staging dir or no config.json is present.
        """
        if self._dir is None:
            raise RuntimeError("write_result() must be called before move_config_sidecar()")
        if self._ts_source_dir is None:
            return
        src = self._ts_source_dir / CONFIG_SIDECAR_FILENAME
        if not src.exists():
            return

        def _patch(payload: dict[str, Any]) -> dict[str, Any]:
            if resolved_config_hash is not None:
                payload["resolved_config_hash"] = resolved_config_hash
            if resolution_log:
                payload["provenance"] = resolution_log
            payload.setdefault("bundle_version", BUNDLE_VERSION)
            return payload

        self._stage_json_artefact(
            src,
            CONFIG_SIDECAR_FILENAME,
            _patch,
            failure_note=(
                "Failed to move config.json sidecar - provenance and authoritative "
                "engine/model identity will be missing from this result"
            ),
        )

    def write_requests(
        self,
        rows: list[RequestLogRow],
        *,
        experiment_id: str | None = None,
        span_start: float | None = None,
        span_end: float | None = None,
    ) -> Path:
        """Write the window's per-request log (requests.parquet) into the bundle.

        The single writer method for the ``requests`` registry artefact (O7.7):
        the server session hands one window's request rows and ``finalize()``
        sweeps the outcome via the registry with no hand-glue. Must run after
        :meth:`write_result` (it writes into the bundle dir). The Parquet carries
        the same identity metadata as the result (experiment id + config hash) plus
        the window's measured ``span_start`` / ``span_end`` (so per-row receipt
        series can be re-clipped to the window offline), so it stays attributable
        and re-derivable if separated from its directory. An empty ``rows`` still
        writes a schema-only Parquet (a truthful empty log), so a window that issued
        no requests never trips the missing-artefact backstop.
        """
        if self._dir is None:
            raise RuntimeError("write_result() must be called before write_requests()")
        from llenergymeasure.results.request_log import write_requests_parquet

        return write_requests_parquet(
            rows,
            self._dir / REQUESTS_FILENAME,
            experiment_id=experiment_id or self._experiment_id,
            declared_config_hash=self._config_hash,
            span_start=span_start,
            span_end=span_end,
        )

    def mark_artefact_absent(self, name: str) -> None:
        """Declare a registered artefact expected-absent for this bundle.

        Suppresses its ``finalize()`` loudness backstop. Used by callers whose
        dispatch legitimately produces no such artefact (e.g. the host-side server
        measurement path writes no config.json - there is no harness subprocess to
        stage one), so its absence is not a silent-loss warning.
        """
        self._skip_backstops.add(name)

    def patch_session_block(self, session: SessionBlock) -> None:
        """Re-stamp the session block into an already-written (not yet finalized) bundle.

        The in-write-path patch (O7.5/O7.6): a window bundle is written at level
        close carrying a PRELIMINARY session block (drain fields null, totals not
        yet final); at clean session close the drain raws and final totals are
        known, so this re-writes result.json and system.json with the complete
        block. It runs BEFORE :meth:`finalize`, so it is the normal write path, not
        a mutation of a finished record. Best-effort per file: a read/write failure
        warns and leaves the preliminary block in place.
        """
        if self._dir is None:
            raise RuntimeError("write_result() must be called before patch_session_block()")
        block = session.model_dump(mode="json")
        from llenergymeasure.domain.bundle_artefacts import RESULT_FILENAME

        for filename in (RESULT_FILENAME, SYSTEM_FILENAME):
            path = self._dir / filename
            if not path.exists():
                continue
            try:
                payload = load_json(path)
                payload["session"] = block
                persistence._atomic_write(
                    json.dumps(payload, indent=2, default=str),
                    path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to patch session block into %s (%s, cycle %d): %s",
                    path,
                    self._config_hash,
                    self._cycle,
                    exc,
                )

    def finalize(self) -> None:
        """Run the loudness backstops uniformly from the artefact registry.

        Sweeps every registered artefact flagged ``warn_if_missing`` and warns
        when one that was expected for this run is absent from the bundle. This
        is the uniform home for the config-sidecar backstop and the
        declared-but-missing timeseries backstop; a newly-registered artefact is
        swept here automatically.
        """
        if self._dir is None:
            raise RuntimeError("write_result() must be called before finalize()")
        for name, spec in ARTEFACTS.items():
            if not spec.warn_if_missing or name in self._skip_backstops:
                continue
            if not (self._dir / spec.filename).exists():
                logger.warning(
                    "Bundle artefact '%s' (%s) missing from %s (%s, cycle %d)%s",
                    name,
                    spec.filename,
                    self._dir,
                    self._config_hash,
                    self._cycle,
                    f" - {spec.missing_note}" if spec.missing_note else "",
                )


@dataclass(frozen=True)
class LoadedBundle:
    """Everything :class:`BundleReader` recovers from one per-experiment bundle.

    Attributes:
        bundle_dir: The directory the bundle was read from.
        result: The parsed ``result.json`` (the one required artefact), with the
            environment snapshot attached to ``result.environment`` when present.
        environment: The parsed ``system.json`` snapshot, or None when the
            sidecar is absent or unparseable. The same object attached to
            ``result.environment``; exposed here for direct access.
        config: The raw ``config.json`` payload dict (provenance, declared
            config, observed hashes), or None when the sidecar is absent.
        paths: Registry key -> on-disk path for every artefact present in the
            bundle. A newly-registered artefact appears here automatically.
    """

    bundle_dir: Path
    result: ExperimentResult
    environment: EnvironmentSnapshot | None
    config: dict[str, Any] | None
    paths: dict[str, Path]


class BundleReader:
    """Read a per-experiment results bundle, discovery driven by the registry.

    The counterpart to :class:`BundleWriter`. :meth:`read` iterates the
    ``ARTEFACTS`` registry to discover which artefacts are present, enforces the
    ``required`` contract (raising when result.json is missing), parses the JSON
    artefacts with their typed models, and returns a :class:`LoadedBundle`.
    :meth:`read_sidecar` is the registry-driven single-artefact accessor for
    consumers that need one JSON sidecar without materialising the whole bundle.
    """

    @staticmethod
    def read(bundle_dir: Path) -> LoadedBundle:
        """Read the bundle under ``bundle_dir`` into a :class:`LoadedBundle`.

        Discovery is registry-driven: every entry in ``ARTEFACTS`` is checked for
        presence (populating ``LoadedBundle.paths``) and a missing ``required``
        artefact raises. result.json is parsed strictly via the shared payload
        parser; system.json and config.json are best-effort (a corrupt or
        absent optional sidecar yields None, never an error). When the result
        references a timeseries whose parquet did not land, a ``UserWarning`` is
        emitted (matching the historical ``load_result`` behaviour).

        Args:
            bundle_dir: The per-experiment bundle directory (the parent of
                result.json).

        Returns:
            The assembled :class:`LoadedBundle`.

        Raises:
            FileNotFoundError: A required artefact (result.json) is missing.
        """
        bundle_dir = Path(bundle_dir)

        paths: dict[str, Path] = {}
        for name, spec in ARTEFACTS.items():
            candidate = bundle_dir / spec.filename
            if candidate.exists():
                paths[name] = candidate
            elif spec.required:
                raise FileNotFoundError(
                    f"Required bundle artefact '{name}' ({spec.filename}) missing from {bundle_dir}"
                )

        result = BundleReader._read_result_artefact(paths["result"])

        environment = BundleReader._read_system_artefact(paths.get("system"))
        if environment is not None:
            result = result.model_copy(update={"environment": environment})

        # config.json is best-effort here (a corrupt sidecar must not break the
        # whole read); read_sidecar's strict variant is for consumers that need
        # to distinguish absent from corrupt.
        config: dict[str, Any] | None = None
        if "config" in paths:
            try:
                config = load_json(paths["config"])
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not load config sidecar %s: %s", paths["config"], exc)

        # Declared-but-missing timeseries: the result references a parquet that is
        # not in the bundle. Preserves the historical load_result degradation.
        if result.timeseries is not None and "timeseries" not in paths:
            warnings.warn(
                f"Timeseries sidecar missing at {bundle_dir / TIMESERIES_FILENAME}. "
                "result.timeseries field preserved but file is not present.",
                UserWarning,
                stacklevel=2,
            )

        return LoadedBundle(
            bundle_dir=bundle_dir,
            result=result,
            environment=environment,
            config=config,
            paths=paths,
        )

    @staticmethod
    def read_sidecar(bundle_dir: Path, key: str) -> dict[str, Any] | None:
        """Read one registry JSON artefact's raw payload from a bundle dir.

        The registry-driven single-artefact read for consumers that want one
        sidecar (e.g. ``report-gaps`` reading the config.json ``provenance``
        section) without loading result.json or the whole bundle. Returns the
        decoded payload, or None when the artefact is absent. A present but
        unparseable file raises so callers can distinguish an absent sidecar
        from a corrupt one.

        Args:
            bundle_dir: The per-experiment bundle directory.
            key: The ``ARTEFACTS`` registry key (e.g. ``"config"``).

        Returns:
            The decoded JSON payload, or None when the artefact is absent.

        Raises:
            ValueError: The keyed artefact is not a JSON artefact.
            OSError / json.JSONDecodeError: The artefact is present but unreadable.
        """
        spec = ARTEFACTS[key]
        if spec.kind != "json":
            raise ValueError(f"Artefact '{key}' is {spec.kind}, not a JSON sidecar")
        path = Path(bundle_dir) / spec.filename
        if not path.exists():
            return None
        payload: dict[str, Any] = load_json(path)
        return payload

    @staticmethod
    def _read_result_artefact(path: Path) -> ExperimentResult:
        """Parse result.json strictly, with the legacy best-effort fallback.

        A legacy result.json - one with no ``bundle_version`` key, or a
        ``bundle_version`` older than the current ``BUNDLE_VERSION`` - is read
        best-effort with a single ``UserWarning`` covering the whole bundle (the
        one documented pre-1.0 break policy). The shared parser (tolerant=False)
        validates the payload; the ExperimentResult before-validator drops keys
        retired across a bundle break (``schema_version``, the top-level
        ``baseline_power_w`` copy) so the legacy file falls back to the current
        defaults rather than being rejected. Legacy system.json shapes (the
        old separate runner block, the renamed CUDA field, the dropped hardware
        fields) are tolerated on the system read path, silently, so this one
        warning is not multiplied per artefact.
        """
        from llenergymeasure.domain.result_payload import parse_experiment_result_payload

        raw = load_json(path)
        if isinstance(raw, dict) and raw.get("bundle_version") != BUNDLE_VERSION:
            warnings.warn(
                f"legacy results bundle (bundle_version={raw.get('bundle_version')!r}); "
                "readable best-effort",
                UserWarning,
                stacklevel=2,
            )
        return parse_experiment_result_payload(raw, tolerant=False, expected_version=None)

    @staticmethod
    def _read_system_artefact(path: Path | None) -> EnvironmentSnapshot | None:
        """Load system.json into an EnvironmentSnapshot (best-effort).

        Returns None when the sidecar is absent or cannot be parsed, so a missing
        or corrupt sidecar never breaks the read. The sidecar's extra
        ``experiment_id`` / ``declared_config_hash`` / ``bundle_version`` keys
        are ignored by EnvironmentSnapshot validation.
        """
        if path is None:
            return None
        from llenergymeasure.domain.environment import EnvironmentSnapshot

        try:
            return EnvironmentSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load environment sidecar %s: %s", path, exc)
            return None
