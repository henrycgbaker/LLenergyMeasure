# Study Resume Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/study-resume.md](../decisions/study-resume.md)
**Status**: DRAFT — manifest schema confirmed; resume logic is a later v2.0 milestone

---

## Summary

- `study_manifest.json` written alongside results during every study (v2.0 — always)
- `llem run --resume study-dir/` is a later v2.0 milestone (after core study execution works)
- Early v2.0 milestones: interrupted studies re-run from scratch

---

## `StudyManifest` vs `StudyResult` — Disambiguation

These are **two distinct Pydantic models** that serve different purposes. Phase 5 implementors
must not conflate them.

| | `StudyManifest` | `StudyResult` |
|---|---|---|
| **File** | `study_manifest.json` | `study_summary.json` |
| **When written** | At study start; updated after each experiment | Once, at study completion |
| **Purpose** | In-progress checkpoint for study recovery | Final return value of `run_study()` |
| **Written by** | `ManifestWriter` (inside `StudyRunner`) | `StudyRunner` (at `run_study()` return) |
| **Overlapping fields** | `study_name`, `started_at`, `llenergymeasure_version` | Same — but semantically different objects |
| **Stays on disk after completion** | Yes — alongside results for provenance | Yes — the user-visible summary |

**Critical distinction:** `StudyManifest` is **incremental state** (written during the run,
reflects partial progress). `StudyResult` is the **complete return value** (written once, at the
end). They have overlapping fields because they describe the same study — but they are not the
same object and must not be merged.

See also: `designs/result-schema.md` § "New: Study Result Schema" for `StudyResult` definition.

---

---

## `study_manifest.json` Schema

Written to `results/{study_name}-{timestamp}/study_manifest.json` at study start.
Updated after each experiment completes (or fails).

```python
# src/llenergymeasure/study/manifest.py

from pydantic import BaseModel
from typing import Literal
from datetime import datetime


class ExperimentManifestEntry(BaseModel):
    config_hash: str                          # identifies the experiment
    config_summary: str                       # human-readable: "pytorch / batch=1 / bf16"
    cycle: int                                # which study cycle (1-indexed)
    status: Literal["pending", "running", "completed", "failed"]
    result_file: str | None = None            # relative path to ExperimentResult JSON
    error_type: str | None = None            # exception type if failed
    error_message: str | None = None         # exception message if failed
    started_at: datetime | None = None
    completed_at: datetime | None = None


class StudyManifest(BaseModel):
    schema_version: str = "2.0"
    study_name: str
    study_yaml_hash: str                      # SHA-256 of study.yaml contents
    study_yaml_path: str                      # path to study.yaml (for reference)
    llenergymeasure_version: str
    started_at: datetime
    completed_at: datetime | None = None

    experiments: list[ExperimentManifestEntry]
```

**File location** in study results directory:
```
results/
  batch-size-sweep-2026-02/
    study_manifest.json                       ← this file
    llama-3.1-8b_pytorch_batch-1_....json    ← per-experiment ExperimentResult
    ...
```

**Concrete example**:
```json
{
  "schema_version": "2.0",
  "study_name": "batch-size-sweep-2026-02",
  "study_yaml_hash": "abc123def456",
  "study_yaml_path": "study.yaml",
  "llenergymeasure_version": "2.0.0",
  "started_at": "2026-02-19T14:30:00Z",
  "completed_at": null,
  "experiments": [
    {
      "config_hash": "def456",
      "config_summary": "pytorch / batch=1 / bf16",
      "cycle": 1,
      "status": "completed",
      "result_file": "llama-3.1-8b_pytorch_batch-1_2026-02-19T14-30.json",
      "started_at": "2026-02-19T14:30:00Z",
      "completed_at": "2026-02-19T14:32:14Z"
    },
    {
      "config_hash": "ghi789",
      "config_summary": "pytorch / batch=4 / bf16",
      "cycle": 1,
      "status": "pending",
      "result_file": null,
      "started_at": null,
      "completed_at": null
    }
  ]
}
```

---

## Manifest Writer (v2.0)

```python
# src/llenergymeasure/study/manifest.py

class ManifestWriter:
    def __init__(self, study: StudyConfig, results_dir: Path) -> None:
        self.path = results_dir / "study_manifest.json"
        self.manifest = StudyManifest(
            study_name=study.name,
            study_yaml_hash=_hash_yaml(study),
            study_yaml_path=str(study.source_path),
            llenergymeasure_version=__version__,
            started_at=datetime.utcnow(),
            experiments=_build_entries(study),
        )
        self._write()

    def mark_running(self, config_hash: str, cycle: int) -> None:
        entry = self._find(config_hash, cycle)
        entry.status = "running"
        entry.started_at = datetime.utcnow()
        self._write()

    def mark_completed(self, config_hash: str, cycle: int, result_file: str) -> None:
        entry = self._find(config_hash, cycle)
        entry.status = "completed"
        entry.result_file = result_file
        entry.completed_at = datetime.utcnow()
        self._write()

    def mark_failed(self, config_hash: str, cycle: int, error: StudyFailed) -> None:
        entry = self._find(config_hash, cycle)
        entry.status = "failed"
        entry.error_type = error.exception_type
        entry.error_message = error.error_message
        entry.completed_at = datetime.utcnow()
        self._write()

    def _find(self, config_hash: str, cycle: int) -> ExperimentManifestEntry:
        for entry in self.manifest.experiments:
            if entry.config_hash == config_hash and entry.cycle == cycle:
                return entry
        raise KeyError(f"No manifest entry for config_hash={config_hash!r}, cycle={cycle}")

    def _write(self) -> None:
        self.path.write_text(self.manifest.model_dump_json(indent=2))
```

**Write frequency**: after every status change (mark_running, mark_completed, mark_failed).
This ensures the manifest is always consistent with the actual state — an interrupted study
leaves the manifest showing which experiments completed before interruption.

<!-- NOTE: Writing to disk on every update is a side effect inside run_study() — this
     is intentional for a study (long-running operation requiring checkpoint state).
     It is consistent with the "library is side-effect free" principle for run_experiment(),
     which is a single-shot operation. See NEEDS_ADDRESSING.md for the full nuance. -->

---

## Resume Logic (v2.0 — later milestone)

```python
# llem run --resume study.yaml (v2.0 resume milestone flag)

def resume_study(study_yaml_path: Path, results_dir: Path) -> StudyResult:
    manifest = StudyManifest.model_validate_json(
        (results_dir / "study_manifest.json").read_text()
    )

    # Verify study.yaml hasn't changed
    current_hash = _hash_yaml(study_yaml_path)
    if current_hash != manifest.study_yaml_hash:
        confirm = click.confirm(
            f"study.yaml has changed since the interrupted run "
            f"(hash mismatch). Experiments may differ. Proceed anyway?"
        )
        if not confirm:
            raise SystemExit(1)

    # Identify what still needs running
    pending = [e for e in manifest.experiments if e.status in ("pending", "running")]
    # "running" = was mid-run when interrupted; treat as pending

    # Re-run pending experiments, skip completed
    ...
```

**config_hash as stable identity**: two experiments with the same `config_hash` are the same
experiment. If `study.yaml` is unchanged, hash mismatches are impossible — they indicate
the study definition changed.

---

## Why Manifest in v2.0 if Resume is a Later v2.0 Milestone

The manifest data is useful even without resume:
1. **Post-hoc inspection**: which experiments succeeded/failed and when
2. **Study reproducibility**: `study_yaml_hash` documents the exact study definition used
3. **Debugging**: failed entries include error type and message
4. **Resume readiness**: manifest is already written in v2.0 — resume logic just reads it in the later milestone

The cost (one small JSON write per experiment) is negligible.

---

## Related

- [../decisions/study-resume.md](../decisions/study-resume.md): Decision rationale
- [experiment-isolation.md](experiment-isolation.md): StudyRunner (where manifest is written)
- [result-schema.md](result-schema.md): StudyResult (different from StudyManifest — see NEEDS_ADDRESSING)
- [cli-commands.md](cli-commands.md): `llem run --resume` (v2.0 resume milestone flag)
