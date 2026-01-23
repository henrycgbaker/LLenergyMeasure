# state/ - Experiment State Management

Persistent state tracking for experiment lifecycle, enabling resumption of interrupted experiments and preventing duplicate work.

## Purpose

- Track experiment progress across process restarts
- Enable `--resume` functionality for interrupted experiments
- Detect stale/orphaned experiment states
- Support multi-process progress tracking

## Key Components

### ExperimentStatus

Lifecycle states for experiments:

```python
class ExperimentStatus(str, Enum):
    INITIALISED = "initialised"  # Created, not started
    RUNNING = "running"          # In progress
    COMPLETED = "completed"      # All processes done
    AGGREGATED = "aggregated"    # Results aggregated
    FAILED = "failed"            # Error occurred
    INTERRUPTED = "interrupted"  # Ctrl+C or crash
```

### ProcessStatus

Per-process states within an experiment:

```python
class ProcessStatus(str, Enum):
    PENDING = "pending"      # Not started
    RUNNING = "running"      # In progress
    COMPLETED = "completed"  # Done
    FAILED = "failed"        # Error
```

### ExperimentState

Full experiment state model:

```python
state = ExperimentState(
    experiment_id="exp_20240115_123456",
    status=ExperimentStatus.RUNNING,
    config_hash="abc123...",        # For matching incomplete experiments
    num_processes=4,
    process_progress={              # Per-process tracking
        0: ProcessProgress(status=ProcessStatus.COMPLETED),
        1: ProcessProgress(status=ProcessStatus.RUNNING),
        ...
    },
)
```

Key properties:
- `can_aggregate()` - Check if all processes completed successfully
- `is_subprocess_running()` - Detect stale states (PID check)
- `processes_completed` / `processes_failed` - Progress counts

### StateManager

Persistent state operations with atomic file I/O:

```python
manager = StateManager(state_dir=Path(".experiment_state"))

# Create new experiment state
state = manager.create("exp_123", total_runs=4)

# Load existing state
state = manager.load("exp_123")

# Save state (atomic write)
manager.save(state)

# Find incomplete experiments
incomplete = manager.find_incomplete()

# Find experiment by config hash (for resumption)
state = manager.find_by_config_hash("abc123...")

# Clean up stale RUNNING states
cleaned = manager.cleanup_stale()
```

## State File Format

States are stored as JSON in `.experiment_state/`:

```
.experiment_state/
├── exp_20240115_123456.json
├── exp_20240116_091234.json
└── ...
```

## Resumption Flow

1. CLI receives `--resume <exp_id>` or detects matching config hash
2. `StateManager.load()` retrieves saved state
3. Orchestrator skips completed processes
4. On completion, state updated to `AGGREGATED`

## Stale State Detection

When an experiment is `RUNNING` but the subprocess PID no longer exists:

```python
state.is_subprocess_running()  # Returns False if PID dead
manager.cleanup_stale()        # Marks as INTERRUPTED
```

## Related

- See `../orchestration/README.md` for experiment lifecycle
- See `../cli.py` for `--resume` flag handling
- See `../security.py` for `sanitize_experiment_id()`
