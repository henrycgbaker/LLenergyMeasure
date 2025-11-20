# Persistent Progress Trackers

This directory contains files and utilities for tracking experiment progress across sessions. The progress tracking system enables resumability for long-running experimental suites that may span multiple days or require multiple restarts.

## Overview

The progress tracking system maintains three key pieces of state:
1. **Experiment ID Counter**: Auto-incrementing unique identifier for each experiment
2. **Cycle ID Counter**: Tracks which experimental cycle is currently running
3. **Configuration Completion Status**: Records which configurations have been completed

## Files

### `experiment_id.txt`
Stores the current experiment ID counter.

**Format:** Single integer on one line
```
4459
```

**Usage:**
- Read current ID
- Increment counter
- Write back to file
- Ensures atomicity across distributed processes

**Example:**
```python
def get_next_experiment_id():
    with open("persistent_progress_trackers/experiment_id.txt", "r") as f:
        current_id = int(f.read().strip())

    next_id = current_id + 1

    with open("persistent_progress_trackers/experiment_id.txt", "w") as f:
        f.write(str(next_id))

    return f"{next_id:04d}"  # Returns "4460"
```

**Concurrency:**
In distributed settings, only the main process (rank 0) should read/write this file. The ID is then broadcast to all other processes.

---

### `cycle_id.txt`
Stores the current experimental cycle number.

**Format:** Single integer on one line
```
2
```

**Purpose:**
Experiments are often run in multiple cycles to:
- Collect statistical samples
- Verify reproducibility
- Account for variance in measurements
- Build confidence intervals

**Usage:**
```python
def get_current_cycle():
    with open("persistent_progress_trackers/cycle_id.txt", "r") as f:
        return int(f.read().strip())

def increment_cycle():
    cycle = get_current_cycle()
    with open("persistent_progress_trackers/cycle_id.txt", "w") as f:
        f.write(str(cycle + 1))
```

**Example Multi-Cycle Workflow:**
```python
num_cycles = 5
for cycle in range(1, num_cycles + 1):
    # Update cycle file
    with open("persistent_progress_trackers/cycle_id.txt", "w") as f:
        f.write(str(cycle))

    # Run all configurations for this cycle
    for config in all_configs:
        config["cycle_id"] = cycle
        run_experiment(config)
```

---

### `configs_run_progress.json`
Large JSON file (531KB+) tracking completion status of all configuration-cycle combinations.

**Format:**
```json
{
    "model_name::suite_name::config_name::cycle_id": "experiment_id",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0::controlled::batching_16::1": "0001",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0::controlled::batching_32::1": "0002",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0::controlled::batching_16::2": "0458",
    "meta-llama/Llama-3.2-1B::controlled::precis_float16::1": "0123",
    "meta-llama/Llama-3.2-1B::controlled::precis_float16::2": null
}
```

**Key Structure:**
```
{model_name}::{suite}::{config_name}::{cycle_id}
```

**Value:**
- `"experiment_id"` (string): ID if completed
- `null`: Not yet completed

**Purpose:**
1. **Resume Capability**: Skip already-completed experiments when restarting
2. **Progress Tracking**: Monitor how many configs are done
3. **Failure Recovery**: Identify which configs failed (missing from progress)
4. **Multi-Cycle Management**: Track completion per cycle independently

---

## Usage Patterns

### Loading Progress

```python
import json

def load_progress():
    try:
        with open("persistent_progress_trackers/configs_run_progress.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
```

### Saving Progress

```python
def save_progress(progress_dict):
    with open("persistent_progress_trackers/configs_run_progress.json", "w") as f:
        json.dump(progress_dict, f, indent=2)
```

### Checking if Configuration is Complete

```python
def is_config_complete(model_name, suite, config_name, cycle_id, progress):
    key = f"{model_name}::{suite}::{config_name}::{cycle_id}"
    return progress.get(key) is not None
```

### Marking Configuration as Complete

```python
def mark_config_complete(model_name, suite, config_name, cycle_id, experiment_id, progress):
    key = f"{model_name}::{suite}::{config_name}::{cycle_id}"
    progress[key] = experiment_id
    save_progress(progress)
```

### Complete Workflow Example

```python
import json
from configs.c_controlled_configs import get_controlled_configs
from experiment_orchestration_utils.b_single_config_workflow import run_single_config_with_retry

# Load progress
with open("persistent_progress_trackers/configs_run_progress.json", "r") as f:
    progress = json.load(f)

# Get current cycle
with open("persistent_progress_trackers/cycle_id.txt", "r") as f:
    cycle_id = int(f.read().strip())

# Get configurations
configs = get_controlled_configs()

for config in configs:
    # Inject cycle ID
    config["cycle_id"] = cycle_id

    # Build progress key
    key = f"{config['model_name']}::{config['suite']}::{config['config_name']}::{cycle_id}"

    # Check if already completed
    if progress.get(key) is not None:
        print(f"✓ Skipping {key} (already completed as experiment {progress[key]})")
        continue

    # Run experiment
    print(f"▶ Running {key}...")
    success = run_single_config_with_retry(config, max_retries=3)

    if success:
        # Get experiment ID from runner
        experiment_id = config.get("experiment_id", "unknown")

        # Mark as complete
        progress[key] = experiment_id
        save_progress(progress)

        print(f"✓ Completed {key} as experiment {experiment_id}")
    else:
        print(f"✗ Failed {key} after all retries")
```

---

## Progress Tracking Utilities

### Generating Progress Keys

```python
def generate_progress_key(config):
    """Generate a unique key for progress tracking."""
    return f"{config['model_name']}::{config['suite']}::{config['config_name']}::{config['cycle_id']}"
```

### Counting Completed Configurations

```python
def count_completed(progress):
    """Count how many configs are complete."""
    return sum(1 for v in progress.values() if v is not None)

def count_total(progress):
    """Count total configs tracked."""
    return len(progress)

def get_completion_percentage(progress):
    """Calculate completion percentage."""
    total = count_total(progress)
    if total == 0:
        return 0.0
    completed = count_completed(progress)
    return (completed / total) * 100
```

### Finding Incomplete Configurations

```python
def get_incomplete_configs(progress):
    """Get list of incomplete configuration keys."""
    return [key for key, value in progress.items() if value is None]

def get_incomplete_for_cycle(progress, cycle_id):
    """Get incomplete configs for a specific cycle."""
    return [
        key for key, value in progress.items()
        if value is None and key.endswith(f"::{cycle_id}")
    ]
```

### Progress Statistics

```python
def get_progress_stats(progress):
    """Get comprehensive progress statistics."""
    total = len(progress)
    completed = sum(1 for v in progress.values() if v is not None)
    incomplete = total - completed

    # Group by cycle
    cycles = {}
    for key, value in progress.items():
        cycle = key.split("::")[-1]
        if cycle not in cycles:
            cycles[cycle] = {"total": 0, "completed": 0}
        cycles[cycle]["total"] += 1
        if value is not None:
            cycles[cycle]["completed"] += 1

    return {
        "total_configs": total,
        "completed": completed,
        "incomplete": incomplete,
        "completion_percentage": (completed / total * 100) if total > 0 else 0,
        "cycles": cycles
    }
```

**Usage:**
```python
stats = get_progress_stats(progress)
print(f"Progress: {stats['completed']}/{stats['total_configs']} "
      f"({stats['completion_percentage']:.1f}%)")

for cycle, cycle_stats in stats['cycles'].items():
    print(f"  Cycle {cycle}: {cycle_stats['completed']}/{cycle_stats['total']}")
```

---

## Multi-Cycle Experiments

### Why Multiple Cycles?

1. **Statistical Significance**: Multiple runs provide mean and variance
2. **Reproducibility**: Verify results are consistent across runs
3. **Outlier Detection**: Identify anomalous results
4. **Confidence Intervals**: Build confidence in measurements
5. **Variance Analysis**: Understand variability in metrics

### Typical Cycle Structure

```python
# Configuration
NUM_CYCLES = 5

# Get all configurations
all_configs = (
    get_model_configs() +
    get_controlled_configs() +
    get_scenario_configs() +
    get_grid_search_configs()
)

# Run experiments
for cycle in range(1, NUM_CYCLES + 1):
    print(f"\n{'='*50}")
    print(f"CYCLE {cycle}/{NUM_CYCLES}")
    print(f"{'='*50}\n")

    # Update cycle file
    with open("persistent_progress_trackers/cycle_id.txt", "w") as f:
        f.write(str(cycle))

    # Load progress
    progress = load_progress()

    # Run all configs for this cycle
    for config in all_configs:
        config["cycle_id"] = cycle
        key = generate_progress_key(config)

        if progress.get(key):
            continue  # Already done

        # Run experiment
        success = run_single_config_with_retry(config)

        if success:
            progress[key] = config["experiment_id"]
            save_progress(progress)

    # Cycle complete
    print(f"\nCycle {cycle} complete!")
    stats = get_progress_stats(progress)
    print(f"Overall progress: {stats['completion_percentage']:.1f}%")
```

---

## Distributed Synchronization

### Important: Main Process Only

Only the main process (rank 0) should read/write progress files to avoid race conditions.

```python
from accelerate import Accelerator

accelerator = Accelerator()

if accelerator.is_main_process:
    # Only main process manages progress
    progress = load_progress()

    # Generate experiment ID
    with open("persistent_progress_trackers/experiment_id.txt", "r") as f:
        exp_id = int(f.read().strip()) + 1
    with open("persistent_progress_trackers/experiment_id.txt", "w") as f:
        f.write(str(exp_id))

    # Broadcast to all processes
    exp_id = [exp_id]
else:
    exp_id = [None]

# Broadcast experiment ID
torch.distributed.broadcast_object_list(exp_id, src=0)
experiment_id = f"{exp_id[0]:04d}"

# ... run experiment ...

# Only main process updates progress
if accelerator.is_main_process:
    key = generate_progress_key(config)
    progress[key] = experiment_id
    save_progress(progress)
```

---

## Recovery from Failures

### Identifying Failed Experiments

Failed experiments are those initialized in progress but never completed:

```python
def find_failed_experiments(progress, expected_configs):
    """
    Find configs that should exist but are missing or None.

    Args:
        progress: Current progress dict
        expected_configs: List of all config keys that should exist

    Returns:
        List of failed config keys
    """
    failed = []
    for expected_key in expected_configs:
        if expected_key not in progress:
            failed.append(expected_key)
        elif progress[expected_key] is None:
            failed.append(expected_key)
    return failed
```

### Re-running Failed Experiments

```python
# Identify failures
expected = [generate_progress_key(c) for c in all_configs]
failed = find_failed_experiments(progress, expected)

print(f"Found {len(failed)} failed experiments. Re-running...")

for key in failed:
    # Parse key back to config parameters
    model, suite, config_name, cycle = key.split("::")

    # Reconstruct config
    config = find_config(model, suite, config_name)
    config["cycle_id"] = int(cycle)

    # Re-run
    success = run_single_config_with_retry(config)

    if success:
        progress[key] = config["experiment_id"]
        save_progress(progress)
```

---

## File Size Management

The `configs_run_progress.json` file can grow large (>500KB) with thousands of configurations and multiple cycles.

### Current Size: ~531KB
- ~4,459 experiments tracked
- Multiple cycles
- Multiple configuration suites

### Optimization Strategies

#### 1. Periodic Archiving
```python
import shutil
from datetime import datetime

def archive_old_progress():
    """Archive completed progress to separate file."""
    progress = load_progress()

    # Separate complete and incomplete
    complete = {k: v for k, v in progress.items() if v is not None}
    incomplete = {k: v for k, v in progress.items() if v is None}

    # Archive complete
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"persistent_progress_trackers/archive/progress_{timestamp}.json"
    with open(archive_path, "w") as f:
        json.dump(complete, f, indent=2)

    # Keep only incomplete
    save_progress(incomplete)
```

#### 2. Compression
```python
import gzip
import json

def save_progress_compressed(progress):
    """Save progress with gzip compression."""
    with gzip.open("persistent_progress_trackers/configs_run_progress.json.gz", "wt") as f:
        json.dump(progress, f)

def load_progress_compressed():
    """Load compressed progress."""
    with gzip.open("persistent_progress_trackers/configs_run_progress.json.gz", "rt") as f:
        return json.load(f)
```

#### 3. Database Backend
For very large experiments (10,000+ configs), consider using SQLite:

```python
import sqlite3

def init_progress_db():
    """Initialize SQLite database for progress tracking."""
    conn = sqlite3.connect("persistent_progress_trackers/progress.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            key TEXT PRIMARY KEY,
            model_name TEXT,
            suite TEXT,
            config_name TEXT,
            cycle_id INTEGER,
            experiment_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def mark_complete_db(conn, key, experiment_id):
    """Mark configuration as complete in database."""
    model, suite, config, cycle = key.split("::")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO progress
        (key, model_name, suite, config_name, cycle_id, experiment_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (key, model, suite, config, int(cycle), experiment_id))
    conn.commit()
```

---

## Monitoring Progress

### Real-Time Progress Display

```python
import time

def monitor_progress(progress_file, update_interval=60):
    """Monitor and display progress in real-time."""
    while True:
        progress = load_progress()
        stats = get_progress_stats(progress)

        print(f"\n{'='*60}")
        print(f"Experiment Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Total Configurations: {stats['total_configs']}")
        print(f"Completed: {stats['completed']} ({stats['completion_percentage']:.1f}%)")
        print(f"Remaining: {stats['incomplete']}")

        print(f"\nPer-Cycle Breakdown:")
        for cycle, cycle_stats in sorted(stats['cycles'].items()):
            pct = (cycle_stats['completed'] / cycle_stats['total'] * 100)
            print(f"  Cycle {cycle}: {cycle_stats['completed']}/{cycle_stats['total']} ({pct:.1f}%)")

        time.sleep(update_interval)
```

### Progress Bar Integration

```python
from tqdm import tqdm

def run_experiments_with_progress_bar(configs, progress):
    """Run experiments with visual progress bar."""
    incomplete = [c for c in configs if not is_config_complete(c, progress)]

    with tqdm(total=len(incomplete), desc="Experiments") as pbar:
        for config in incomplete:
            success = run_single_config_with_retry(config)
            if success:
                key = generate_progress_key(config)
                progress[key] = config["experiment_id"]
                save_progress(progress)
                pbar.update(1)
```

---

## Best Practices

1. **Always Load Progress Before Starting**: Avoid re-running completed experiments
2. **Save Progress Immediately After Completion**: Don't wait to batch updates
3. **Use Main Process Only**: Prevent race conditions in distributed settings
4. **Handle Corruption Gracefully**: Keep backup copies of progress files
5. **Archive Completed Progress**: Keep working file size manageable
6. **Log Progress Updates**: Track when configs are marked complete
7. **Validate Progress Keys**: Ensure consistent key format across codebase
8. **Monitor Disk Space**: Large progress files can fill up disk

---

## Backup and Recovery

### Automatic Backups

```python
import shutil
from datetime import datetime

def backup_progress():
    """Create timestamped backup of progress file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"persistent_progress_trackers/backups/progress_backup_{timestamp}.json"

    shutil.copy(
        "persistent_progress_trackers/configs_run_progress.json",
        backup_path
    )

    print(f"Progress backed up to {backup_path}")

# Run backup before starting large experimental suite
backup_progress()
```

### Restoring from Backup

```python
def restore_progress_from_backup(backup_path):
    """Restore progress from a backup file."""
    shutil.copy(
        backup_path,
        "persistent_progress_trackers/configs_run_progress.json"
    )
    print(f"Progress restored from {backup_path}")
```

---

## Future Improvements

1. **Database Backend**: Migrate to SQLite for better performance with large progress files
2. **Web Dashboard**: Real-time visualization of experimental progress
3. **Distributed Locking**: Support multiple concurrent experiment runners
4. **Progress API**: REST API for querying and updating progress
5. **Cloud Sync**: Automatic backup to cloud storage (S3, GCS)
6. **Versioning**: Track progress file versions for rollback capability
7. **Conflict Resolution**: Handle progress conflicts in distributed scenarios

---

## Author

Henry Baker
