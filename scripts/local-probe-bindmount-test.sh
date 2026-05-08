#!/usr/bin/env bash
# Local PoC harness - verify probe --output writes host-readable JSON via bind mount.
#
# Catches the entire class of Docker bind-mount permission issues that bit
# us repeatedly while iterating on engine-image-build.yml. Runs in ~10 sec
# on any host with Docker. Use BEFORE pushing changes to scripts/_probe.py
# or to the workflow probe-step pattern.
#
# Usage: bash scripts/local-probe-bindmount-test.sh
#
# Exits 0 on success, non-zero on failure. Prints diagnostic info.

set -euo pipefail

OUT_DIR="/tmp/llem-probe-poc-$$"
mkdir -p "$OUT_DIR"
trap 'rm -rf "$OUT_DIR"' EXIT

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== local probe bind-mount PoC ==="
echo "Repo root:   $REPO_ROOT"
echo "Out dir:     $OUT_DIR"
echo "Host UID/GID: $(id -u)/$(id -g)"
echo

# Run the probe inside a minimal python container, mirroring the CI's docker
# run shape: container as root, host bind mount, --output to bind-mounted dir.
docker run --rm \
  -v "$REPO_ROOT:/repo:ro" \
  -v "$OUT_DIR:/probe-out:rw" \
  -e PYTHONPATH=/repo/src:/repo \
  -w /repo \
  --entrypoint python3 \
  python:3.12-slim \
  -m pip install --quiet pyyaml packaging 2>&1 | tail -2

docker run --rm \
  -v "$REPO_ROOT:/repo:ro" \
  -v "$OUT_DIR:/probe-out:rw" \
  -e PYTHONPATH=/repo/src:/repo \
  -w /repo \
  --entrypoint python3 \
  python:3.12-slim \
  -c "
import subprocess, sys
subprocess.check_call(['pip', 'install', '--quiet', 'pyyaml', 'packaging'])
sys.path.insert(0, '/repo')
from scripts import _probe
from scripts._probe import ProbeReport, _write_report_to_file
from pathlib import Path
r = ProbeReport(
    engine='test', producer='invariants', verdict='pass',
    library_version='9.9.9', version_inside_envelope=True,
    fingerprint='deadbeef', fingerprint_drift=[], landmarks_missing=[],
)
out = Path('/probe-out/probe-test.json')
_write_report_to_file(out, r)
print('container view:', oct(out.stat().st_mode & 0o777), 'size=', out.stat().st_size)
"

echo
echo "=== host-side verification ==="
ls -la "$OUT_DIR/"
echo
HOST_MODE=$(stat -c '%a' "$OUT_DIR/probe-test.json")
HOST_OWNER=$(stat -c '%U:%G' "$OUT_DIR/probe-test.json")
echo "host mode:    $HOST_MODE"
echo "host owner:   $HOST_OWNER"
echo

# Verify host can read (the actual bug we keep regressing on)
if jq -r '.verdict' "$OUT_DIR/probe-test.json" > /dev/null 2>&1; then
  VERDICT=$(jq -r '.verdict' "$OUT_DIR/probe-test.json")
  echo "✅ host can read probe JSON (jq verdict=$VERDICT)"
else
  echo "❌ host CANNOT read probe JSON via bind mount" >&2
  echo "   This is the bug class that broke PR #530 invariants/schemas runs." >&2
  echo "   The probe writes to /probe-out as root with mkstemp's default 0600;" >&2
  echo "   host (non-root) gets Permission denied. Fix: os.chmod(0o644)." >&2
  exit 1
fi

# Mode check - must be world-readable
if [[ "$HOST_MODE" != "644" ]]; then
  echo "❌ file mode is $HOST_MODE, expected 644" >&2
  echo "   The chmod fix in _write_report_to_file may have regressed." >&2
  exit 1
fi
echo "✅ file mode is 644 (host-readable)"

echo
echo "=== ALL CHECKS PASSED ==="
