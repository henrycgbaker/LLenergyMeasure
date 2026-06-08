"""Round 0b bump-delta-recovery curve (deliverable B), offline.

For each consecutive bump-pair (A -> B) per engine: of the constraints that are
NEW in B's gate-confirmed GT vs A's (the changed/added surface), how many does
the UNCHANGED deterministic miner recover when run on B - with NO edit to its
code or landmark list? This is the self-updating property: a high recovery means
the cheap method tracks the bump without human intervention (the generalised-glob
primitive is the mechanism). Tolerant grain (leaf, coarse_bucket).
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, "research/mining-substrate-trial/scripts")
from strategies.wave2 import a_improved_det_v2 as v2

GT_ROOT = Path("research/mining-substrate-trial/findings/study/ground_truth")
# engine -> ordered [(version, vslug, src)]
WINDOWS = {
    "tensorrt": [
        ("v0.20.0", "v0_20_0", "/tmp/trt-llm-0.20.0/tensorrt_llm"),
        ("v0.21.0", "v0_21_0", "/tmp/trt-llm-0.21.0/tensorrt_llm"),
        ("v1.0.0", "v1_0_0", "/tmp/trt-llm-1.0.0/tensorrt_llm"),
        ("v1.1.0", "v1_1_0", "/tmp/trt-llm-1.1.0/tensorrt_llm"),
        ("v1.2.1", "v1_2_1", "/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm"),
    ],
    "vllm": [
        ("v0.18.1", "v0_18_1", "/tmp/vllm-0.18.1/vllm"),
        ("v0.19.1", "v0_19_1", "/tmp/vllm-0.19.1/vllm"),
        ("v0.20.0", "v0_20_0", "/tmp/vllm-0.20.0/vllm"),
        ("v0.21.0", "v0_21_0", "/tmp/vllm-0.21.0/vllm"),
        ("v0.22.0", "v0_22_0", "/tmp/vllm-0.22.0/vllm"),
    ],
    "transformers": [
        ("v5.6.2", "v5_6_2", "/tmp/tfvenv-5.6.2/lib/python3.12/site-packages/transformers"),
        ("v5.7.0", "v5_7_0", "/tmp/tfvenv-5.7.0/lib/python3.12/site-packages/transformers"),
        ("v5.8.1", "v5_8_1", "/tmp/tfvenv-5.8.1/lib/python3.12/site-packages/transformers"),
        ("v5.9.0", "v5_9_0", "/tmp/tfvenv-5.9.0/lib/python3.12/site-packages/transformers"),
        ("v5.10.2", "v5_10_2", "/tmp/tfvenv-5.10.2/lib/python3.12/site-packages/transformers"),
    ],
}


def gt_conf(engine, vslug):
    p = GT_ROOT / engine / vslug / "invariants" / "PILOT_GT.yaml"
    gt = yaml.safe_load(p.read_text()) or {}
    return {tuple(e["tolerant_key"]) for e in (gt.get("confirmed") or []) if e.get("tolerant_key")}


def mech(engine, version, src):
    out = v2.run_cell(
        engine=engine, version=version, engine_source_root=Path(src), task="invariants"
    )
    env = yaml.safe_load(Path(out.invariants_path).read_text()) or {}
    return {v2._tolerant_identity(i) for i in env.get("invariants", [])}


print(f"{'bump':28} {'added':>5} {'removed':>7} {'recov_added':>11} {'recovery%':>9}")
print("-" * 64)
overall = []
for engine, cells in WINDOWS.items():
    for (va, sa, _), (vb, sb, srcb) in zip(cells, cells[1:]):
        if not Path(srcb).exists():
            print(f"{engine + ' ' + va + '->' + vb:28} SRC MISSING")
            continue
        ga, gb = gt_conf(engine, sa), gt_conf(engine, sb)
        mb = mech(engine, vb, srcb)
        added = gb - ga
        removed = ga - gb
        recov = mb & added
        pct = 100.0 * len(recov) / len(added) if added else float("nan")
        if added:
            overall.append(pct)
        print(
            f"{engine + ' ' + va + '->' + vb:28} {len(added):5d} {len(removed):7d} {len(recov):11d} {pct:8.1f}"
        )
    print("-" * 64)
import math  # noqa: E402

vals = [p for p in overall if not math.isnan(p)]
print(
    f"mean bump-delta-recovery (added surface): {sum(vals) / len(vals):.1f}%  over {len(vals)} bumps"
)
