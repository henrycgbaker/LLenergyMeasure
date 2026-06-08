"""Round 0b deterministic SURFACING-recall (offline, no container).

For each cell: does the cheap deterministic miner (improved-det-v2 + Round 0b
primitives) SURFACE the constraints the gate-confirmed GT contains? Recall =
|mech tolerant-keys INTERSECT GT-confirmed tolerant-keys| / |GT-confirmed|.
This is the deterministic frontier point; it does NOT require mech candidates to
self-gate (the GT already validated each constraint via the union)."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, "research/mining-substrate-trial/scripts")
from strategies.wave2 import a_improved_det_v2 as v2

GT_ROOT = Path("research/mining-substrate-trial/findings/study/ground_truth")
CELLS = [
    ("tensorrt", "v1.2.1", "v1_2_1", "/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm"),
    ("tensorrt", "v0.20.0", "v0_20_0", "/tmp/trt-llm-0.20.0/tensorrt_llm"),
    ("tensorrt", "v0.21.0", "v0_21_0", "/tmp/trt-llm-0.21.0/tensorrt_llm"),
    ("tensorrt", "v1.0.0", "v1_0_0", "/tmp/trt-llm-1.0.0/tensorrt_llm"),
    ("tensorrt", "v1.1.0", "v1_1_0", "/tmp/trt-llm-1.1.0/tensorrt_llm"),
    ("vllm", "v0.18.1", "v0_18_1", "/tmp/vllm-0.18.1/vllm"),
    ("vllm", "v0.19.1", "v0_19_1", "/tmp/vllm-0.19.1/vllm"),
    ("vllm", "v0.20.0", "v0_20_0", "/tmp/vllm-0.20.0/vllm"),
    ("vllm", "v0.21.0", "v0_21_0", "/tmp/vllm-0.21.0/vllm"),
    ("vllm", "v0.22.0", "v0_22_0", "/tmp/vllm-0.22.0/vllm"),
    (
        "transformers",
        "v5.6.2",
        "v5_6_2",
        "/tmp/tfvenv-5.6.2/lib/python3.12/site-packages/transformers",
    ),
    (
        "transformers",
        "v5.7.0",
        "v5_7_0",
        "/tmp/tfvenv-5.7.0/lib/python3.12/site-packages/transformers",
    ),
    (
        "transformers",
        "v5.8.1",
        "v5_8_1",
        "/tmp/tfvenv-5.8.1/lib/python3.12/site-packages/transformers",
    ),
    (
        "transformers",
        "v5.9.0",
        "v5_9_0",
        "/tmp/tfvenv-5.9.0/lib/python3.12/site-packages/transformers",
    ),
    (
        "transformers",
        "v5.10.2",
        "v5_10_2",
        "/tmp/tfvenv-5.10.2/lib/python3.12/site-packages/transformers",
    ),
]


def mech_tkeys(engine, version, src):
    out = v2.run_cell(
        engine=engine, version=version, engine_source_root=Path(src), task="invariants"
    )
    env = yaml.safe_load(Path(out.invariants_path).read_text()) or {}
    return {v2._tolerant_identity(i) for i in env.get("invariants", [])}, len(
        env.get("invariants", [])
    )


def gt_conf_tkeys(engine, vslug):
    p = GT_ROOT / engine / vslug / "invariants" / "PILOT_GT.yaml"
    gt = yaml.safe_load(p.read_text()) or {}
    return {tuple(e["tolerant_key"]) for e in (gt.get("confirmed") or []) if e.get("tolerant_key")}


print(f"{'cell':22} {'GTconf':>6} {'mech':>5} {'recov':>6} {'recall%':>8} {'prec%':>6}")
print("-" * 60)
agg = {}
for engine, version, vslug, src in CELLS:
    if not Path(src).exists():
        print(f"{engine + ' ' + version:22} SOURCE MISSING")
        continue
    try:
        mk, n_raw = mech_tkeys(engine, version, src)
        gk = gt_conf_tkeys(engine, vslug)
        recov = mk & gk
        recall = 100.0 * len(recov) / len(gk) if gk else 0.0
        prec = 100.0 * len(recov) / len(mk) if mk else 0.0
        agg.setdefault(engine, []).append(recall)
        print(
            f"{engine + ' ' + version:22} {len(gk):6d} {len(mk):5d} {len(recov):6d} {recall:7.1f} {prec:5.1f}"
        )
    except Exception as exc:
        print(f"{engine + ' ' + version:22} ERROR {exc}")
print("-" * 60)
for eng, rs in agg.items():
    print(f"{eng:22} mean recall {sum(rs) / len(rs):.1f}%  (n={len(rs)})")
