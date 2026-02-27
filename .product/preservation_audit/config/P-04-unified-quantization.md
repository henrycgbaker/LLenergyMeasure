# P-04: Unified Quantization Abstraction

**Module**: `src/llenergymeasure/config/quantization.py`
**Risk Level**: HIGH
**Decision**: Pending — must decide: keep as translation layer (Path A) or backend-specific only (Path B)
**Planning Gap**: `designs/experiment-config.md` shows only backend-specific quantization fields; the unified cross-backend abstraction is completely absent.

---

## What Exists in the Code

**Primary file**: `src/llenergymeasure/config/quantization.py` (126 lines)
**Key classes**:
- `CalibrationConfig` (line 24) — PTQ calibration dataset, split, num_samples, max_length
- `UnifiedQuantizationConfig` (line 52) — the cross-backend abstraction:
  - `method: Literal["none", "int8", "int4", "fp8", "auto"]` (line 83) — user-level intent
  - `weight_only: bool = True` (line 87) — weight-only vs full quantization
  - `calibration: CalibrationConfig | None` (line 94) — PTQ config for TensorRT
  - `backend_method: str | None` (line 100) — escape hatch for explicit backend method (e.g., "gptq", "awq")
  - `@property enabled: bool` (line 106)
  - `@property bits: int | None` (line 112) — 8, 4, or None

Backend intent-to-method mapping (documented in module docstring, lines 7–14):

| User Intent | PyTorch | vLLM | TensorRT |
|-------------|---------|------|----------|
| `int8` | BitsAndBytes 8b | AWQ/SqueezeLLM | TRT INT8 (calib) |
| `int4` | BitsAndBytes 4b | GPTQ/AWQ | TRT INT4 (calib) |
| `fp8` | Not supported | FP8 (Hopper+) | TRT FP8 |
| `auto` | Detect from model | Detect | Detect from model |

## Why It Matters

Enables users to specify quantization intent once without knowing backend-specific method names (e.g., "int4" vs "gptq" vs "awq" vs "bitsandbytes"). A user writing a config that targets multiple backends can write `quantization: {method: int4}` and the translation layer maps it to the correct method per backend. Without it, multi-backend studies require duplicated backend-specific configs.

## Planning Gap Details

`designs/experiment-config.md` shows quantization only as backend-specific fields:
```yaml
vllm:
  quantization: awq   # only vLLM-specific
```
No unified `quantization:` block appears at the top-level `ExperimentConfig`. Two implementation paths are possible and both are defensible:

- **Path A (keep)**: `UnifiedQuantizationConfig` is an internal translation layer. Users write intent; at runtime, the backend loader maps it to backend-specific parameters.
- **Path B (remove)**: Drop unified abstraction. Users specify backend-specific quantization directly. Simpler but breaks the multi-backend config story.

The current codebase has invested in Path A. v2.0's composition model makes Path A slightly more complex (no shared base class to put it on).

## Recommendation for Phase 5

Resolve this decision explicitly before coding the config module. Add to `designs/experiment-config.md`:

> **Quantization**: [PATH A / PATH B — human decision needed]
>
> Path A (unified): `quantization: {method: int4}` in top-level `ExperimentConfig`. At runtime,
> the backend loader calls `UnifiedQuantizationConfig.to_backend_params(backend)` to produce
> backend-specific fields. Translation table: see `config/quantization.py` module docstring.
>
> Path B (backend-specific): Remove `UnifiedQuantizationConfig`. Users write:
> `pytorch: {load_in_4bit: true}` / `vllm: {quantization: "gptq"}` / `tensorrt: {quantization: "int4"}`.

If Path B chosen: delete `config/quantization.py` and update `config/models.py` accordingly.
