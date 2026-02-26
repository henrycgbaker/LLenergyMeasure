# N-X09: Protocols — Dependency Injection Interfaces

**Module**: `src/llenergymeasure/protocols.py`
**Risk Level**: HIGH
**Decision**: Keep — v2.0 (with updates to match the new architecture)
**Planning Gap**: `protocols.py` is not mentioned in any planning document. The `designs/architecture.md` module layout does not include it. Yet it defines the formal interfaces for all pluggable components (`ModelLoader`, `InferenceEngine`, `MetricsCollector`, `EnergyBackend`, `ResultsRepository`) — the backbone of the dependency injection design. A Phase 5 rebuild that ignores this file will likely recreate these interfaces informally or inconsistently.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/protocols.py`
**Key classes/functions** (all decorated `@runtime_checkable`):

- `ModelLoader` (line 21) — Protocol with one method:
  ```python
  def load(self, config: ExperimentConfig) -> tuple[Any, Any]: ...
  ```
  Returns `(model, tokenizer)`. Uses `Any` for both because PyTorch, vLLM, and TensorRT use incompatible types.

- `InferenceEngine` (line 36) — Protocol with one method:
  ```python
  def run(self, model: Any, tokenizer: Any, prompts: list[str], config: ExperimentConfig) -> Any: ...
  ```
  Returns `Any` (backend-specific result type, typically `InferenceMetrics` or `BackendInferenceResult`).

- `MetricsCollector` (line 62) — Protocol with one method:
  ```python
  def collect(self, model: Any, inference_result: Any, config: ExperimentConfig) -> CombinedMetrics: ...
  ```
  The return type is concrete (`CombinedMetrics`) unlike the inputs.

- `EnergyBackend` (line 85) — Protocol with three methods and one property:
  ```python
  @property
  def name(self) -> str: ...
  def start_tracking(self) -> Any: ...
  def stop_tracking(self, tracker: Any) -> EnergyMetrics: ...
  def is_available(self) -> bool: ...
  ```
  The `tracker` handle pattern enables stateful energy measurement without global state.

- `ResultsRepository` (line 125) — Protocol with four methods:
  ```python
  def save_raw(self, experiment_id: str, result: RawProcessResult) -> Path: ...
  def list_raw(self, experiment_id: str) -> list[Path]: ...
  def load_raw(self, path: Path) -> RawProcessResult: ...
  def save_aggregated(self, result: AggregatedResult) -> Path: ...
  ```
  Supports the late-aggregation pattern: raw per-process results saved separately, aggregated on demand.

All five protocols are decorated `@runtime_checkable`, enabling `isinstance(obj, ModelLoader)` checks at runtime — important for factory functions and validation.

## Why It Matters

The `EnergyBackend` Protocol is explicitly referenced in `designs/energy-backends.md` as the interface that `NVMLBackend`, `ZeusBackend`, and `CodeCarbonBackend` must implement. If the protocol definition drifts from the current file during Phase 5, the energy backend plugin system breaks. The `ResultsRepository` protocol defines the contract that `FileSystemRepository` implements — changing it without updating the protocol creates silent interface violations that only surface at runtime. The `@runtime_checkable` decoration means these are not just type hints — they are used for `isinstance` checks in factory code.

## Planning Gap Details

`designs/energy-backends.md` (lines 12–18) reproduces the `EnergyBackend` protocol verbatim:
```python
class EnergyBackend(Protocol):
    @property
    def name(self) -> str: ...
    def start_tracking(self) -> Any: ...
    def stop_tracking(self, tracker: Any) -> EnergyMetrics: ...
    def is_available(self) -> bool: ...
```
This confirms the protocol is expected to survive — but the document does not say "this is in `protocols.py`" nor that the other four protocols exist alongside it.

`designs/architecture.md` lists `core/energy/protocol.py` and `core/backends/protocol.py` as separate files in the v2.0 layout — suggesting the protocols will be split and co-located with their domain. This is an architectural change from the current single `protocols.py`. The `@runtime_checkable` behaviour and method signatures must be preserved regardless of file location.

## Recommendation for Phase 5

If the v2.0 architecture co-locates protocols with their domain (per `designs/architecture.md`):
- `core/backends/protocol.py` — `ModelLoader` + `InferenceEngine`
- `core/energy/protocol.py` — `EnergyBackend`
- `results/protocol.py` (new) — `ResultsRepository`
- `core/metrics_protocol.py` (or similar) — `MetricsCollector`

If keeping a single `protocols.py` for simplicity: this is also fine — co-location is an organisation preference, not a correctness requirement.

Either way, preserve:
1. All five protocol definitions with their exact method signatures
2. The `@runtime_checkable` decorator on all five
3. The `Any` typing on model/tokenizer parameters (this is intentional — backends use different types)
4. The `tracker: Any` pattern in `EnergyBackend` (the tracker is backend-specific; cannot be typed more precisely without generics)

Add `protocols.py` (or the split equivalents) to the module layout in `designs/architecture.md`.
