# Phase 3: Parameter Completeness - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Expand backend parameter coverage to 90%+ of energy/throughput-impactful parameters for PyTorch, vLLM, and TensorRT. Add escape hatch for undocumented/niche parameters. Ensure SSOT introspection auto-discovers new parameters and generates documentation automatically.

</domain>

<decisions>
## Implementation Decisions

### Parameter Gap Strategy
- Identify missing params via systematic backend docs audit (PyTorch, vLLM, TensorRT official documentation)
- Prioritise all backends equally (parallel audit, not sequential)
- Add params to schema with documented constraints in docstrings (e.g., "Requires Ampere+ GPU", "Only for MoE models")
- Runtime warnings handled by existing smoke test infrastructure from Phase 2.4

### Extra/Escape Hatch Design
- Include `extra:` escape hatch for power users and research edge cases
- Strict passthrough: `extra:` dict passed directly to backend with no validation
- Claude's discretion on config structure (per-backend vs top-level placement)

### Audit Campaign Output
- Both levels: summary coverage table AND per-param detail breakdown
- Runtime validation: actually run inference with each param combination via smoke tests
- Results regenerate `docs/parameter-support-matrix.md` (existing file, updated from test results)
- Params that work but emit warnings: mark as "supported (with warnings)" — distinct status

### Documentation Approach
- Keep two docs separate: `config-reference.md` (schema) and `parameter-support-matrix.md` (runtime results)
- Both docs fully auto-generated from SSOT (Pydantic models + introspection.py)
- Doc regeneration via pre-commit hook (auto-regenerate when Pydantic models change)
- Coverage % mentioned subtly in overview section, not prominently badged

### Example Configs
- After final audit: fully develop example configs to showcase each backend's capabilities
- Each backend example should demonstrate all supported parameters (not just basics)
- Serves as living documentation of what each backend can do

### Claude's Discretion
- Exact placement of `extra:` field in config structure (per-backend vs top-level)
- Which specific params to add (based on docs audit findings)
- Pre-commit hook implementation details
- Audit script invocation pattern

</decisions>

<specifics>
## Specific Ideas

- "90%+ coverage" targets from ROADMAP: PyTorch 95%+ (from 93.8%), vLLM 90%+ (from 81.9%), TensorRT 95%+ (from 93.8%)
- Existing `test_all_params.py --discover` mode should pick up all params from SSOT
- Address divergence between parameter-support-matrix.md (GPU-tested only) and config-reference.md (all Pydantic fields)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-parameter-completeness*
*Context gathered: 2026-02-04*
