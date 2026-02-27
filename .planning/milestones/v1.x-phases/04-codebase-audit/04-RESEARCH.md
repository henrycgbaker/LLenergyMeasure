# Phase 4: Codebase Audit - Research

**Researched:** 2026-02-05
**Domain:** Codebase audit methodology for research benchmarking tools
**Confidence:** MEDIUM

## Summary

This research investigated how to conduct a systematic codebase audit for a Python-based LLM benchmarking tool. The audit domain is unique—it's not about implementing a feature, but about systematically evaluating code quality, identifying technical debt, and comparing against industry standards in similar research tools.

Research focused on three key areas: (1) how comparable tools (lm-evaluation-harness, vLLM benchmarks, nanoGPT, LLMPerf) structure their codebases and CLIs, (2) what tooling exists for automated code analysis in Python, and (3) what patterns define successful research tool architecture.

Key findings show research tools strongly favor simplicity: nanoGPT's ~750 lines, LLMPerf's 2-script approach, lm-eval-harness's YAML-first configuration. Most use minimal CLI surfaces (2-4 main commands), single-script execution patterns, and straightforward config-via-YAML rather than complex orchestration. Docker usage is targeted (for serving/inference), not universal.

**Primary recommendation:** Audit should use research-tool benchmarks (lm-eval-harness, nanoGPT, vLLM) as simplicity targets, employ automated tools (vulture, deadcode) for dead code detection, and systematically compare each aspect (CLI, config, Docker usage, execution paths) against industry norms with concrete "they do X, we do Y" evidence.

## Standard Stack

### Core Audit Tools

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| vulture | Latest | Dead code detection | Most popular Python unused code detector; uses AST analysis |
| deadcode | Latest | Unused code detection | More comprehensive rules than vulture; includes --fix option |
| ast module | stdlib | Code introspection | Python standard library; basis for all static analysis |
| ruff | Latest | Linting + complexity | Fast, comprehensive Python linter with complexity metrics |
| grep/ripgrep | Any | Pattern search | Quick identification of specific patterns across codebase |

### Supporting Analysis Tools

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| Semgrep | Latest | Pattern-based scanning | Custom rule detection (API patterns, architecture violations) |
| MyPy | Latest | Type coverage analysis | Check which code paths have type annotations |
| Bandit | Latest | Security pattern detection | Identify security anti-patterns (if security is a concern) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|-----------|-----------|----------|
| vulture + deadcode | SonarQube | SonarQube is enterprise-focused, requires server; tools are simpler for one-time audit |
| Manual comparison | Automated benchmarking | No good automated tools for "compare my tool to lm-eval-harness"; manual is necessary |

**Installation:**
```bash
pip install vulture deadcode ruff semgrep mypy bandit
```

## Architecture Patterns

### Reference Tool Comparison Matrix

Research benchmarking tools follow consistent patterns. Use this matrix for comparison:

| Aspect | lm-eval-harness | vLLM benchmarks | nanoGPT | LLMPerf | Pattern |
|--------|----------------|-----------------|---------|---------|---------|
| **CLI structure** | Subcommand-based (`lm-eval run/ls/validate`) | Script-per-function (`benchmark_serving.py`) | Single scripts | Script-per-function | 2-4 main commands OR script-per-task |
| **Config format** | YAML (via --config) | CLI args + Python files | Python config files | CLI args only | YAML primary, Python for advanced |
| **Execution model** | Direct Python execution | Direct Python scripts | Direct Python (`train.py`, `sample.py`) | Direct Python scripts | No complex orchestration; run scripts directly |
| **Docker usage** | Optional (for serving) | For serving benchmarks | Not used | Not used | Docker for serving/inference, not required for tool |
| **Result output** | CSV + console table | JSON + results directory | Checkpoints to disk | CSV + JSON to results-dir | Simple file output to directory |
| **Campaign/grid** | Not present | Not present | Not present | Parameters via CLI args | No built-in campaign orchestration; users script it |

**Key insight:** Research tools are script-oriented, not orchestration-oriented. Users run individual experiments and aggregate results externally.

### Pattern 1: Subcommand CLI vs Script-per-Function

**What:** Tools use either unified CLI with subcommands OR separate scripts for each task.

**When to use:**
- **Subcommand:** When tool provides 3-5 tightly integrated operations (lm-eval: run/ls/validate)
- **Script-per-function:** When operations are independent or specialized (vLLM: serving vs throughput vs latency)

**Example from lm-eval-harness:**
```bash
# Subcommand pattern
lm-eval run --model hf --tasks hellaswag --batch_size 4
lm-eval ls tasks
lm-eval validate --config my_config.yaml
```

**Example from vLLM:**
```bash
# Script-per-function pattern
python benchmarks/benchmark_serving.py --model meta-llama/Llama-2-7b-chat-hf --backend vllm
python benchmarks/benchmark_throughput.py --model meta-llama/Llama-2-7b-chat-hf
python benchmarks/benchmark_latency.py --model meta-llama/Llama-2-7b-chat-hf
```

### Pattern 2: Configuration Hierarchy

**What:** YAML for standard config, Python for advanced/dynamic config, CLI args for overrides.

**Hierarchy (from lm-eval-harness, DeepSpeed):**
1. YAML config file (shareable, reproducible)
2. Python config files (dynamic, programmable)
3. CLI args override both

**Example from lm-eval-harness:**
```bash
# YAML config approach
lm-eval run --config experiment.yaml

# Or all via CLI
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --batch_size 4
```

**Key principle:** Config files are for reproducibility; CLI args are for quick iteration.

### Pattern 3: Direct Execution Model

**What:** Users run Python scripts directly; no daemon/server/orchestrator.

**Why it matters:** Research tools prioritize transparency and hackability over abstraction.

**From nanoGPT:**
```bash
# Training - direct Python execution
python train.py config/train_shakespeare_char.py

# Sampling - direct Python execution
python sample.py --out_dir=out-shakespeare-char
```

**Anti-pattern seen in complex tools:** Multiple execution modes (local vs Docker, daemon vs direct, API vs CLI) that diverge in behavior.

### Pattern 4: Docker for Serving, Not Tool Operation

**What:** Docker used for model serving/inference backends, not for wrapping the benchmarking tool itself.

**Evidence:**
- **vLLM benchmarks:** Docker for vLLM server, benchmark script runs locally
- **lm-eval-harness:** Optional Docker for certain model backends
- **nanoGPT:** No Docker
- **LLMPerf:** No Docker for tool; tests against remote serving endpoints

**When Docker appears:** Inference serving (vLLM, TensorRT) requires containerization; benchmarking tool does not.

### Anti-Patterns to Avoid

- **Campaign orchestration in core tool:** Similar tools don't have built-in campaign/grid systems; users script sweeps externally
- **Complex execution paths:** Multiple code paths (local, Docker, daemon) that behave differently
- **Over-abstraction of backends:** Research tools call inference APIs directly; minimal wrapper layers
- **Config generation embedded in execution:** Config generation and experiment execution are typically separate concerns
- **Result aggregation as core feature:** Tools output raw results; analysis/aggregation happens in separate scripts or notebooks

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dead code detection | Manual grep for imports | `vulture` or `deadcode` packages | AST-based analysis catches unreachable code, unused classes/methods, confidence scoring |
| Code complexity metrics | Custom LOC counters | `ruff` or `mccabe` | Cyclomatic complexity, cognitive complexity metrics validated by research |
| Unused imports | Visual inspection | `ruff check --select F401` | Deterministic, fast, covers all edge cases |
| Pattern detection across codebase | Regex in bash loops | `semgrep` with custom rules | AST-aware, language-agnostic, shareable rules |
| Config format comparison | Manual inspection | Web research of comparable tools | Industry standards exist; don't guess |
| Stub detection | Grep for 'pass' or 'NotImplementedError' | `grep -r "def.*:.*pass" + ast.parse` | Need to distinguish intentional minimal implementations from actual stubs |

**Key insight:** Python ecosystem has mature static analysis tools. Manual approaches miss edge cases and take longer.

## Common Pitfalls

### Pitfall 1: Confirmation Bias in Simplification

**What goes wrong:** Audit finds complexity everywhere because auditor expects to find it. Recommendations become "remove everything not immediately understood."

**Why it happens:** User requested simplification pass; auditor feels pressure to find issues.

**How to avoid:**
- Require evidence from comparable tools for each recommendation
- Format: "lm-eval-harness does X, we do Y because Z"
- If Z is weak, recommend changing to X
- If comparable tools also have the complexity, it may be inherent to the domain

**Warning signs:** Recommendations lack specific tool comparisons; use words like "seems complex" without evidence.

### Pitfall 2: Conflating Unwired with Unnecessary

**What goes wrong:** Code exists but isn't connected end-to-end, audit recommends removal, but feature may be needed.

**Why it happens:** Phases 1-3 may have built infrastructure not yet wired up.

**How to avoid:**
- Flag as "unwired" not "dead code"
- Cross-reference with planning docs (was this feature planned?)
- Defer wire-up-or-remove decision to Phase 5

**Warning signs:** Removing code without checking if it was intentionally staged for future work.

### Pitfall 3: Line Count Theater

**What goes wrong:** Audit reports "X lines of code" metrics without context, implying fewer lines = better.

**Why it happens:** LOC is easy to measure and sounds objective.

**How to avoid:**
- LOC is meaningless without comparison to similar tools
- nanoGPT has ~750 lines (training), lm-eval-harness has ~15,000 (evaluation framework)
- Different tools, different appropriate sizes
- Focus on **unnecessary complexity** not **absolute size**

**Warning signs:** Audit leads with LOC metrics; recommends deletions to "reduce lines."

### Pitfall 4: Architecture Astronautics

**What goes wrong:** Audit identifies legitimate abstractions as "over-engineering" and recommends flattening everything.

**Why it happens:** Research tools trend minimal; auditor overcorrects.

**How to avoid:**
- SSOT introspection system (already identified as keeper in CONTEXT.md)
- Backend abstraction (necessary for multi-backend support)
- Distinguish: "abstraction with no consumers" (remove) vs "abstraction with 3+ implementations" (keep)

**Warning signs:** Recommending removal of protocol/interface classes that have multiple implementations.

### Pitfall 5: Docker Dogmatism

**What goes wrong:** Audit recommends either "all Docker" or "no Docker" without evidence.

**Why it happens:** Docker adds complexity; auditor wants consistency.

**How to avoid:**
- Check comparable tools: vLLM uses Docker for serving, not for benchmark runner
- Pattern: Docker for **inference backends that require isolation**, not for **tool execution**
- PyTorch can run locally; TensorRT benefits from Docker for dependency isolation

**Warning signs:** "All backends should use Docker" or "Docker adds unnecessary complexity" without tool-specific justification.

### Pitfall 6: Missing Archaeological Context

**What goes wrong:** Audit flags code as dead/redundant without understanding why it was added.

**Why it happens:** Phases 1-3 added features iteratively; later decisions may have made earlier work redundant.

**How to avoid:**
- Cross-reference git history and planning docs
- Pattern: "Added in Phase X for reason Y, now redundant because Phase Z changed approach"
- Note which phase introduced each finding

**Warning signs:** "This code appears unused" without checking if a later phase superseded it.

## Code Examples

### Using vulture for Dead Code Detection

```python
# Run vulture to find unused code
# Source: https://github.com/jendrikseipp/vulture
import subprocess

result = subprocess.run(
    ["vulture", "src/", "--min-confidence", "80"],
    capture_output=True,
    text=True
)

# Output format:
# src/module.py:42: unused function 'old_helper' (80% confidence)
# src/config.py:15: unused class 'DeprecatedConfig' (100% confidence)
```

### Using deadcode with Auto-Fix

```bash
# Find unused code with higher accuracy than vulture
# Source: https://github.com/albertas/deadcode
deadcode src/

# Automatically remove detected unused code
deadcode src/ --fix

# Dry-run to see what would be removed
deadcode src/ --fix --dry
```

### AST-Based Stub Detection

```python
# Detect stub functions (functions with only pass/raise NotImplementedError)
# Source: Python ast module documentation
import ast
from pathlib import Path

def find_stubs(file_path):
    """Find functions that are stubs (pass/NotImplementedError only)."""
    with open(file_path) as f:
        tree = ast.parse(f.read(), filename=file_path)

    stubs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if body is just pass or raise NotImplementedError
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    stubs.append((node.name, node.lineno, "pass"))
                elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                        stubs.append((node.name, node.lineno, "NotImplementedError"))

    return stubs

# Find all stubs in src/
for py_file in Path("src").rglob("*.py"):
    stubs = find_stubs(py_file)
    for name, line, type in stubs:
        print(f"{py_file}:{line}: stub function '{name}' ({type})")
```

### Semgrep for Unwired Config Fields

```yaml
# Custom semgrep rule to find Pydantic fields never accessed
# Source: https://semgrep.dev/docs/writing-rules/
rules:
  - id: unused-pydantic-field
    pattern: |
      class $CLASS(BaseModel):
        ...
        $FIELD: $TYPE
    message: "Field $FIELD defined but may be unused"
    languages: [python]
    severity: INFO
```

### Comparing CLI Command Surface

```bash
# Manual comparison - count CLI commands in similar tools

# lm-eval-harness
lm-eval --help | grep "^  [a-z]" | wc -l
# Result: 3 commands (run, ls, validate)

# Our tool
lem --help | grep "^  [a-z]" | wc -l
# Compare: how many commands do we have?

# Pattern: research tools have 2-5 main commands
# More than 7 suggests over-complexity
```

## State of the Art

| Aspect | Old Approach | Current Approach (2026) | When Changed | Impact |
|--------|--------------|-------------------------|--------------|--------|
| Config format | JSON, custom formats | **YAML primary, TOML for package config** | PEP-518 (2018), industry adoption 2024+ | YAML is human-readable, allows comments; TOML for pyproject.toml |
| Dead code detection | Manual grep, unused-imports | **AST-based tools (vulture, deadcode)** | 2023+ | Higher accuracy, confidence scoring, auto-fix |
| CLI structure | Monolithic argument parsers | **Subcommand-based (Click, Typer)** | 2020+ | Better organization, help text per command |
| Docker usage | All-or-nothing | **Targeted (for serving, not tool)** | 2024+ | Reduce unnecessary containerization |
| Static analysis | Pylint, flake8 | **Ruff (all-in-one)** | 2023+ | 10-100x faster, replaces multiple tools |
| Campaign/grid execution | Built into tools | **External scripting (Hydra, Weights & Biases Sweeps)** | 2023+ | Tools stay simple; orchestration separate |
| Result aggregation | Built-in to benchmarking tool | **Jupyter notebooks, separate analysis scripts** | Always | Keeps tool focused on data collection |

**Deprecated/outdated:**
- **Monolithic CLI parsers**: Use subcommands (Click/Typer) instead of flat argparse
- **JSON config for experiments**: YAML is now standard for readability
- **Built-in campaign systems**: Research tools output raw results; users handle sweeps
- **Pylint/flake8**: Ruff is faster and more comprehensive

## Open Questions

### 1. Campaign vs Experiment Command Distinction

**What we know:**
- Similar tools (lm-eval-harness, vLLM, LLMPerf) don't have campaign commands
- Users run single experiments, script sweeps externally
- Campaign functionality currently exists in our tool

**What's unclear:**
- Is campaign functionality solving a real need or adding complexity?
- Could grid generation be separated from execution?
- Should we keep campaign but simplify implementation?

**Recommendation:** Audit should document how campaign is currently implemented and compare against Hydra/W&B Sweeps (external orchestration) vs built-in. Let evidence drive whether to keep, simplify, or remove.

### 2. PyTorch Local vs Docker

**What we know:**
- vLLM/TensorRT clearly benefit from Docker (dependency isolation, GPU routing)
- PyTorch is in core dependencies (local execution works)
- Comparable tools (lm-eval-harness, nanoGPT) run PyTorch locally

**What's unclear:**
- Would Docker-only for all backends simplify execution paths?
- Does local PyTorch cause any actual problems?
- What's the user experience tradeoff?

**Recommendation:** Audit should count distinct execution paths (local vs Docker, backend detection) and compare complexity. If local PyTorch adds significant code, evaluate removal. If it's simple, keep it.

### 3. Detection System Overlap

**What we know:**
- CONTEXT.md identifies: docker_detection, backend_detection, env_setup, container strategy
- Multiple detection systems added across phases
- May have redundancy or conflicting logic

**What's unclear:**
- Do these systems actually overlap or handle different concerns?
- Can they be unified without breaking functionality?
- What's the minimum detection system needed?

**Recommendation:** Audit must trace all detection code paths and document what each system does. Create a flow diagram showing overlap. Recommend unification only with specific consolidation plan.

### 4. Test Quality vs Test Quantity

**What we know:**
- Tests exist across codebase
- Some may test removed features
- Some may have weak assertions ("test passes if no exception")

**What's unclear:**
- How many tests actually validate behavior vs just execute code?
- Are there tests for unwired features?
- Do tests cover the actual critical paths?

**Recommendation:** Audit should check for "always-passing tests" (no assertions, or only assert True) and tests referencing removed/unwired features. Recommend removal or strengthening.

## Sources

### Primary (HIGH confidence)

- [GitHub - EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - CLI structure, config format patterns
- [vLLM Benchmark CLI Documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/) - Benchmark command patterns
- [GitHub - karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal research tool file structure
- [GitHub - ray-project/llmperf](https://github.com/ray-project/llmperf) - Script-per-function CLI pattern
- [Python ast module documentation](https://docs.python.org/3/library/ast.html) - Code introspection patterns
- [GitHub - jendrikseipp/vulture](https://github.com/jendrikseipp/vulture) - Dead code detection tool
- [GitHub - albertas/deadcode](https://github.com/albertas/deadcode) - Unused code detection with auto-fix

### Secondary (MEDIUM confidence)

- [Streamlining lm-eval Architecture · Issue #3083](https://github.com/EleutherAI/lm-evaluation-harness/issues/3083) - Architecture complexity discussions
- [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/) - Multi-backend config patterns
- [GenAI-Perf Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf.html) - NVIDIA benchmarking patterns
- [The Code Audit Playbook: A Smarter Approach for 2026](https://softjourn.com/insights/code-audit-playbook) - Audit methodology
- [Top 10 Python Code Analysis Tools in 2026](https://www.jit.io/resources/appsec-tools/top-python-code-analysis-tools-to-improve-code-quality) - Tool landscape
- [JSON vs YAML vs TOML: Which Configuration Format Should You Use in 2026?](https://dev.to/jsontoall_tools/json-vs-yaml-vs-toml-which-configuration-format-should-you-use-in-2026-1hlb) - Config format standards
- [Docker 2026: 92% Adoption and the Container Tipping Point](https://www.programming-helper.com/tech/docker-2026-container-adoption-enterprise-kubernetes-python) - Docker usage patterns

### Tertiary (LOW confidence)

- [Web search results on parameter sweep configuration patterns](https://ieeexplore.ieee.org/document/843757/) - Academic grid computing research (may not reflect modern tool patterns)
- [Web search results on benchmark result aggregation](https://mlbenchmarks.org/12-problem-aggregation.html) - General benchmarking theory
- Various Medium/blog posts on Docker and Python patterns (unverified against official sources)

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** - Vulture, deadcode, ast, ruff are widely documented and used
- Architecture patterns: **MEDIUM** - Based on analysis of comparable tools' public repos and docs; direct repo inspection limited to README/structure, not full code audit
- Pitfalls: **MEDIUM** - Derived from audit methodology best practices and comparable tool analysis; not from direct experience auditing similar tools
- Industry tool comparisons: **MEDIUM** - Based on public GitHub repos, official docs, and CLI help text; didn't run all tools or perform deep code analysis

**Research date:** 2026-02-05
**Valid until:** 30 days (audit methodology is stable; tools update slowly)

**Research limitations:**
- Did not execute comparable tools to measure actual complexity metrics
- Did not perform full code analysis of comparable tools (only examined structure, CLI, config patterns)
- Industry "norms" derived from 4-7 tools; not exhaustive survey
- Docker patterns observed in documentation; actual usage by researchers may differ

**What was NOT researched:**
- Specific codebase inspection (that's the audit itself, not the research)
- Execution of automated tools on our codebase (audit task, not research task)
- Detailed comparison of every feature (audit deliverable, not research deliverable)
- User interviews or surveys on research tool preferences

**For the planner:**
This research provides methodology and tools for conducting the audit. The audit itself (Phase 4 plans) will use these tools and patterns to inspect the codebase and produce findings. Research answers "how to audit"; plans answer "what to audit and in what order."
