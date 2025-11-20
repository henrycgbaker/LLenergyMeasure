# v3.0 Implementation Plan: Chain of Reasoning Support

## Overview

v3.0 adds comprehensive support for measuring efficiency of Chain of Reasoning (CoT) models and workflows, including multi-step reasoning, conditional computation, and iterative refinement.

## Architecture Strategy

### Core Insight
CoT introduces **non-uniform computation** - different problems require different reasoning depths. We need to track:
- Per-step metrics (not just per-token)
- Reasoning phases (thinking vs answering)
- Conditional branching patterns
- Quality-efficiency trade-offs

### Design Principles
1. **Backwards Compatible**: v2.0 experiments still work
2. **Opt-In CoT**: Enable via `reasoning` config parameter
3. **Model Agnostic**: Support native CoT and prompted CoT
4. **Granular Tracking**: Step-level metrics, not just aggregate
5. **Efficient Storage**: Reasoning traces can be large

---

## Phase 1: Configuration & Data Model (v3.1.0)

### New Configuration

**`config/reasoning.py`:**
```python
@dataclass
class ReasoningConfig:
    """Configuration for chain of reasoning experiments."""

    # Reasoning type
    enabled: bool = False
    reasoning_type: str = "chain_of_thought"  # cot, self_consistency, tree_of_thought

    # Step control
    max_reasoning_steps: int = 10
    min_reasoning_steps: int = 1
    step_delimiter: str = "\n\n"  # How to detect step boundaries

    # Prompting
    use_system_prompt: bool = True
    cot_trigger: str = "Let's think step by step:"
    thought_prefix: str = "<think>"
    thought_suffix: str = "</think>"
    answer_prefix: str = "<answer>"
    answer_suffix: str = "</answer>"

    # Self-consistency
    num_samples: int = 1  # For self-consistency sampling
    temperature: float = 0.7

    # Stopping
    stop_on_answer: bool = True
    max_think_tokens: int = 2000
    max_answer_tokens: int = 500
```

**Update `ExperimentConfig`:**
```python
@dataclass
class ExperimentConfig:
    # ... existing fields ...

    # NEW: Chain of reasoning support
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
```

### Extended Metrics

**`metrics/reasoning.py`:**
```python
@dataclass
class ReasoningStepMetrics:
    """Metrics for a single reasoning step."""
    step_number: int
    tokens: int
    duration_seconds: float
    energy_kwh: float
    flops: int
    content: Optional[str] = None  # The actual reasoning text

@dataclass
class ReasoningMetrics:
    """Aggregate reasoning metrics."""

    # Step tracking
    total_steps: int
    average_steps: float  # Across batch
    min_steps: int
    max_steps: int

    # Phase breakdown
    think_phase_tokens: int
    answer_phase_tokens: int
    think_phase_duration: float
    answer_phase_duration: float
    think_phase_energy: float
    answer_phase_energy: float

    # Efficiency metrics
    tokens_per_step: float
    energy_per_step: float
    flops_per_step: float

    # Quality indicators (if available)
    self_corrections: int = 0
    backtracking_events: int = 0
    branching_factor: float = 1.0

    # Per-step details
    steps: List[ReasoningStepMetrics] = field(default_factory=list)
```

**Update `EfficiencyMetrics`:**
```python
@dataclass
class EfficiencyMetrics:
    # ... existing fields ...

    # NEW: Optional reasoning metrics
    reasoning: Optional[ReasoningMetrics] = None
```

---

## Phase 2: Reasoning Inference Engine (v3.2.0)

### New Inference Module

**`core/reasoning.py`:**
```python
class ReasoningInferenceEngine:
    """
    Inference engine with chain of reasoning support.

    Tracks multi-step reasoning, phase detection, and per-step metrics.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: ReasoningConfig,
        energy_tracker,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.energy_tracker = energy_tracker

    def run_reasoning_batch(
        self,
        prompts: List[str],
    ) -> Tuple[List[str], ReasoningMetrics]:
        """
        Run inference with reasoning tracking.

        Returns:
            outputs: Generated text for each prompt
            metrics: Reasoning metrics
        """
        all_steps = []

        for prompt in prompts:
            # Inject CoT trigger if needed
            if self.config.cot_trigger:
                prompt = f"{prompt}\n{self.config.cot_trigger}"

            # Track per-step generation
            steps = self._generate_with_step_tracking(prompt)
            all_steps.append(steps)

        # Aggregate metrics
        metrics = self._compute_reasoning_metrics(all_steps)

        return [self._extract_final_answer(steps) for steps in all_steps], metrics

    def _generate_with_step_tracking(
        self,
        prompt: str,
    ) -> List[ReasoningStepMetrics]:
        """Generate with per-step tracking."""
        steps = []
        current_text = prompt

        for step_num in range(self.config.max_reasoning_steps):
            # Generate one reasoning step
            step_start = time.time()
            energy_start = self.energy_tracker.get_current_energy()

            # Generate until step delimiter
            output = self._generate_step(current_text)

            # Measure step metrics
            step_duration = time.time() - step_start
            step_energy = self.energy_tracker.get_current_energy() - energy_start
            step_tokens = len(self.tokenizer.encode(output))

            step_metrics = ReasoningStepMetrics(
                step_number=step_num,
                tokens=step_tokens,
                duration_seconds=step_duration,
                energy_kwh=step_energy,
                flops=self._estimate_step_flops(step_tokens),
                content=output,
            )
            steps.append(step_metrics)

            # Check stopping criteria
            if self._should_stop_reasoning(output, steps):
                break

            current_text += output

        return steps

    def _detect_reasoning_phases(
        self,
        steps: List[ReasoningStepMetrics],
    ) -> Tuple[List[int], List[int]]:
        """
        Detect which steps are thinking vs answering.

        Returns:
            think_step_indices, answer_step_indices
        """
        # Simple heuristic: last step is usually answer
        if self.config.answer_prefix:
            # Parse based on markers
            answer_steps = [
                i for i, s in enumerate(steps)
                if self.config.answer_prefix in s.content
            ]
            think_steps = [i for i in range(len(steps)) if i not in answer_steps]
        else:
            # Default: all but last are thinking
            think_steps = list(range(len(steps) - 1))
            answer_steps = [len(steps) - 1] if steps else []

        return think_steps, answer_steps
```

**Update `core/experiment.py`:**
```python
def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run experiment with optional CoT support."""

    # ... existing model loading ...

    # Choose inference engine
    if config.reasoning.enabled:
        engine = ReasoningInferenceEngine(
            model=model,
            tokenizer=tokenizer,
            config=config.reasoning,
            energy_tracker=energy_tracker,
        )
        outputs, reasoning_metrics = engine.run_reasoning_batch(prompts)
        metrics.reasoning = reasoning_metrics
    else:
        # Standard inference (existing path)
        engine = InferenceEngine(...)
        outputs, metrics = engine.run_batch(prompts)

    # ... rest of experiment ...
```

---

## Phase 3: Prompting Strategies (v3.3.0)

### Prompt Templates

**`data/prompts/reasoning.py`:**
```python
class CoTPromptStrategy:
    """Prompting strategies for chain of reasoning."""

    @staticmethod
    def zero_shot_cot(question: str) -> str:
        """Zero-shot CoT prompting."""
        return f"{question}\n\nLet's think step by step:"

    @staticmethod
    def few_shot_cot(question: str, examples: List[Tuple[str, str]]) -> str:
        """Few-shot CoT with examples."""
        prompt = "Here are some examples:\n\n"
        for q, reasoning in examples:
            prompt += f"Q: {q}\nA: {reasoning}\n\n"
        prompt += f"Q: {question}\nA: Let's think step by step:"
        return prompt

    @staticmethod
    def structured_cot(question: str) -> str:
        """Structured CoT with XML tags."""
        return f"""Question: {question}

Please reason step-by-step using the following structure:

<think>
Step 1: [Your reasoning]
Step 2: [Your reasoning]
...
</think>

<answer>
[Final answer]
</answer>"""

    @staticmethod
    def self_consistency(question: str, temperature: float = 0.7) -> str:
        """Self-consistency CoT (sample multiple times)."""
        return f"{question}\n\nLet's approach this step by step:"
```

**Dataset Integration:**
```python
# Add reasoning-specific datasets
REASONING_DATASETS = {
    "gsm8k": "grade school math",
    "math": "mathematical reasoning",
    "strategyqa": "implicit reasoning",
    "commonsenseqa": "commonsense reasoning",
}
```

---

## Phase 4: Analysis Tools (v3.4.0)

### Reasoning Analysis

**`analysis/reasoning.py`:**
```python
def analyze_reasoning_efficiency(
    experiments: List[ExperimentResult],
) -> Dict[str, Any]:
    """
    Analyze reasoning efficiency across experiments.

    Returns:
        Analysis including:
        - Optimal reasoning depth
        - Efficiency vs depth curves
        - Cost-quality trade-offs
    """
    analysis = {
        "depth_distribution": {},
        "efficiency_by_depth": {},
        "phase_breakdown": {},
    }

    for exp in experiments:
        if not exp.metrics.reasoning:
            continue

        rm = exp.metrics.reasoning

        # Depth distribution
        analysis["depth_distribution"][exp.experiment_id] = {
            "mean": rm.average_steps,
            "min": rm.min_steps,
            "max": rm.max_steps,
            "std": calculate_std([s.tokens for s in rm.steps]),
        }

        # Efficiency by depth
        for step in rm.steps:
            depth = step.step_number
            if depth not in analysis["efficiency_by_depth"]:
                analysis["efficiency_by_depth"][depth] = []

            analysis["efficiency_by_depth"][depth].append({
                "energy": step.energy_kwh,
                "tokens": step.tokens,
                "duration": step.duration_seconds,
            })

        # Phase breakdown
        analysis["phase_breakdown"][exp.experiment_id] = {
            "think_pct": rm.think_phase_energy / exp.metrics.total_energy_kwh,
            "answer_pct": rm.answer_phase_energy / exp.metrics.total_energy_kwh,
        }

    return analysis

def compare_reasoning_strategies(
    experiments: List[ExperimentResult],
) -> str:
    """Compare different reasoning strategies."""
    # Group by reasoning type
    by_strategy = {}
    for exp in experiments:
        strategy = exp.config.reasoning.reasoning_type
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(exp)

    # Generate comparison report
    report = "REASONING STRATEGY COMPARISON\n"
    report += "=" * 80 + "\n\n"

    for strategy, exps in by_strategy.items():
        avg_steps = statistics.mean([
            e.metrics.reasoning.average_steps for e in exps
        ])
        avg_energy = statistics.mean([
            e.metrics.total_energy_kwh for e in exps
        ])

        report += f"{strategy}:\n"
        report += f"  Avg steps: {avg_steps:.1f}\n"
        report += f"  Avg energy: {avg_energy:.6f} kWh\n"
        report += f"  Efficiency: {avg_energy/avg_steps:.8f} kWh/step\n\n"

    return report
```

---

## Phase 5: Examples & Documentation (v3.5.0)

### Example Scripts

**`examples/11_cot_basic.py`:**
```python
"""
Chain of Reasoning - Basic Example
===================================

Measure efficiency of CoT vs direct answering.
"""

from llm_efficiency.config import ExperimentConfig, ReasoningConfig

# Baseline: Direct answering
baseline_config = ExperimentConfig(
    model_name="gpt2",
    reasoning=ReasoningConfig(enabled=False),
    dataset_name="gsm8k",
    num_batches=20,
)

# CoT: Step-by-step reasoning
cot_config = ExperimentConfig(
    model_name="gpt2",
    reasoning=ReasoningConfig(
        enabled=True,
        reasoning_type="chain_of_thought",
        max_reasoning_steps=5,
        cot_trigger="Let's solve this step by step:",
    ),
    dataset_name="gsm8k",
    num_batches=20,
)

# Run comparison
baseline_result = run_experiment(baseline_config)
cot_result = run_experiment(cot_config)

# Compare
print(f"Baseline: {baseline_result.metrics.total_energy_kwh:.6f} kWh")
print(f"CoT: {cot_result.metrics.total_energy_kwh:.6f} kWh")
print(f"Overhead: {(cot_result.metrics.total_energy_kwh / baseline_result.metrics.total_energy_kwh - 1) * 100:.1f}%")
```

**`examples/12_cot_depth_analysis.py`:**
```python
"""
Reasoning Depth Analysis
========================

Find optimal reasoning depth for efficiency.
"""

from llm_efficiency.analysis import analyze_reasoning_efficiency

# Test different max depths
depths = [1, 3, 5, 10, 20]
results = []

for depth in depths:
    config = ExperimentConfig(
        model_name="gpt2",
        reasoning=ReasoningConfig(
            enabled=True,
            max_reasoning_steps=depth,
        ),
        num_batches=10,
    )
    result = run_experiment(config)
    results.append(result)

# Analyze
analysis = analyze_reasoning_efficiency(results)

# Plot efficiency vs depth curve
for depth, metrics in analysis["efficiency_by_depth"].items():
    avg_energy = statistics.mean([m["energy"] for m in metrics])
    print(f"Depth {depth}: {avg_energy:.8f} kWh/step")
```

---

## Implementation Timeline

### v3.1.0: Configuration & Data Model (Week 1)
- [ ] Add `ReasoningConfig` dataclass
- [ ] Add `ReasoningMetrics` dataclass
- [ ] Update `ExperimentConfig` and `EfficiencyMetrics`
- [ ] Update storage format
- [ ] Tests for new data structures

### v3.2.0: Reasoning Inference Engine (Week 2)
- [ ] Implement `ReasoningInferenceEngine`
- [ ] Add step tracking and phase detection
- [ ] Integrate with existing experiment flow
- [ ] Tests for reasoning engine

### v3.3.0: Prompting Strategies (Week 3)
- [ ] Add CoT prompt templates
- [ ] Add reasoning datasets
- [ ] Implement strategy selector
- [ ] Tests for prompting

### v3.4.0: Analysis Tools (Week 4)
- [ ] Add reasoning analysis functions
- [ ] Implement strategy comparison
- [ ] Add depth optimization
- [ ] Tests for analysis

### v3.5.0: Examples & Documentation (Week 5)
- [ ] Create 5+ CoT examples
- [ ] Update USAGE_GUIDE
- [ ] Add CoT best practices
- [ ] Update CHANGELOG

### v3.0.0: Final Release (Week 6)
- [ ] Integration testing
- [ ] Documentation review
- [ ] API finalization
- [ ] Release notes

---

## Key Design Decisions

### 1. Step Detection
**Challenge**: How to identify reasoning step boundaries?

**Solution**: Configurable with multiple strategies:
- Delimiter-based (e.g., `\n\n`)
- Token-based (generate N tokens per step)
- Marker-based (e.g., `<step>...</step>`)
- Model-specific (native CoT models have built-in steps)

### 2. Phase Detection
**Challenge**: Distinguish thinking vs answering?

**Solution**: Hybrid approach:
- Parse markers if available (`<think>`, `<answer>`)
- Heuristic: last step is usually answer
- Allow manual annotation in config

### 3. Metrics Granularity
**Challenge**: Track every token vs every step?

**Solution**: Per-step tracking with optional per-token:
- Always track: step-level metrics (manageable size)
- Optional: token-level details (for deep analysis)
- Aggregate: experiment-level summary

### 4. Backwards Compatibility
**Challenge**: Don't break v2.0 experiments

**Solution**:
- CoT is opt-in via `reasoning.enabled=True`
- Default behavior unchanged
- `reasoning: Optional[ReasoningMetrics]` in results

### 5. Model Support
**Challenge**: Different models handle CoT differently

**Solution**: Three tiers:
- **Tier 1**: Native CoT models (o1-style) - best support
- **Tier 2**: Prompted CoT (GPT-4, Claude) - good support
- **Tier 3**: Standard models + prompting - basic support

---

## Success Metrics

v3.0 is successful if we can:

1. **Measure CoT overhead accurately**
   - Energy cost of thinking vs answering
   - Per-step efficiency profiling

2. **Compare reasoning strategies**
   - Direct vs CoT vs self-consistency
   - Optimal depth determination

3. **Identify efficiency patterns**
   - When does more reasoning help?
   - Diminishing returns curves

4. **Support multiple models**
   - Works with native CoT and prompted CoT
   - Model-agnostic interface

5. **Maintain performance**
   - Minimal overhead for tracking
   - Efficient storage of reasoning traces

---

## Next Steps

1. **Review this plan** - Does this align with your vision?
2. **Prioritize phases** - Which should we build first?
3. **Prototype** - Build v3.1.0 configuration?
4. **Testing strategy** - How to validate CoT tracking?

Ready to proceed with implementation!
