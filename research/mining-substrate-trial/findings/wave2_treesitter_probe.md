# Wave 2 a_treesitter probe (Tier A, WAVE2_PROTOCOL section 3)

Universal tree-sitter query walker scored against the validated-union
reference on two distinct tasks: schema discovery (Task 1) and invariant
mining (Task 2). 6 query-iteration rounds over ~45 min wall.

## Headline numbers

| Engine        | Task         | Recall | Precision | Wall (sec) |
|---------------|--------------|--------|-----------|------------|
| transformers  | schema       | 0.812  | 0.919     | 0.117      |
| transformers  | invariants   | 0.482  | 0.587     | 0.067      |
| vllm          | schema       | 0.985  | 1.000     | 0.015      |
| vllm          | invariants   | 0.564  | 0.190     | 0.131      |

Reference cardinalities: transformers schema=112 fields, invariants=56;
vllm schema=135, invariants=39. Per-engine wall is dominated by file I/O,
not parsing; each engine touches at most 3 source files.

## Query set (final)

Schema (Task 1):
- class_definition + body walker -> typed class-scope assignments
  (`x: int = 3`) and bare `_ClassVar = ...` constants.
- function-signature walker for __init__: handles
  `typed_default_parameter`, `typed_parameter`, `default_parameter`
  (untyped-with-default, e.g. `bnb_4bit_compute_dtype=None`), bare
  `identifier`.
- `self.X = kwargs.pop("X", default)` walker -- specifically targets the
  transformers GenerationConfig pattern; 67 sampling fields live here.
- Named-method signature walker -- pulls `from_pretrained` parameters
  into engine_params.

Invariants (Task 2):
- Pass 0: if-elif-elif-else-raise type-guard detection. Recovers
  `bnb_4bit_compute_dtype`-style `type_is_not` invariants where the raise
  lives in the else-branch of an isinstance chain.
- Pass 1: every `if_statement` -> if it has a direct raise OR a "soft
  sink" (`minor_issues[k] = v`, `warnings.warn`, `logger.warning`), emit
  one invariant per (primary, secondary) cross-field pair. Walks the
  enclosing if-chain to discover gating fields.
- Predicate-kind classifier: AST-shape-first (comparison_operator children
  give `>`/`<`/`==`/`!=`/`>=`/`<=`/`is`/`is not`), boolean_operator
  recursion -- if the left side is a presence check (`is None` /
  `is not None`), defer to the right side so the substantive value
  predicate wins (`X is not None and X < 0` -> `lt`, not `present`).
- Validator-decorator pass: catches `@field_validator("foo")` /
  `@model_validator` / `@validator` -- low yield for transformers + vllm
  but cheap.
- Allowlist gate: only emit from classes in the target list +
  namespace-override map (drops noise from transformers' 20+ peripheral
  quantization configs).
- Bare-identifier fallback: when an `if`-raise has no `self.X` refs,
  scan bare identifiers (filtered against a builtin/keyword denylist)
  -- catches `if load_in_4bit and load_in_8bit: raise` where the
  field names are method parameters, not attributes.

## Task 1 verdict: production-viable on its own

`vllm` already at 98.5% / 100%. transformers at 81.2% / 91.9%. The
transformers gap is structural: 21 of 112 engine_params fields are
sourced from Sphinx `:param:` docstring blocks
(`x-source: kwargs_docstring`) on `PreTrainedModel.from_pretrained` /
`AutoModelForCausalLM.from_pretrained` / `BitsAndBytesConfig.__doc__`.
Tree-sitter sees the docstring as a single `(string)` node; it does not
parse the Sphinx grammar. This 19-point gap is not closeable inside the
tree-sitter substrate -- it requires either a parallel docstring miner
(very Wave 1) or LLM extraction of the `:param X: <type>` lines.

**Verdict Task 1: YES.** vllm is production-ready as-is; transformers
needs a complementary Sphinx-docstring lifter for the remaining ~20%.
That lifter is a one-time per-engine investment, not a per-bump cost.

## Task 2 verdict: complement, not standalone

transformers 48% / 59% and vllm 56% / 19% are well below the WAVE2
production bar of 65% recall + 65% precision (section 9). The precision
collapse on vllm is the headline failure: scanning every `if-raise` in
`config.py` produces ~100 emissions, of which only 22 hit the reference
identity space. The reference catalogue is a curated subset of "validated
on construction"; many predicates in `_verify_*` methods are legitimate
invariants the reference simply does not enumerate (lifecycle invariants
that fire later in the engine setup pipeline). Tree-sitter cannot
distinguish "validator that runs at __init__" from "runtime check inside
a setup method" without semantic context.

### Systematic Task 2 gap analysis (transformers)

29 of 56 reference invariants missed. Categorisation:

**Tree-sitter could catch with more queries (~12 of 29):**
- `('transformers.sampling', 'num_beams', 'exact', '')` and the
  `num_beams` cross-field variants -- the outer `if self.num_beams == 1:`
  block dominates lines 600-612 with multiple inner predicates. My code
  treats it as gating-only (emits only the inner-field variants); needs
  also a standalone-outer-condition emission.
- `('transformers.sampling', 'epsilon_cutoff', 'present', '')` and 3
  siblings -- the inner cond is `self.epsilon_cutoff is not None and
  self.epsilon_cutoff != 0.0`. Currently classifies as `present` (correct)
  but only emits cross-field with `do_sample`; the single-field present
  variant is dropped because the cross-field pair wins.
- `('transformers', 'load_in_4bit', 'exact', 'load_in_8bit')` -- the
  predicate is `if load_in_4bit and load_in_8bit: raise`. Bare-identifier
  walker catches `load_in_4bit` but with kind `ge` (wrong; from
  text-fallback) -- needs a proper "bare boolean truthiness" classifier
  that returns `exact`.

**Semantic resolution required (~17 of 29):**
- `('transformers.sampling', 'return_dict_in_generate', 'exact',
  'output_attentions')` -- the inner cond is `if getattr(self,
  extra_output_flag) is True:` where `extra_output_flag` is a loop
  variable iterating over `self.extra_output_flags`. Tree-sitter sees
  the loop variable name, not the resolved attribute. To recover the
  field set, the walker would need to chase the variable to its
  iterator and resolve the iterator to a known tuple literal. That is
  semantic analysis.
- `('transformers', 'bnb_4bit_quant_storage', 'not_in', '')` -- nested
  inside an elif-isinstance arm: `elif isinstance(X, str): if X not in
  [...]: raise`. The not_in predicate is real, but its enclosing
  context is type-guard-then-value-check. The pass-0 type guard pulls
  the outer field; pass-1 finds the inner cond but its enclosing class
  is correctly bnb so it emits, except the field-resolution walks
  produce `X` (bare identifier) -- recall succeeds for the type_is_not
  variant but the secondary not_in needs distinct enclosing-context
  handling.
- `('transformers.sampling', 'logits_processor', 'present',
  'stopping_criteria')` -- the field name lives inside a string tuple
  iterated by `for arg in generate_arguments:` AND tested via
  `hasattr(self, arg)`. Resolving `arg` to specific names requires
  reading the literal tuple. Not within tree-sitter scope.
- `('transformers.sampling', 'cache_implementation', 'not_in', '')` --
  the test is `self.cache_implementation not in
  ALL_CACHE_IMPLEMENTATIONS`. The literal `not_in` predicate IS caught
  with kind `not_in`, but the cell emits `('transformers.sampling',
  'cache_implementation', 'not_in', '')` -- this one IS captured.

So roughly 12 of 29 are addressable with more queries (single-field
emission policy, bare-bool truthiness classifier, attribute-of-loop-var
walker); 17 of 29 require resolving a self.X / variable / literal across
ASTs to a known field name. That second bucket is the wall.

### Systematic Task 2 gap analysis (vllm)

17 of 39 missed. Headline misses:
- 5 in `vllm.engine.lora` / `vllm.engine.prompt_adapter` -- one of them
  recovered (max_loras lt); the rest need finer per-method classification.
- `('vllm.sampling', 'stop', 'present', 'detokenize')` and `('seed',
  'exact')` and others -- the SamplingParams validators have
  `self.stop is not None and (not isinstance(self.stop, list) or len(...))`
  shapes that mix presence + type + list-comprehension. Definitely
  semantic.

Precision collapses because `_verify_*` methods contain ~80 raise sites
all over the vllm config tree, only 22 of which match the validated
reference's identity space. Pruning would require either knowing
"this raise fires at __init__" vs "this raise fires later in setup", or
restricting to a hand-curated method allowlist (defeats the
universal-walker premise).

**Verdict Task 2: NO standalone.** As a complement to an LLM: YES,
high-value. Tree-sitter catches 48-56% of validated invariants in
< 200 ms wall per engine with reasonable precision on transformers
(59%). The remaining 50% comes in two buckets: (a) ~20% recoverable
with more iteration on the query set, (b) ~30% requires semantic
self.X resolution. A hybrid where tree-sitter mines the easy ones and a
small-LLM disambiguates `self.X = kwargs.pop(...)` chains + resolves
loop-variable predicates is the natural next step.

## Cost

Per cell: 15-130 ms wall, no GPU. Memory < 50 MB (the AST). Per-bump CI
cost (rerunning both tasks on all 3 engines): ~0.5 sec wall total +
~20 sec source-tree unpack. Negligible. The cost frontier is moot for
this substrate; the gating constraint is recall.

## What we'd test next

1. **Tree-sitter + small-LLM disambiguation pass.** Treesitter emits
   candidate predicates including unresolved self.X references; LLM
   resolves each unresolved chain in one shot per file. Hypothesis: the
   LLM dispatch happens on a query-extracted skeleton, not raw source,
   so cell cost stays at ~5 sec wall and ~3 cents per cell, while
   recall lifts toward Wave 1 H6 levels (~70%).
2. **Per-engine query overlay tables.** vllm precision can almost
   certainly be lifted to > 50% by explicitly listing which `_verify_*`
   methods are construction-time vs deferred. Same as Wave 1's pattern
   of per-engine config classes, just smaller (allowlist not walker).
3. **kwargs.pop call-graph cross-reference for schema.** transformers
   schema gap of 19% is entirely Sphinx-kwargs-docstring fields. A
   separate Sphinx-XML lifter (W2-b-doc was dropped) might close it
   without LLM cost, but it's per-engine substrate prep work.

## Iteration log

| Round | Schema (T/V) | Inv (T/V) | Change |
|-------|--------------|-----------|--------|
| 1     | 5/99         | 0/0       | Baseline queries; bare class field walker only. |
| 2     | 81/99        | 7/26      | Add kwargs.pop walker + from_pretrained signature + namespace overrides. |
| 3     | 81/99        | 27/44     | If-statement walker + soft-signal (minor_issues) + nested-if chain pairing + elif/else type guard. |
| 4     | 81/99        | 41/44     | Comparison-operator kind classifier with `is`/`is not`/`not in` text fallback. |
| 5     | 81/99        | 46/56     | Allowlist gate + drop `unknown` kind + finer vllm sub-namespaces (lora / prompt_adapter / tokenizer). |
| 6     | 81/99        | 48/56     | Bare-identifier parameter-name walker for boolean-truthiness predicates. Diminishing returns; stop. |
