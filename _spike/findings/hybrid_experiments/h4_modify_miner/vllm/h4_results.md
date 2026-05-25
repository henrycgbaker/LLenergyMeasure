# H4 results: vllm v0.7.3

**Pattern:** H4 LLM-modifies-miner. Llama3.1:70b q4 (container Ollama @ 11435) reads
the static walker, gap inventory, and engine-source excerpts, then emits
structured `diagnoses` + `patches` (anchor + replacement, not unified diffs).

Run wall-clock: 78.4s (LLM call only, ~50s).

## Counts

| Metric | Value |
|---|---|
| Canonical reference count (mined earlier) | 26 |
| Baseline count (unpatched walker, re-run today) | 66 |
| Patched count | 0 (subprocess crashed) |
| Patched run OK | False |
| Diagnoses produced | 3 |
| Patches proposed | 3 |
| Patches applied (anchor matched + edit succeeded) | 1 |

**Note on baseline-vs-canonical drift (66 vs 26):** the canonical YAML was mined
at an earlier point with a fully-installed vLLM. The trial worktree can't
install vLLM 0.7.3 (transitive deps msgspec/zmq/blake3/openai not present),
so the live re-run uses a file-based-resolver monkey-patch of
`_check_landmarks` and the walker exposes ~40 more invariants than the
historical canonical. Likely cause: subsequent walker improvements emit
more candidates than at canonical-mint time. The 66 is the realistic
"today's unpatched (a) ceiling" floor for H4 to lift against.

## Diagnoses (all three CORRECT and STRUCTURALLY INFORMATIVE)

### G-vllm-1: EngineArgs.__post_init__ normalisation patterns

> The walker only emits invariants from `if X: raise` patterns, but
> EngineArgs uses `if self.X is None: self.X = default` for validation.

LLM's structural reading matches `post_trial_a_gap_closure.md` exactly:
EngineArgs at v0.7.3 has zero raises; all validation is normalisation.

Proposed fix: extend walker to emit normalisation patterns as
`severity=dormant`. Plausible (and the walker already has a
`_detect_self_assign` detector that emits `severity=dormant`); the gap is
that EngineArgs isn't on the walker's `_CLASS_TARGETS` list.

### G-vllm-2: ModelConfig predicate aliases (local-var compare)

> The walker's `_self_attr` predicate extractor only matches `self.X`,
> not a local var that aliases `self.X`.

Correct. Concrete example: `_verify_tokenizer_mode` does
`tokenizer_mode = self.tokenizer_mode.lower(); if tokenizer_mode not in
[...]: raise`. The walker's `_extract_compare` only ties `self.X`
predicates; the local var `tokenizer_mode` is opaque.

Proposed fix: light alias tracking (`<local> = self.<field>.<method>()`
or `<local> = self.<field>`). Plausible scope ~50-100 LoC.

### G-vllm-3: CacheConfig._verify_cache_dtype elif/else chains

> The walker handles `if X: raise` patterns, but not `else:` branches in
> elif/else chains that contain a raise.

Correct. The walker's `_handle_if` descends into nested `ast.If` orelse
nodes (lines 729-733 of the original walker) but does NOT emit on
non-`If` statements in the orelse (e.g. a bare `raise` in `else:`).

Proposed fix: descend into the final `else:` with negated conditions
accumulated. Conceptually sound; the LLM's proposed code references
`_handle_else` (function that doesn't exist) instead of inlining the
descent - sketch, not working code.

## Patches

### P-vllm-1: extend `_handle_if` to detect ast.Assign as well

- **Status:** not applied (anchor not found in walker source).
- **LLM's anchor:** `if isinstance(stmt, ast.If):\n    _handle_if(stmt, frame)`
- **Reality:** the walker's descent uses `if isinstance(stmt, ast.If):\n
    _handle_if(stmt, frame)\n elif isinstance(stmt, ast.For):` etc.;
  anchor was close but not exact.
- **LLM's `new_code`:** references `_handle_assign_or_if` (function that
  doesn't exist). Sketch, not working code.
- **Verdict:** anchor brittleness + helper hallucination. Diagnosis was
  useful; patch can't be applied.

### P-vllm-2: extend `_self_attr` to also handle alias assignments

- **Status:** not applied (anchor not found in walker source).
- **LLM's anchor:** mentions `if isinstance(node, ast.Attribute) and ...`
  - this is on line 432-437 of the walker but the LLM's anchor includes
  leading whitespace that doesn't match.
- **LLM's `new_code`:** sketches the alias-tracking shape (`isinstance(node,
  ast.Assign)`) but the new_code body says `# alias tracking logic here`
  - placeholder, not working code.
- **Verdict:** anchor mismatch + placeholder body. Diagnosis useful;
  patch is a stub.

### P-vllm-3: handle else-branches in if/elif/else chains

- **Status:** APPLIED at line 728. Resulted in walker crash on subprocess.
- **LLM's new_code:**
  ```python
  for sub in if_node.orelse:
      if isinstance(sub, (ast.If, ast.Raise)):
          # descend into else branch with negated conditions
          _handle_else(sub, frame, negated_conditions)
  ```
- **Why crashed:** `_handle_else` is undefined; `negated_conditions` is
  undefined.
- **Verdict:** the LLM understands the elif/else gap correctly but emits
  pseudocode referencing helpers it didn't define. Patch needs human
  inlining of the descent + negation logic. Conceptually mergeable.

## What worked

- The 3 diagnoses match `post_trial_a_gap_closure.md`'s G-vllm-{1,2,3}
  text essentially line-by-line. The LLM correctly characterised:
  - EngineArgs as a normalisation-only (no-raise) class.
  - ModelConfig as a local-variable-alias compare site.
  - CacheConfig as an if/elif/else chain.
- The LLM honoured the structured-JSON output contract (no markdown
  fences; valid JSON; proper schema match) on first try.
- The harness applied the one matching-anchor patch deterministically.

## What didn't work

- **Patches reference undefined helper functions** (`_handle_else`,
  `_handle_assign_or_if`). The LLM at 70B-q4 sketches the function
  signature without inlining or defining the helper body.
- **Anchor brittleness**: two of three patches' anchor_text strings
  didn't match the walker source verbatim. The LLM reconstructs the
  anchor from memory rather than copy-pasting; one-character whitespace
  drift breaks the match.
- **Walker semantics not preserved**: the one APPLIED patch (P-vllm-3)
  introduced a reference to undefined `_handle_else` and crashed the
  walker entirely (0 candidates emitted), masking the patch's intent.

## Spike-refactor value

Despite the patches not running, the DIAGNOSES are directly mergeable
into the spike branch's vllm mining refactor backlog as text:

1. **G-vllm-3 (else-branch descent)** is a small, locally-scoped walker
   patch. The LLM identified the exact line range. A human can implement
   the negated-condition descent in ~30 LoC. This is a clean post-trial
   spike issue.
2. **G-vllm-2 (local-var alias tracking)** is medium-scoped (~50-100
   LoC). The LLM's description maps cleanly to a `_TrackedAliases` frame
   class. Mergeable as a future spike issue.
3. **G-vllm-1 (normalisation-as-dormant)** is more architectural - it
   requires adding EngineArgs to `_CLASS_TARGETS` AND ensuring the
   `_detect_self_assign` detector fires correctly there. Mergeable as a
   spike feature issue.

## Negative findings

- 70B-q4 isn't a competent walker-maintenance engineer at this prompt
  scope. It diagnoses well but writes pseudocode for the fix.
- Larger model (or chain-of-thought reasoning prompt) might close the
  patch-quality gap. Future H4 variants: try Anthropic/Claude as backend.

## Artefacts

- `vllm/raw_llm_outputs/prompt.txt` - full prompt sent to Ollama.
- `vllm/raw_llm_outputs/raw_response.txt` - raw LLM response.
- `vllm/raw_llm_outputs/diagnoses.json` - parsed diagnoses.
- `vllm/raw_llm_outputs/patches.json` - parsed patches.
- `vllm/proposed_patches/P-vllm-{1,2,3}.json` - per-patch records.
- `vllm/patched_producer/static_invariant_miner.py` - patched walker
  (with the one-applied edit + crash).
- `vllm/patched_outputs/subprocess_{stdout,stderr}.txt` - run logs.
- `vllm/baseline_unpatched_run/invariants.proposed.yaml` - baseline (66
  candidates) for the patched-vs-baseline diff.
- `vllm/h4_summary.json` - structured summary.
