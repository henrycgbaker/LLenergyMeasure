"""
bakeoff_B_script.py - LLM-as-knowledge-extractor POC

Tests whether llama3.1:70b via Ollama can replace ~3800 LoC of handwritten
transformers introspection machinery at acceptable quality.

Two Ollama calls:
  1. Schema call  -> bakeoff_B_schema.json
  2. Invariants call -> bakeoff_B_invariants.yaml

Then diffs against ground truth and writes bakeoff_B_local_llm.md.
"""

import inspect
import json
import re
import time
from pathlib import Path

import requests
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FINDINGS = Path(__file__).parent
GROUND_TRUTH_SCHEMA = (
    Path(__file__).parent.parent.parent
    / "engine_versions/transformers/v4_57_3/outputs/schema.discovered.json"
)
GROUND_TRUTH_INV = (
    Path(__file__).parent.parent.parent
    / "engine_versions/transformers/v4_57_3/outputs/invariants.proposed.yaml"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:70b"


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------
def get_sources():
    print(
        "Fetching source for GenerationConfig, from_pretrained, BitsAndBytesConfig, nested configs..."
    )
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
    from transformers.generation.configuration_utils import (
        CompileConfig,
        SynthIDTextWatermarkingConfig,
        WatermarkingConfig,
    )

    sources = {
        "GenerationConfig.__init__": inspect.getsource(GenerationConfig.__init__),
        "GenerationConfig.validate": inspect.getsource(GenerationConfig.validate),
        # We include a summary of GenerationConfig's full field list from __init__ docstring
        "GenerationConfig.__doc__": (GenerationConfig.__doc__ or "")[:3000],
        "AutoModelForCausalLM.from_pretrained": inspect.getsource(
            AutoModelForCausalLM.from_pretrained
        ),
        "BitsAndBytesConfig": inspect.getsource(BitsAndBytesConfig),
        "CompileConfig": inspect.getsource(CompileConfig),
        "WatermarkingConfig": inspect.getsource(WatermarkingConfig),
        "SynthIDTextWatermarkingConfig": inspect.getsource(SynthIDTextWatermarkingConfig),
    }
    # Measure total context
    total = sum(len(v) for v in sources.values())
    print(f"Total source context: {total:,} chars")
    return sources


# ---------------------------------------------------------------------------
# Ollama call (streaming=False)
# ---------------------------------------------------------------------------
def ollama_call(
    prompt: str, use_json_format: bool = False, timeout: int = 1800
) -> tuple[str, float]:
    """Returns (response_text, wall_seconds). Uses streaming to avoid timeout on large responses."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,  # streaming avoids read timeout on large outputs
    }
    if use_json_format:
        payload["format"] = "json"

    t0 = time.time()
    print(
        f"  Sending request to Ollama ({MODEL}), format={'json' if use_json_format else 'text'} (streaming)..."
    )

    chunks = []
    last_print = t0
    with requests.post(OLLAMA_URL, json=payload, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(chunk_size=None):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = data.get("response", "")
            chunks.append(token)
            # Progress print every 30s
            now = time.time()
            if now - last_print >= 30:
                elapsed_so_far = now - t0
                print(f"  ... {elapsed_so_far:.0f}s elapsed, {len(''.join(chunks))} chars so far")
                last_print = now
            if data.get("done"):
                break

    elapsed = time.time() - t0
    text = "".join(chunks)
    print(f"  Got response in {elapsed:.1f}s, {len(text)} chars")
    return text, elapsed


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
SCHEMA_PROMPT_TEMPLATE = """You are a code analyser. Your task is to extract the parameter schema for the HuggingFace Transformers library (v4.57.3) engine.

You will be given Python source code for:
1. AutoModelForCausalLM.from_pretrained (engine params)
2. BitsAndBytesConfig (quantisation config fields that also surface as engine params)
3. GenerationConfig.__init__ (sampling params - all kwargs)
4. GenerationConfig.__doc__ (docstring describing fields)
5. CompileConfig, WatermarkingConfig, SynthIDTextWatermarkingConfig (nested config classes)

OUTPUT: Return a JSON object matching EXACTLY this schema:

{{
  "engine": "transformers",
  "engine_version": "4.57.3",
  "engine_params": {{
    "<param_name>": {{
      "type": "<json_schema_type>",   // one of: "string","integer","number","boolean","array","object","null", or omit if unknown
      "default": <value_or_null>,     // omit if no default
      "description": "<brief>",       // optional
      "enum": [<values>]              // only if constrained to a set of values
    }}
  }},
  "sampling_params": {{
    "<param_name>": {{
      "type": "<json_schema_type>",
      "default": <value_or_null>,
      "description": "<brief>",
      "enum": [<values>]
    }}
  }},
  "$defs": {{
    "CompileConfig": {{
      "type": "object",
      "properties": {{
        "<field>": {{"type": "...", "default": ...}}
      }}
    }}
  }}
}}

Rules:
- engine_params come from AutoModelForCausalLM.from_pretrained signature + BitsAndBytesConfig fields
- sampling_params come from GenerationConfig.__init__ signature (all kwargs except self, **kwargs)
- $defs should contain CompileConfig properties
- For fields with no type annotation, omit "type" key but include "default" if known
- For None defaults, set "default": null
- Return ONLY valid JSON - no commentary, no markdown code blocks

=== SOURCE: AutoModelForCausalLM.from_pretrained ===
{from_pretrained_src}

=== SOURCE: BitsAndBytesConfig ===
{bnb_src}

=== SOURCE: GenerationConfig.__init__ ===
{init_src}

=== SOURCE: GenerationConfig docstring (truncated) ===
{gc_doc}

=== SOURCE: CompileConfig ===
{compile_config_src}

=== SOURCE: WatermarkingConfig ===
{watermarking_src}
"""

INVARIANTS_PROMPT_TEMPLATE = """You are a code analyser. Your task is to extract validation invariants from HuggingFace Transformers GenerationConfig.

You will be given:
1. GenerationConfig.validate() source - the primary validation method
2. GenerationConfig.__init__ source - also validates some fields at construction time
3. A description of what invariant types to look for

OUTPUT: Return a YAML document with this exact structure:

```yaml
schema_version: 1.0.0
engine: transformers
engine_version: 4.57.3
mined_at: '2026-05-24T00:00:00+00:00'
invariants:
- id: <snake_case_unique_id>
  engine: transformers
  library: transformers
  invariant_under_test: <what GenerationConfig method flags and why>
  severity: <error|dormant|warning>
  native_type: transformers.GenerationConfig
  miner_source:
    path: transformers/generation/configuration_utils.py
    method: <validate|__init__>
    line_at_scan: <line_number>
  match:
    engine: transformers
    fields:
      transformers.sampling.<field_name>: <value_or_predicate>
  kwargs_positive:
    <field>: <value_that_triggers_invariant>
  kwargs_negative:
    <field>: <value_that_does_NOT_trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<exact error/warning message text from source>'
  references:
  - <citation>
  added_by: llm_miner
  added_at: '2026-05-24'
```

INVARIANT TYPES TO EXTRACT:

1. ERROR invariants (raise ValueError at construction/__init__ time):
   - Field value not in allowed set (enum violation)
   - Field type mismatch (wrong Python type)
   - Field value out of range (e.g. max_new_tokens <= 0)

2. DORMANT invariants (logger.warning_once at validate() time - parameter will be silently ignored):
   - Sampling-only params (temperature, top_p, top_k, min_p, epsilon_cutoff, eta_cutoff, typical_p) set when do_sample=False
   - Beam-only params (early_stopping, length_penalty) set when num_beams=1
   - Output params (output_attentions, output_hidden_states, output_scores) set when return_dict_in_generate=False
   - num_beams=1 AND early_stopping=True
   - use_cache=False when cache_implementation is set
   - pad_token_id < 0

3. CROSS-FIELD invariants (combinations that error):
   - num_beams not divisible by num_return_sequences
   - num_return_sequences > num_beams when do_sample=False
   - num_beams=1 when diversity_penalty > 0 or num_beam_groups > 1

SEVERITY:
- "error" = raises ValueError (hard failure)
- "dormant" = logs warning, parameter silently normalised/ignored
- "warning" = logs warning, execution continues with user-set value

For each `match.fields` predicate, use:
- Exact value: `transformers.sampling.field: value`
- Presence: `transformers.sampling.field: {{present: true}}`
- Not in list: `transformers.sampling.field: {{present: true, not_in: [val1, val2]}}`
- Not equal: `transformers.sampling.field: {{present: true, not_equal: value}}`
- Greater than: `transformers.sampling.field: {{'>'  : value}}`
- Type not in: `transformers.sampling.field: {{present: true, type_is_not: [TypeName]}}`

Return ONLY the YAML - no commentary, no code blocks, no preamble.

=== SOURCE: GenerationConfig.validate() ===
{validate_src}

=== SOURCE: GenerationConfig.__init__ ===
{init_src}
"""


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------
def load_ground_truth():
    with open(GROUND_TRUTH_SCHEMA) as f:
        gt_schema = json.load(f)
    with open(GROUND_TRUTH_INV) as f:
        gt_inv = yaml.safe_load(f)
    return gt_schema, gt_inv


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def extract_all_fields_from_schema(schema: dict) -> set[str]:
    """Collect all field names from engine_params and sampling_params."""
    fields = set()
    for section in ("engine_params", "sampling_params"):
        for k in schema.get(section, {}).keys():
            fields.add(f"{section}.{k}")
    return fields


def score_schema(gt_schema: dict, llm_schema: dict) -> dict:
    gt_fields = extract_all_fields_from_schema(gt_schema)
    llm_fields = extract_all_fields_from_schema(llm_schema)

    intersection = gt_fields & llm_fields
    union = gt_fields | llm_fields
    jaccard = len(intersection) / len(union) if union else 0.0

    # Type accuracy on overlapping fields
    type_matches = 0
    type_total = 0
    for section in ("engine_params", "sampling_params"):
        for field, gt_def in gt_schema.get(section, {}).items():
            llm_section = llm_schema.get(section, {})
            if field in llm_section:
                type_total += 1
                gt_type = gt_def.get("type")
                llm_type = llm_section[field].get("type")
                if gt_type == llm_type:
                    type_matches += 1

    missed = gt_fields - llm_fields
    spurious = llm_fields - gt_fields

    return {
        "gt_field_count": len(gt_fields),
        "llm_field_count": len(llm_fields),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": jaccard,
        "recall": len(intersection) / len(gt_fields) if gt_fields else 0.0,
        "precision": len(intersection) / len(llm_fields) if llm_fields else 0.0,
        "type_accuracy": type_matches / type_total if type_total else 0.0,
        "type_match_count": type_matches,
        "type_total_overlapping": type_total,
        "spurious_count": len(spurious),
        "spurious_rate": len(spurious) / len(llm_fields) if llm_fields else 0.0,
        "missed_fields": sorted(missed),
        "spurious_fields": sorted(spurious),
    }


def make_invariant_key(inv: dict) -> tuple:
    """Extract (severity, primary_field, predicate_kind) triple for matching."""
    severity = inv.get("severity", "")
    match_fields = inv.get("match", {}).get("fields", {})

    # Primary field: the first sampling field in the match
    primary_field = ""
    predicate_kind = "exact"
    for k, v in match_fields.items():
        # k looks like "transformers.sampling.field_name"
        parts = k.split(".")
        if len(parts) >= 3:
            primary_field = parts[-1]
        if isinstance(v, dict):
            if "not_in" in v:
                predicate_kind = "not_in"
            elif "not_equal" in v:
                predicate_kind = "not_equal"
            elif ">" in v:
                predicate_kind = "gt"
            elif "<" in v:
                predicate_kind = "lt"
            elif "type_is_not" in v:
                predicate_kind = "type_is_not"
            elif "present" in v:
                predicate_kind = "present"
        break  # use first field only for keying

    return (severity, primary_field, predicate_kind)


def score_invariants(gt_inv: dict, llm_inv: dict | list) -> dict:
    gt_list = gt_inv.get("invariants", []) if isinstance(gt_inv, dict) else []
    if isinstance(llm_inv, list):
        llm_list = llm_inv
    elif isinstance(llm_inv, dict):
        llm_list = llm_inv.get("invariants", [])
    else:
        llm_list = []

    gt_keys = {}
    for inv in gt_list:
        k = make_invariant_key(inv)
        gt_keys[k] = inv

    llm_keys = {}
    for inv in llm_list:
        k = make_invariant_key(inv)
        llm_keys[k] = inv

    gt_key_set = set(gt_keys.keys())
    llm_key_set = set(llm_keys.keys())

    intersection = gt_key_set & llm_key_set
    union = gt_key_set | llm_key_set

    # Also match by invariant id for simpler reporting
    gt_ids = {inv.get("id", "") for inv in gt_list}
    llm_ids = {inv.get("id", "") for inv in llm_list}

    return {
        "gt_count": len(gt_list),
        "llm_count": len(llm_list),
        "key_intersection": len(intersection),
        "key_union": len(union),
        "jaccard": len(intersection) / len(union) if union else 0.0,
        "recall": len(intersection) / len(gt_key_set) if gt_key_set else 0.0,
        "precision": len(intersection) / len(llm_key_set) if llm_key_set else 0.0,
        "id_overlap": len(gt_ids & llm_ids),
        "missed_keys": sorted(gt_key_set - llm_key_set),
        "spurious_keys": sorted(llm_key_set - gt_key_set),
        "severity_breakdown_gt": _severity_breakdown(gt_list),
        "severity_breakdown_llm": _severity_breakdown(llm_list),
    }


def _severity_breakdown(inv_list: list) -> dict:
    counts = {"error": 0, "dormant": 0, "warning": 0, "other": 0}
    for inv in inv_list:
        sev = inv.get("severity", "other")
        counts[sev] = counts.get(sev, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# YAML extraction from text (model may emit markdown)
# ---------------------------------------------------------------------------
def extract_yaml_from_text(text: str) -> str:
    """Strip markdown code fences if present (yaml or yml, with or without closing fence)."""
    stripped = text.strip()
    # Try to extract ```yaml/yml ... ``` block (closing fence may be missing)
    m = re.search(r"```(?:ya?ml)?\s*\n(.*?)(?:```|$)", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If starts with fence line, strip first line
    lines = stripped.splitlines()
    if lines and re.match(r"^```", lines[0]):
        # Strip opening fence and optional closing fence
        content_lines = lines[1:]
        if content_lines and re.match(r"^```", content_lines[-1]):
            content_lines = content_lines[:-1]
        return "\n".join(content_lines).strip()
    return stripped


def extract_json_from_text(text: str) -> str:
    """Strip markdown code fences if present."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find JSON object
    m2 = re.search(r"(\{.*\})", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    schema_score: dict,
    inv_score: dict,
    schema_elapsed: float,
    inv_elapsed: float,
    schema_prompt: str,
    inv_prompt: str,
    llm_schema_raw: str,
    llm_inv_raw: str,
    llm_schema_parse_error: str | None,
    llm_inv_parse_error: str | None,
) -> str:

    total_elapsed = schema_elapsed + inv_elapsed
    schema_recall = schema_score.get("recall", 0.0)
    inv_recall = inv_score.get("recall", 0.0)

    # Verdict
    if schema_recall >= 0.80 and inv_recall >= 0.60:
        verdict = "WORTH PURSUING - high overlap signals viable LLM-as-extractor approach"
        verdict_tier = "HIGH"
    elif schema_recall >= 0.50 and inv_recall >= 0.30:
        verdict = "HYBRID VIABLE - medium overlap; LLM as backstop, handwritten machinery as floor"
        verdict_tier = "MEDIUM"
    else:
        verdict = (
            "NOT WORTH PURSUING - low overlap; handwritten machinery has more value than feared"
        )
        verdict_tier = "LOW"

    lines = []
    lines.append("# Bakeoff B: Local LLM (llama3.1:70b) as Knowledge Extractor")
    lines.append("")
    lines.append(f"**Verdict: {verdict_tier} - {verdict}**")
    lines.append("")
    lines.append(
        f"Schema recall: {schema_recall:.1%} | Invariants recall: {inv_recall:.1%} | Total wall-clock: {total_elapsed:.0f}s"
    )
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append("| Model | llama3.1:70b (Q4_K_M, ~42GB) |")
    lines.append("| Ollama endpoint | http://localhost:11434 |")
    lines.append("| Hardware | A100-40GB |")
    lines.append(f"| Schema call wall-clock | {schema_elapsed:.0f}s |")
    lines.append(f"| Invariants call wall-clock | {inv_elapsed:.0f}s |")
    lines.append(f"| Total wall-clock | {total_elapsed:.0f}s |")
    lines.append(f"| Schema prompt chars | {len(schema_prompt):,} |")
    lines.append(f"| Invariants prompt chars | {len(inv_prompt):,} |")
    lines.append("")

    lines.append("## Schema Scoring")
    lines.append("")
    if llm_schema_parse_error:
        lines.append(f"**PARSE ERROR**: `{llm_schema_parse_error}`")
        lines.append("")
        lines.append("Raw LLM output (first 2000 chars):")
        lines.append("```")
        lines.append(llm_schema_raw[:2000])
        lines.append("```")
    else:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Ground truth fields | {schema_score['gt_field_count']} |")
        lines.append(f"| LLM-produced fields | {schema_score['llm_field_count']} |")
        lines.append(f"| Intersection (overlap) | {schema_score['intersection']} |")
        lines.append(f"| Recall (GT coverage) | {schema_score['recall']:.1%} |")
        lines.append(f"| Precision | {schema_score['precision']:.1%} |")
        lines.append(f"| Jaccard | {schema_score['jaccard']:.1%} |")
        lines.append(
            f"| Type accuracy (overlapping) | {schema_score['type_accuracy']:.1%} ({schema_score['type_match_count']}/{schema_score['type_total_overlapping']}) |"
        )
        lines.append(
            f"| Spurious fields | {schema_score['spurious_count']} ({schema_score['spurious_rate']:.1%} of LLM output) |"
        )
        lines.append("")

        if schema_score["missed_fields"]:
            lines.append("### Missed fields (in GT, not in LLM output)")
            lines.append("")
            for f in schema_score["missed_fields"][:30]:
                lines.append(f"- `{f}`")
            if len(schema_score["missed_fields"]) > 30:
                lines.append(f"- ... and {len(schema_score['missed_fields']) - 30} more")
            lines.append("")

        if schema_score["spurious_fields"]:
            lines.append("### Spurious fields (in LLM output, not in GT)")
            lines.append("")
            for f in schema_score["spurious_fields"][:20]:
                lines.append(f"- `{f}`")
            if len(schema_score["spurious_fields"]) > 20:
                lines.append(f"- ... and {len(schema_score['spurious_fields']) - 20} more")
            lines.append("")

    lines.append("## Invariants Scoring")
    lines.append("")
    if llm_inv_parse_error:
        lines.append(f"**PARSE ERROR**: `{llm_inv_parse_error}`")
        lines.append("")
        lines.append("Raw LLM output (first 2000 chars):")
        lines.append("```")
        lines.append(llm_inv_raw[:2000])
        lines.append("```")
    else:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Ground truth invariants | {inv_score['gt_count']} |")
        lines.append(f"| LLM-produced invariants | {inv_score['llm_count']} |")
        lines.append(f"| Key-match overlap | {inv_score['key_intersection']} |")
        lines.append(f"| Recall (GT coverage) | {inv_score['recall']:.1%} |")
        lines.append(f"| Precision | {inv_score['precision']:.1%} |")
        lines.append(f"| Jaccard | {inv_score['jaccard']:.1%} |")
        lines.append(f"| ID exact matches | {inv_score['id_overlap']} |")
        lines.append("")

        lines.append("### Severity breakdown")
        lines.append("")
        lines.append("| Severity | Ground Truth | LLM Output |")
        lines.append("|----------|-------------|-----------|")
        for sev in ("error", "dormant", "warning", "other"):
            gt_n = inv_score["severity_breakdown_gt"].get(sev, 0)
            llm_n = inv_score["severity_breakdown_llm"].get(sev, 0)
            lines.append(f"| {sev} | {gt_n} | {llm_n} |")
        lines.append("")

        if inv_score["missed_keys"]:
            lines.append("### Missed invariant keys (in GT, not matched in LLM)")
            lines.append("")
            for k in inv_score["missed_keys"][:20]:
                lines.append(f"- `{k}`")
            if len(inv_score["missed_keys"]) > 20:
                lines.append(f"- ... and {len(inv_score['missed_keys']) - 20} more")
            lines.append("")

        if inv_score["spurious_keys"]:
            lines.append("### Spurious invariant keys (in LLM output, not in GT)")
            lines.append("")
            for k in inv_score["spurious_keys"][:20]:
                lines.append(f"- `{k}`")
            if len(inv_score["spurious_keys"]) > 20:
                lines.append(f"- ... and {len(inv_score['spurious_keys']) - 20} more")
            lines.append("")

    lines.append("## Prompts Used")
    lines.append("")
    lines.append("### Schema prompt (truncated to 3000 chars)")
    lines.append("```")
    lines.append(schema_prompt[:3000])
    lines.append("...")
    lines.append("```")
    lines.append("")
    lines.append("### Invariants prompt (truncated to 3000 chars)")
    lines.append("```")
    lines.append(inv_prompt[:3000])
    lines.append("...")
    lines.append("```")
    lines.append("")

    lines.append("## Analysis of Misses and Spurious")
    lines.append("")
    lines.append("### Schema analysis")
    lines.append("")
    if not llm_schema_parse_error:
        missed = schema_score["missed_fields"]
        ep_missed = [f for f in missed if f.startswith("engine_params.")]
        sp_missed = [f for f in missed if f.startswith("sampling_params.")]
        lines.append(f"- Missed engine_params: {len(ep_missed)} fields")
        lines.append(f"- Missed sampling_params: {len(sp_missed)} fields")
        lines.append(f"- Spurious fields: {len(schema_score['spurious_fields'])}")
        lines.append("")
        lines.append(
            "The schema call tests whether the model can read Python function signatures and class definitions "
            "and emit structured JSON Schema output. Key failure modes: (a) missing optional/rare params, "
            "(b) type misattribution for weakly-annotated fields, (c) missing BitsAndBytesConfig fields "
            "that surface as engine kwargs."
        )
    else:
        lines.append("Schema call failed to parse - see error above.")
    lines.append("")

    lines.append("### Invariants analysis")
    lines.append("")
    if not llm_inv_parse_error:
        lines.append(
            "The invariants call tests whether the model can read validation logic and emit structured "
            "predicate representations. Key failure modes: (a) missing cross-field invariants, "
            "(b) incorrect predicate kinds (e.g. calling 'not_in' when it should be 'type_is_not'), "
            "(c) spurious invariants that don't match actual library raises, "
            "(d) YAML structure violations."
        )
        lines.append("")
        gt_sev = inv_score["severity_breakdown_gt"]
        llm_sev = inv_score["severity_breakdown_llm"]
        lines.append(
            f"GT has {gt_sev.get('error', 0)} error / {gt_sev.get('dormant', 0)} dormant / {gt_sev.get('warning', 0)} warning invariants."
        )
        lines.append(
            f"LLM produced {llm_sev.get('error', 0)} error / {llm_sev.get('dormant', 0)} dormant / {llm_sev.get('warning', 0)} warning invariants."
        )
    else:
        lines.append("Invariants call failed to parse - see error above.")
    lines.append("")

    lines.append("## Cost Estimate")
    lines.append("")
    lines.append(
        f"Wall-clock: {total_elapsed:.0f}s ({schema_elapsed:.0f}s schema + {inv_elapsed:.0f}s invariants)"
    )
    lines.append("")
    lines.append(
        "Energy: A100 TDP ~250W. At full load: "
        f"{total_elapsed * 250 / 3600 / 1000 * 1000:.1f} Wh (~{total_elapsed * 250 / 3600 / 1000:.3f} kWh). "
        "Actual draw likely 150-200W during inference."
    )
    lines.append("")
    lines.append(
        "Per-run cost vs handwritten machinery: The handwritten miners are one-time authored; "
        "this LLM approach runs per version bump. If the model needs to be re-queried on each "
        "library version upgrade (~6/year), this is still trivially cheap vs engineering time."
    )
    lines.append("")

    lines.append("## Verdict and Next Steps")
    lines.append("")
    lines.append(f"**{verdict_tier}: {verdict}**")
    lines.append("")

    if verdict_tier == "HIGH":
        lines.append("Recommended next steps:")
        lines.append(
            "- Invest in structuring the LLM-as-extractor approach as a producer pipeline stage"
        )
        lines.append(
            "- Add validation step: run each LLM-emitted invariant's kwargs_positive/kwargs_negative through the live library"
        )
        lines.append("- Evaluate against a second engine (vllm) to confirm generalisability")
        lines.append(
            "- Consider using a commercial frontier model (GPT-4o, claude-3-7-sonnet) for comparison"
        )
    elif verdict_tier == "MEDIUM":
        lines.append("Recommended next steps:")
        lines.append("- Use LLM as a first-pass walker to identify candidate fields/invariants")
        lines.append("- Keep handwritten machinery as the ground-truth floor")
        lines.append(
            "- Investigate which invariant categories the LLM consistently gets right vs wrong"
        )
        lines.append("- Evaluate whether prompt improvements move recall above the HIGH threshold")
    else:
        lines.append("Recommended next steps:")
        lines.append("- Keep the handwritten machinery as the primary approach")
        lines.append(
            "- Consider LLM only for documentation/description enrichment, not structural extraction"
        )
        lines.append(
            "- Revisit with a more capable model (70B Q4_K_M may be too constrained for this task)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Bakeoff B: Local LLM Knowledge Extractor POC ===")
    print()

    # Load ground truth
    gt_schema, gt_inv = load_ground_truth()
    gt_inv_count = len(gt_inv.get("invariants", []))
    gt_field_count = len(extract_all_fields_from_schema(gt_schema))
    print(f"Ground truth: {gt_field_count} schema fields, {gt_inv_count} invariants")
    print()

    # Get source code
    sources = get_sources()
    print()

    # ---------------------------------------------------------------------------
    # Call 1: Schema
    # ---------------------------------------------------------------------------
    print("=== CALL 1: Schema extraction ===")
    schema_prompt = SCHEMA_PROMPT_TEMPLATE.format(
        from_pretrained_src=sources["AutoModelForCausalLM.from_pretrained"],
        bnb_src=sources["BitsAndBytesConfig"],
        init_src=sources["GenerationConfig.__init__"],
        gc_doc=sources["GenerationConfig.__doc__"],
        compile_config_src=sources["CompileConfig"],
        watermarking_src=sources["WatermarkingConfig"],
    )
    print(f"Schema prompt: {len(schema_prompt):,} chars")

    llm_schema_raw = ""
    llm_schema = {}
    llm_schema_parse_error = None
    schema_elapsed = 0.0

    try:
        llm_schema_raw, schema_elapsed = ollama_call(schema_prompt, use_json_format=True)
        # Save raw
        (FINDINGS / "bakeoff_B_schema_raw.txt").write_text(llm_schema_raw)
        # Parse
        json_text = extract_json_from_text(llm_schema_raw)
        llm_schema = json.loads(json_text)
        (FINDINGS / "bakeoff_B_schema.json").write_text(json.dumps(llm_schema, indent=2))
        print(
            f"Schema parsed OK: engine_params={len(llm_schema.get('engine_params', {}))}, sampling_params={len(llm_schema.get('sampling_params', {}))}"
        )
    except Exception as e:
        llm_schema_parse_error = str(e)
        print(f"Schema call/parse FAILED: {e}")
        # Save whatever we got
        (FINDINGS / "bakeoff_B_schema_raw.txt").write_text(llm_schema_raw)
        (FINDINGS / "bakeoff_B_schema.json").write_text(
            json.dumps({"error": llm_schema_parse_error, "raw": llm_schema_raw[:5000]}, indent=2)
        )

    print()

    # ---------------------------------------------------------------------------
    # Call 2: Invariants
    # ---------------------------------------------------------------------------
    print("=== CALL 2: Invariants extraction ===")
    inv_prompt = INVARIANTS_PROMPT_TEMPLATE.format(
        validate_src=sources["GenerationConfig.validate"],
        init_src=sources["GenerationConfig.__init__"],
    )
    print(f"Invariants prompt: {len(inv_prompt):,} chars")

    llm_inv_raw = ""
    llm_inv = {}
    llm_inv_parse_error = None
    inv_elapsed = 0.0

    try:
        llm_inv_raw, inv_elapsed = ollama_call(inv_prompt, use_json_format=False)
        # Save raw
        (FINDINGS / "bakeoff_B_invariants_raw.txt").write_text(llm_inv_raw)
        # Parse YAML
        yaml_text = extract_yaml_from_text(llm_inv_raw)
        llm_inv = yaml.safe_load(yaml_text)
        if llm_inv is None:
            llm_inv = {}
        with open(FINDINGS / "bakeoff_B_invariants.yaml", "w") as f:
            yaml.dump(llm_inv, f, default_flow_style=False, allow_unicode=True)
        inv_count = len(llm_inv.get("invariants", [])) if isinstance(llm_inv, dict) else 0
        print(f"Invariants parsed OK: {inv_count} invariants")
    except Exception as e:
        llm_inv_parse_error = str(e)
        print(f"Invariants call/parse FAILED: {e}")
        (FINDINGS / "bakeoff_B_invariants_raw.txt").write_text(llm_inv_raw)
        (FINDINGS / "bakeoff_B_invariants.yaml").write_text(
            f"# PARSE ERROR: {llm_inv_parse_error}\n# Raw output:\n{llm_inv_raw[:5000]}"
        )

    print()

    # ---------------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------------
    print("=== SCORING ===")
    schema_score = score_schema(gt_schema, llm_schema)
    inv_score = score_invariants(gt_inv, llm_inv)

    print(
        f"Schema: recall={schema_score['recall']:.1%} precision={schema_score['precision']:.1%} jaccard={schema_score['jaccard']:.1%}"
    )
    print(
        f"Invariants: recall={inv_score['recall']:.1%} precision={inv_score['precision']:.1%} jaccard={inv_score['jaccard']:.1%}"
    )
    print()

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("=== GENERATING REPORT ===")
    report = generate_report(
        schema_score=schema_score,
        inv_score=inv_score,
        schema_elapsed=schema_elapsed,
        inv_elapsed=inv_elapsed,
        schema_prompt=schema_prompt,
        inv_prompt=inv_prompt,
        llm_schema_raw=llm_schema_raw,
        llm_inv_raw=llm_inv_raw,
        llm_schema_parse_error=llm_schema_parse_error,
        llm_inv_parse_error=llm_inv_parse_error,
    )

    report_path = FINDINGS / "bakeoff_B_local_llm.md"
    report_path.write_text(report)
    print(f"Report written to {report_path}")

    # Save scores as JSON for programmatic use
    scores = {
        "schema": schema_score,
        "invariants": inv_score,
        "timing": {"schema_elapsed": schema_elapsed, "inv_elapsed": inv_elapsed},
    }
    (FINDINGS / "bakeoff_B_scores.json").write_text(json.dumps(scores, indent=2))

    print()
    print("=== DONE ===")
    print(f"Schema recall: {schema_score['recall']:.1%}")
    print(f"Invariants recall: {inv_score['recall']:.1%}")


if __name__ == "__main__":
    main()
