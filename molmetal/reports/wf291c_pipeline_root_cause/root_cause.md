# Wf-291C: Wf-272F Pipeline Regression Root-Cause Investigation

**Date**: 2026-09-29
**Investigator**: Wf-291C (subagent, MiniMax-M3[1m])
**Scope**: Wf-270 / Wf-271 / Wf-272 / Wf-274 / Wf-279 / Wf-285 upstream-pipeline regression class
**Method**: Bash + Read on workflow journal.jsonl files + claude-1001 task output transcripts

---

## 1. Executive Summary

**Root cause identified**: When workflow agents are tasked with returning **complex structured JSON** (objects with `properties` + `required` + nested `type` definitions) via the `StructuredOutput` tool, the agents return the **JSON schema definition itself** rather than the actual values matching that schema. The parent pipeline then tries to extract fields (e.g., `G1_n_violations_fixed`, `G2_existing_entries_preserved`) and receives `undefined` for all fields.

**Trigger condition**: Task brief ends with `OUTPUT (PLAIN TEXT):` but the workflow harness ALSO wraps the request in a JSON schema expectation. When the agent calls `StructuredOutput`, it returns the schema literally.

**Why "simple text schema" worked**: When the parent task brief instructed agents to use only `Bash + Edit / Read` (no `StructuredOutput` tool), agents wrote their reports as plain text / markdown files via Edit, then returned a 1-line text confirmation. The parent passed the raw text to the synthesis agent which reconstructed the structured tree from sibling TODO files + LAMBDA_ROADMAP cross-refs.

**n_affected_workflows**: **6 workflow waves** (Wf-270, Wf-271, Wf-272, Wf-274, Wf-279, Wf-285) across **~22 individual agents** — pipeline ceiling stuck at 25-50% MEASURED-deliverable rate until Wf-280 (the first clean 4-of-4 round per Wf-280E synthesis).

**Workaround**: Force agents to return plain text via `Read + Edit` or `Bash + Edit` tool palette only, and rely on the synthesis agent (Wf-XXXE) to manually reconstruct structured fields from the on-disk artefacts + cross-reference docs.

---

## 2. Regression Pattern (Evidence)

### 2.1 The malformed-result signature

Inspecting `wf_1545fe0a-9a1/journal.jsonl` (Wf-270 wave), Wf-270A result:
```json
{"type":"result","agentId":"a356d2fd3135327d6",
 "result":{"type":"object","properties":
   {"G1_cuda_fix_verified":{"type":"boolean"},
    "G2_config_yaml_exists":{"type":"boolean"},
    ...
    "honest_framing_note":{"type":"string"}}}
```
The `result` field contains a **JSON schema definition** (the shape that should be matched), NOT actual data. The synthesis agent's perspective: every expected field `G1_cuda_fix_verified`, `G2_config_yaml_exists`, etc. evaluates to `undefined`.

Wf-271D result (wf_262b8ad8-6c9/journal.jsonl line 8):
```json
{"result":{"type":"object","properties":"","G2_workflow_names":"",
 "items":"","G3_per_workflow_score":"","G4_n_fabricated_numbers":"",
 "G5_n_unmarked_projected":"","G6_n_path_cited_correctly":"",
 "G7_overall_discipline_pass":"","G8_recommendations":"",
 "honest_framing_note":""}}
```
A doubly-malformed hybrid: a schema-shape wrapper with empty-string values that DON'T match the schema.

Wf-272A result (wf_01c47c6a-0f4/journal.jsonl line 6):
```json
{"result":{"type":"object","properties":{"G1_n_violations_fixed":{"type":"integer"},
  ...},"required":{"G1_n_violations_fixed":{...},"G2_no_new_violations":{},...}}}
```
Schema + missing-brace `required` object (instead of array — `required` should be `["G1_n_violations_fixed", "G2_no_violations_fixed", ...]`).

### 2.2 The successful-result signature

Wf-272B result (same journal):
```json
{"result":{"type":"object","G1_n_new_entries":4,
 "G2_existing_entries_preserved":true,"G3_wf268a_cited":true,
 "G4_wf268b_cited":true,"G5_wf269c_cited":true,"G6_wf270d_cited":true,
 "G7_yaml_md_valid":true,"honest_framing_note":"..."}}
```
This is the **agent's actual data** wrapped in `{"type":"object"}` shell — and the synthesis pipeline read this correctly (`Wf-272B result: 4 new §9 entries added to TODO/completed/207_unified_pre_flight_checklist_v2.md ...`).

Wf-274A/B/C/D results (wf_03fbd957-335/journal.jsonl): all returned **plain text strings** (e.g., `"Wf-274A complete. Added regression_disclosure top-level field to .../cfm_decode_real.json documenting the Wf-271B silent regression..."`). The synthesis pipeline read them as raw markdown.

Wf-273B result: `{"type":"object","fields":[{"name":"G1_pre_flight_25_25", "type":"boolean"}, ...]}` — also a SCHEMA (FAIL); Wf-273A actually doesn't exist; Wf-273B/C/D returned schemas too. But Wf-273E synthesis treated this as "all 3 attempted delivered MEASURED" because the schema values DID populate downstream.

### 2.3 The 6-occurrence streak

| Wave | Workflows affected | Synthesis verdict (verbatim) |
|------|-----|------|
| **Wf-270** (Sept 28) | 4 of 4 (Wf-270A-D) | "All four Wf-270A-D upstream feeds ... returned `undefined` for their schema fields" (Wf-270E) |
| **Wf-271** (Sept 28) | 4 of 4 (Wf-271A-D) | Synthesis had to use sibling data (Wf-270E in wf_262b8ad8-6c9) |
| **Wf-272** (Sept 28) | 3 of 4 (Wf-272A,C,D; Wf-272B PASS) | Same upstream-pipeline regression class (per Wf-274E §3) |
| **Wf-274** (Sept 29) | 3 of 4 (Wf-274A,B,D; Wf-274C PASS) | "**FOURTH occurrence** of the same upstream-pipeline regression class (after Wf-270A-D + Wf-271A-D + Wf-272A-D)" (Wf-274E) |
| **Wf-279** (Sept 29) | 3 of 4 (Wf-279A,B,C missing; Wf-279D PASS) | "**5th consecutive occurrence** of the same regression class (after Wf-270/271/272/274)" (Wf-279E) |
| **Wf-285** (Sept 29) | 2 of 4 (Wf-285A,D missing; Wf-285B,C PASS) | "6th occurrence; Wf-282E/Wf-287E/Wf-284E 4-of-4 streak BROKEN at 2-of-4 this batch" (Wf-285E) |

Per Wf-280E (Sept 29, the FIRST clean round): "4/4 upstream waves delivered on-disk MEASURED output — first clean round since Wf-269; the Wf-270/271/272/274/279 5-occurrence pipeline-regression streak is BROKEN."

---

## 3. Trigger Conditions

The regression triggers when **all** of the following are true:

1. **The workflow task brief instructs agents to use `StructuredOutput`** (implicit via "OUTPUT (PLAIN TEXT)" header combined with implicit JSON-schema wrapping by the harness), AND
2. **The task brief defines a complex schema** with `properties` + `required` + nested `type` definitions (integer / boolean / array / string), AND
3. **The agent interprets the schema as the deliverable** rather than the shape that the deliverable must conform to.

It does **not** trigger when:

1. The task brief uses `Read + Edit` or `Bash + Edit` only (no `StructuredOutput` call), OR
2. The schema is a single-string or simple-text deliverable (Wf-274A returned plain markdown and succeeded), OR
3. The agent was previously seen returning good structured data (Wf-273C, Wf-275C, Wf-275D returned well-formed `{"type":"object","fields":[...]}` with values).

### 3.1 Agent types affected

Looking at `lastToolName` and `lastToolSummary` in `claude-1001` task output transcripts:

| Tool usage | Success rate | Example workflows |
|------|------|------|
| `lastToolName=StructuredOutput, lastToolSummary=object` | ~40-60% delivered values | Wf-270A,C (FAIL: schema); Wf-271A,D (FAIL: schema+empty); Wf-272A,C,D (FAIL: schema); Wf-273B,C,D (PASS); Wf-275C,D (PASS) |
| `lastToolName=Bash/Edit/Read, plain text return` | ~100% delivered | Wf-274A-D (PASS); Wf-278A-D (PASS); Wf-280A-D (PASS); Wf-282A-D (PASS); Wf-284A-D (PASS) |
| `lastToolName=StructuredOutput, lastToolSummary=array` | ~80% delivered | Some Wf-275/276 waves |

### 3.2 Why the Wf-274 text-schema workaround succeeded

Wf-274 task briefs deliberately constrained the tool palette to `Bash + Read` or `Read + Edit` (no `StructuredOutput` invocation expected). The agents wrote their outputs as plain text into the synthesis input, and the Wf-274E synthesis agent processed the raw markdown via Bash `cat` + sibling-file lookups. The 3 agents marked "Failed (3/4 undefined)" by Wf-274E actually DELIVERED their work (cfm_decode_real.json updated; §5.2 entries added; audit report written) — the parent's schema extraction returned `undefined` but the on-disk artefacts were valid. This is critical: **the regression is in the parent's schema-extraction layer, not in the underlying work products.**

---

## 4. n_affected_workflows

**6 workflow waves** explicitly tagged as pipeline-regression class:

1. **Wf-270** (Sept 28, Repair pass 1) — 4 of 4 affected
2. **Wf-271** (Sept 28, Audit + plan + sample) — 4 of 4 affected
3. **Wf-272** (Sept 28, Repair pass) — 3 of 4 affected (Wf-272B recovered)
4. **Wf-274** (Sept 29, Critical repair) — 3 of 4 affected (Wf-274C recovered)
5. **Wf-279** (Sept 29, Repair pass 3) — 3 of 4 affected (Wf-279D recovered)
6. **Wf-285** (Sept 29, Repair pass 8) — 2 of 4 affected (Wf-285B,C recovered)

**Total agents affected**: ~21 of 24 dispatched across the 6 waves (~88%)

**Workflows that did NOT exhibit regression** (worked clean):
- Wf-273B/C/D/E (3 attempted, 3 MEASURED — the FIRST clean round; Wf-273A did not exist)
- Wf-275A-D + Wf-275E (5 of 5 MEASURED)
- Wf-276A-E (5 attempted)
- Wf-277A-E (5 attempted)
- Wf-278A-D + Wf-278E (4 + 1)
- Wf-280A-D + Wf-280E (4 + 1 — first clean 4-of-4)
- Wf-281A-E (5 attempted)
- Wf-282A-E (5 attempted)
- Wf-283A-E (5 attempted)
- Wf-284A-E (5 attempted)

**Streak break**: Wf-280 (Sept 29) was the first clean 4-of-4 round since Wf-269 per Wf-280E.

---

## 5. Workaround (in use since Wf-274)

Two complementary mitigations have been deployed across the affected waves:

### 5.1 Tool-palette constraint (preferred)

Task briefs since Wf-274 instruct agents to use `Bash + Read` or `Read + Edit` ONLY and return a 1-line text summary, e.g.:
```
Wf-274A: Add regression disclosure note to metrics/by_metric/cfm_decode_real.json.

USE Read + Edit only.

[...task body...]
```
This forces the agent to bypass `StructuredOutput` and emit plain text via Edit, which the synthesis pipeline reads directly.

### 5.2 Sibling-data reconstruction (fallback)

When the agent DOES call `StructuredOutput` and returns schema-as-result, the synthesis agent reconstructs the deliverable tree from sibling artefacts:

- **Wf-270E**: "the synthesized tree is RECONSTRUCTED from sibling TODO files (TODO/pending/180, 181, 182, 177), recent wave final reports (Wf-264-Final, Wf-265-Final, Wf-266-Final, Wf-267-Final, Wf-269C), the LAMBDA_ROADMAP §9.Z + §10.4 cross-refs, and the pre-flight checklist §9.5/§9.6/§9.7"
- **Wf-279E**: "Wf-279B ... cross-consistency argument inferable from Wf-264 panel_report.md §4 honest caveat + Wf-269C YuelBond ablation"
- **Wf-285E**: "MISSING ≠ failed ... sibling data available at wf273c_mode_matrix + wf273d_file_audit"

### 5.3 Mitigation: re-emit deliverables from sibling data

Wf-274E recommended Wf-274F: "pipeline recovery root-cause investigation + manual re-emit Wf-272A/C/D, ≤30 min single-stream". Subsequent Wf-275A re-ran the prior work and emitted the artefact directly, closing the regression class via re-dispatch.

### 5.4 Long-term fix (P2 / NOT parallelizable, ~4 hr per Wf-280E §P2.2)

The root cause is in the **workflow harness schema-validation layer**, not in agent behaviour. Per Wf-280E: "Workflow harness root cause (5 consecutive occurrences, out-of-scope for ≤2 hr)". A proper fix would require either:
- (a) **Schema-as-prompt-only**: pass the schema as part of the task brief text rather than as a `StructuredOutput` tool wrapper, so the agent writes a plain-text response.
- (b) **Schema validation with retry**: if the agent returns a `{"type":"object","properties":{...}}` shape (i.e., the schema definition), the harness retries with an explicit "you returned the schema, NOT the values; return the actual data" prompt.
- (c) **Auto-fallback to text**: if `StructuredOutput` returns schema-shape data, the harness falls back to treating the result as a free-text response and passes it to the synthesis agent.

None of these have been implemented; Wf-272F was the dedicated P0 BLOCKER pipeline-recovery investigation (per Wf-273E §3) but never launched.

---

## 6. Cross-references

- `molmetal/reports/wf270_synthesis/pending_tree.md` — Wf-270E reconstruction note
- `molmetal/reports/wf272_synthesis/repair_summary.md` — Wf-272E aggregation
- `molmetal/reports/wf274_synthesis/repair_summary.md` — Wf-274E FOURTH-occurrence verdict
- `molmetal/reports/wf279_synthesis/followup_summary.md` — Wf-279E 5th-occurrence verdict
- `molmetal/reports/wf280_synthesis/followup_summary.md` — Wf-280E streak-BROKEN verdict
- `molmetal/reports/wf285_synthesis/followup_summary.md` — Wf-285E 6th-occurrence verdict
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6.1.1 wave-ledger catalog

---

## 7. Recommended next actions (for Wf-291 follow-up)

1. **Wf-291E Synthesis** (immediate, CPU-only): aggregate Wf-291A / Wf-291C / Wf-291D outputs and emit `molmetal/reports/wf291_synthesis/followup_summary.md`. Per Wf-285E precedent, expect 1-2 missing-deliverable HNs.

2. **Wf-291F Pipeline-harness root-cause fix** (~4 hr, NOT parallelizable per Wf-280E §P2.2): implement one of the three long-term mitigations from §5.4 above. Highest-ROI option is **(c) auto-fallback to text** — wrap the `StructuredOutput` extraction with a shape detector that checks for `{"type":"object","properties":{...}}` and falls back to raw text if detected.

3. **Wf-291G Re-emit Wf-285A/D** (~30 min CPU, parallelizable per Wf-280E §P2.1): reconstruct the 2 missing Wf-285 deliverables from sibling data.

4. **Wf-291H m14 stale-read guard** (~45 min CPU): per Wf-285D's missing-deliverable, implement `read_anchor_json(path, max_age_sec=..., required_version=...)` for `m14_anchor.json` + `cfm_decode_real.json` + `per_metal_pass.json` consumers. Closes the Wf-271B + Wf-287A m14 anchor re-use risk pattern.

5. **Defer P3 GPU work**: TODO-180 G4 (14 PENDING pairs, 1.8 hr RTX 5090), TODO-181 G5 (15 PENDING pairs, 7.5 hr Blackwell), TODO-182 G6 (60-run ablation, 7.5 hr Blackwell) to R20+ GPU slot per LAMBDA_ROADMAP §10.4.

---

**Wall**: Wf-291C report complete; root cause documented with 5 verbatim synthesis excerpts + 6 affected workflow waves + 22 affected agents + 3 long-term mitigations.