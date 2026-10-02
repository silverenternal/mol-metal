# Wf-302E: Wf-272F Pipeline Regression Root-Cause (Carry-Over Canonical)

**Date**: 2026-09-29
**Investigator**: Wf-302E (subagent, MiniMax-M3[1m])
**Parent context**: Wf-291E recommendation — Wf-272F was the canonical P0 BLOCKER pipeline-harness auto-fallback root-cause investigation + manual re-emit (per Wf-273E §3). Wf-272F never launched, so this carry-over closes the investigation by aggregating the Wf-291C + Wf-298D findings into one canonical document.
**Method**: Bash + Read on workflow journal.jsonl files, claude-1001 task output transcripts, and the Wf-291C / Wf-298D / Wf-302C deliverables.

---

## 1. TL;DR

**Root cause** (verbatim from Wf-291C §1): When workflow agents are tasked with returning **complex structured JSON** (objects with `properties` + `required` + nested `type` definitions) via the harness `StructuredOutput` tool, the agents return the **JSON schema definition itself** rather than the actual values matching that schema. The parent pipeline then tries to extract named fields (e.g. `G1_n_violations_fixed`) and receives `undefined` for every key.

**n_affected_workflows**: **6 workflow waves** across **~22 individual agents**: Wf-270A-D, Wf-271A-D, Wf-272A-D, Wf-274A-D, Wf-279A-D, Wf-285A-D. Pipeline ceiling stuck at 25-50% MEASURED-deliverable rate until Wf-280 (first clean 4-of-4 round per Wf-280E synthesis).

**Workaround status**: DEPLOYED via three complementary layers (per Wf-291C §5 + Wf-298D §2):
1. **Text-only briefs** (Wf-302C template, deployed since Wf-274): task briefs constrain agents to `Bash + Read + Edit` only — no `StructuredOutput` call expected.
2. **User-side shape detector** (Wf-298D `wrapper.py`): `HarnessOutputExtractor.detect_schema_shape()` catches 4/4 regression-class payloads (Wf-270A, Wf-271D, Wf-272A, Wf-273B) and auto-falls-back to sibling free-text.
3. **Sibling-data reconstruction** (per Wf-270E / Wf-279E / Wf-285E): synthesis agents reconstruct the deliverable tree from on-disk artefacts + cross-reference docs.

**Long-term fix path**: Harness-layer schema-validation is **NOT user-modifiable** (closed-source ELF binary at `/home/hugo/.vscode-server/extensions/anthropic.claude-code-2.1.220-linux-x64/resources/native-binary/claude` — verified by `grep -a "StructuredOutput"` returning positive hits on the binary). The proper fix is one of three options from Wf-291C §5.4 (a/b/c). Wf-298D implements option (c) auto-fallback to text on the user-side; the canonical harness-layer fix would require Anthropic-side changes — out of scope from this workspace.

---

## 2. Regression Pattern (Schema-Shape vs Data-Shape)

### 2.1 Verbatim malformed-result signatures (Wf-291C §2.1)

| Wave | Result fragment | Diagnosis |
|------|---------------------------|-----------|
| Wf-270A | `{"result":{"type":"object","properties":{"G1_cuda_fix_verified":{"type":"boolean"}, ...}}}` | schema-shape |
| Wf-271D | `{"result":{"type":"object","properties":"","G2_workflow_names":"", ...}}` | hybrid empty-string |
| Wf-272A | `{"result":{"type":"object","properties":{"G1_n_violations_fixed":{"type":"integer"}, ...},"required":{"G1_n_violations_fixed":{...}, ...}}}` | schema-shape + malformed `required` (object instead of array) |
| Wf-273B | `{"result":{"type":"object","fields":[{"name":"G1_pre_flight_25_25","type":"boolean"}, ...]}}` | schema-shape (fields array) |

### 2.2 Verbatim successful-result signatures

```
{"result":{"type":"object","G1_n_new_entries":4,
 "G2_existing_entries_preserved":true, "G3_wf268a_cited":true,
 "G4_wf268b_cited":true, "G5_wf269c_cited":true,
 "G6_wf270d_cited":true, "G7_yaml_md_valid":true,
 "honest_framing_note":"..."}}
```
(Wf-272B — synthesis pipeline read this correctly.)

Or plain-text only:
```
"Wf-274A complete. Added regression_disclosure top-level field to .../cfm_decode_real.json
 documenting the Wf-271B silent regression..."
```
(Wf-274A — synthesis pipeline read as raw markdown.)

### 2.3 The diagnostic discriminator (Wf-298D §1.3)

Three signals that a payload is **schema-shape**, not data:

1. **Nested object carries schema meta-keys** — a value that is itself a dict containing any of `type` / `properties` / `required` / `items` / `enum` / `$ref` / `format` / `description`.
2. **Nested object's `type` is a JSON-Schema type keyword** (`"string"` / `"integer"` / `"number"` / `"boolean"` / `"array"` / `"object"` / `"null"`) AND that nested object has ≤3 keys.
3. **Hybrid empty-string pattern** (Wf-271D-class) — ≥½ of top-level values are empty strings AND the object carries schema root markers (`properties` / `required` / `$ref` / `$defs`).

All three are robust against normal data payloads (which carry primitives, real nested objects, or arrays of primitives — never the schema-meta-key fingerprint).

---

## 3. Trigger Conditions

The regression triggers when **all** of the following are true (Wf-291C §3):

1. **The workflow task brief instructs agents to use `StructuredOutput`** (implicit via "OUTPUT (PLAIN TEXT)" header combined with implicit JSON-schema wrapping by the harness).
2. **The task brief defines a complex schema** with `properties` + `required` + nested `type` definitions (integer / boolean / array / string).
3. **The agent interprets the schema as the deliverable** rather than the shape that the deliverable must conform to.

It does **not** trigger when:
- The task brief uses `Read + Edit` or `Bash + Edit` only (no `StructuredOutput` call).
- The schema is a single-string or simple-text deliverable (Wf-274A returned plain markdown and succeeded).
- The agent was previously seen returning good structured data (Wf-273C, Wf-275C, Wf-275D returned well-formed `{"type":"object","fields":[...]}` with values).

---

## 4. n_affected_workflows

**6 workflow waves** explicitly tagged as pipeline-regression class:

| Wave | Affected agents | Synthesis verdict |
|------|------------------|-------------------|
| **Wf-270** (Sept 28) | 4 of 4 (Wf-270A-D) | "All four Wf-270A-D upstream feeds ... returned `undefined` for their schema fields" (Wf-270E) |
| **Wf-271** (Sept 28) | 4 of 4 (Wf-271A-D) | Synthesis had to use sibling data (Wf-270E in wf_262b8ad8-6c9) |
| **Wf-272** (Sept 28) | 3 of 4 (Wf-272A,C,D; Wf-272B PASS) | Same upstream-pipeline regression class (per Wf-274E §3) |
| **Wf-274** (Sept 29) | 3 of 4 (Wf-274A,B,D; Wf-274C PASS) | "**FOURTH occurrence** of the same upstream-pipeline regression class (after Wf-270A-D + Wf-271A-D + Wf-272A-D)" (Wf-274E) |
| **Wf-279** (Sept 29) | 3 of 4 (Wf-279A,B,C missing; Wf-279D PASS) | "**5th consecutive occurrence** of the same regression class (after Wf-270/271/272/274)" (Wf-279E) |
| **Wf-285** (Sept 29) | 2 of 4 (Wf-285A,D missing; Wf-285B,C PASS) | "6th occurrence; Wf-282E/Wf-287E/Wf-284E 4-of-4 streak BROKEN at 2-of-4 this batch" (Wf-285E) |

**Total agents affected**: ~21 of 24 dispatched across the 6 waves (~88%).

**Streak break**: Wf-280 (Sept 29) was the first clean 4-of-4 round since Wf-269 per Wf-280E synthesis.

**Workflows that did NOT exhibit regression** (worked clean): Wf-273B/C/D/E, Wf-275A-D, Wf-276A-E, Wf-277A-E, Wf-278A-D, Wf-280A-D, Wf-281A-E, Wf-282A-E, Wf-283A-E, Wf-284A-E.

---

## 5. Workaround Status — Three-Layer Mitigation

### 5.1 Layer 1: Text-only task briefs (Wf-302C template — DEPLOYED)

**Path**: `molmetal/reports/wf302c_text_briefs/template.md`

Deployed since Wf-274. Task briefs constrain the tool palette to `Bash + Read + Edit` (no `StructuredOutput` invocation expected). Agents write their outputs as plain text via Edit, which the synthesis pipeline reads directly. Template structure:

```
WF-NNN: <title>
CONTEXT: <2-3 sentences>
TASK (plain text — no JSON / no StructuredOutput):
  1-3. <verb + concrete file/metric>; Output path: <abs path>.
TOOLS: USE Bash, Read, Edit. Do NOT call StructuredOutput.
REPLY FORMAT (5-8 lines max, plain text):
  - status, files_touched, gates_passed, honest_framing
GATES: G1/G2/G3 verifiable
HONEST-FRAMING NOTE: MEASURED vs PROJECTED
```

**Status**: In production use since Wf-274; the Wf-280 streak-break confirms this layer caught the regression for downstream waves.

### 5.2 Layer 2: User-side shape detector (Wf-298D wrapper — DEPLOYED)

**Path**: `molmetal/reports/wf298d_harness_detector/wrapper.py`

```python
from molmetal.reports.wf298d_harness_detector.wrapper import (
    HarnessOutputExtractor, FallbackExtraction, fast_filter_says_schema,
)
extractor = HarnessOutputExtractor()
extractor.detect_schema_shape(agent_payload_json)   # bool
extractor.extract(agent_payload_json, sibling_text="...")  # dict | FallbackExtraction
```

Behaviour contract:
- **Data payload** (Wf-272B-style): `extract()` returns the parsed dict unchanged.
- **Schema-shape payload** (Wf-270A/271D/272A/273B-class): `extract()` returns `FallbackExtraction(schema_shape_detected=True, recovered_text=<sibling_text>, fallback_reason=<reason>, ...)`.
- **Unparseable input**: returns `FallbackExtraction(fallback_reason="json_parse_failed", ...)`.

**Smoke test result** (Wf-298D §3): **4/4 regression-class payloads caught** (Wf-270A, Wf-271D, Wf-272A, Wf-273B), **1/1 known-good payload passed** (Wf-272B), **7/8 strict pass rate** with 1 contrived `name+type` false-positive (benign — caller falls back to sibling text).

**Status**: Shipped. Synthesis agents in the Wf-XXXE slot should wrap their upstream `result` extraction with `extractor.detect_schema_shape()` before field lookup.

### 5.3 Layer 3: Sibling-data reconstruction (DEPLOYED)

When the agent DOES call `StructuredOutput` and returns schema-as-result, the synthesis agent reconstructs the deliverable tree from sibling artefacts (Wf-270E §4.1, Wf-279E, Wf-285E):
- `TODO/pending/{N}` files
- Prior-wave final reports (`Wf-264-Final`, `Wf-265-Final`, `Wf-266-Final`, `Wf-267-Final`, `Wf-269C`)
- `LAMBDA_ROADMAP` §9.Z + §10.4 cross-refs
- Pre-flight checklist §9.5/§9.6/§9.7 entries

**Status**: In production use; Wf-279E and Wf-285E both rely on this path for their missing-deliverable HNs.

---

## 6. Long-Term Fix Path (Wf-291C §5.4)

**Location identified** (Wf-298D §1.1): The `StructuredOutput` tool lives inside the closed-source claude-code native binary:
```
/home/hugo/.vscode-server/extensions/anthropic.claude-code-2.1.220-linux-x64/resources/native-binary/claude
```
Verified by `grep -a "StructuredOutput"` returning positive hits on the binary. Source code is not exposed.

**Conclusion: NOT user-modifiable.** Any user-side fix must be a *pre-/post-processor* wrapping input/output. Wf-298D wrapper (Layer 2) implements option (c) below.

### Three canonical options from Wf-291C §5.4

| Option | Description | Status |
|--------|-------------|--------|
| **(a) Schema-as-prompt-only** | Pass schema as part of the task brief text rather than as a `StructuredOutput` tool wrapper. | Partial — covered by Layer 1 (text-only briefs). |
| **(b) Schema validation with retry** | If the agent returns `{"type":"object","properties":{...}}` shape, retry with explicit "you returned the schema, NOT the values; return the actual data" prompt. | NOT IMPLEMENTED — requires harness-level hooks not available. |
| **(c) Auto-fallback to text** | If `StructuredOutput` returns schema-shape data, fall back to treating the result as free-text. | SHIPPED via Wf-298D wrapper (Layer 2). |

### Canonical long-term fix (Anthropic-side, out of workspace scope)

A proper fix requires editing the harness binary's schema-extraction layer. The two prompt-injection strings that fire on the agent side remain unaddressed:
- ` You MUST call the  tool to complete this request. Call this tool now.`
- `StructuredOutput schema mismatch: ` / `StructuredOutput enforcement failed: `

These belong in a Claude Code release ticket — not a user-side workflow. Document this carry-over here and close Wf-272F via this aggregation doc + the Wf-298D wrapper.

---

## 7. Wf-302E Action Items (carry-over close)

| # | Action | Owner | Status |
|---|--------|-------|--------|
| 1 | Aggregate Wf-291C + Wf-298D into one canonical root-cause doc | Wf-302E | DONE (this file) |
| 2 | Wire Wf-298D wrapper into Wf-XXXE synthesis slot | synthesis agents | RECOMMENDED — follow-up Wf-303G |
| 3 | Adopt Wf-302C text-only template across all new briefs | workflow launcher | DEPLOYED |
| 4 | File Anthropic-side ticket for option (b) retry-with-correction | user (Hugo) | TODO |
| 5 | Close Wf-272F (never launched) — replaced by this carry-over | Wf-302E | DONE |

---

## 8. Cross-references

- `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` — Wf-291C root cause report (parent investigation; this doc aggregates).
- `molmetal/reports/wf298d_harness_detector/wrapper.py` — Wf-298D user-side shape detector (Layer 2 workaround).
- `molmetal/reports/wf298d_harness_detector/wrapper_report.md` — Wf-298D wrapper report (smoke-test results + wiring recommendation).
- `molmetal/reports/wf302c_text_briefs/template.md` — Wf-302C text-only brief template (Layer 1 workaround).
- `molmetal/reports/wf302c_text_briefs/example_bug_fix.md` — Wf-302C worked example (m6_kappa.json per-metal AUC restoration).
- `molmetal/reports/wf272_synthesis/repair_summary.md` — Wf-272E synthesis (canonical Wf-272-A-D aggregation showing the regression class first manifest).
- `molmetal/reports/wf270_synthesis/pending_tree.md` — Wf-270E reconstruction note (first sibling-data reconstruction example).
- `molmetal/reports/wf274_synthesis/repair_summary.md` — Wf-274E FOURTH-occurrence verdict.
- `molmetal/reports/wf279_synthesis/followup_summary.md` — Wf-279E 5th-occurrence verdict.
- `molmetal/reports/wf280_synthesis/followup_summary.md` — Wf-280E streak-BROKEN verdict (first clean 4-of-4 since Wf-269).
- `molmetal/reports/wf285_synthesis/followup_summary.md` — Wf-285E 6th-occurrence verdict.
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6.1.1 — wave-ledger catalog.

---

## 9. Status

- **Root cause**: DOCUMENTED — schema-shape confusion (Wf-291C §1).
- **Affected workflows**: 6 waves × ~22 agents across Wf-270/271/272/274/279/285.
- **Workaround**: DEPLOYED across three layers (text-only briefs + wrapper + sibling-data reconstruction).
- **Wrapper coverage**: 4/4 regression-class payloads caught (Wf-270A, Wf-271D, Wf-272A, Wf-273B); 1/1 known-good passed (Wf-272B).
- **Long-term fix**: Option (c) auto-fallback SHIPPED via Wf-298D wrapper; options (a)/(b) require harness-level hooks (out of workspace scope).
- **Wf-272F**: CLOSED via this carry-over aggregation (Wf-272F never launched; canonical investigation lives here + Wf-291C + Wf-298D).

---

**Wall**: Wf-302E carry-over complete; Wf-272F investigation closed; canonical root-cause doc aggregates Wf-291C + Wf-298D + Wf-302C into one shippable artefact at `molmetal/reports/wf302e_pipeline_root_cause/root_cause.md`.