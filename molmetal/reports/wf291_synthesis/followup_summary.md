# Wf-291E — Synthesis: Wf-291A/B/C/D outcomes + critical remaining

**Date**: 2026-09-29
**Wf**: Wf-291E (aggregate Wf-291A / B / C / D outcomes)
**Owner**: Hugo Lee + workflow harness
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §6 honest-negative
**Mode**: Read-only aggregation; no source files / launchers / configs modified.

---

## §0 TL;DR

| Gate | Value | Verdict |
|---|---|---|
| **G1_n_upstream_agents** | 4 dispatched (Wf-291A / B / C / D) | — |
| **G2_n_with_on_disk_artefact** | **1** (Wf-291C ✅ only) | ⚠️ 1/4 dedicated reports; Wf-291A/B/D no dedicated report directories |
| **G3_n_failed** | **0 explicit crashes** (1 missing-deliverable + 1 inline-cross-ref only + 1 dual-side land-without-report) | ⚠️ classified per §1 |
| **G4_n_critical_honest_negatives_remaining** | **3 items** (see §3) | ⚠️ 3 HNs open |
| **G5_n_next_priorities** | 5 items (see §4) | — |
| **G6_report_path** | `molmetal/reports/wf291_synthesis/followup_summary.md` | PASS — this file |

**Honest framing (party-twenty-spirit §3.5)**: Only Wf-291C produced a dedicated on-disk artefact (`molmetal/reports/wf291c_pipeline_root_cause/root_cause.md`). Wf-291A is **effectively COMPLETED-via-inline-re-verification** (the 207 v2 §9.AH carries a "Wf-291A re-verification" note dated 2026-09-29 confirming §9.AH values are correct; no value replacement required). Wf-291D is **PARTIAL** (both LAMBDA_ROADMAP §9.AF cross-ref AND `wave_budget_tracker.py --cfm-yuelbond-required` flag landed, but no synthesis report directory). Wf-291B is **MISSING-DELIVERABLE** (no on-disk artefact; the stale-read guard retry did not produce a dedicated report). This matches the Wf-291C root-cause prediction of "expect 1-2 missing-deliverable HNs" (per Wf-291C §7.1).

---

## §1 Per-Agent Summary

### §1.1 Wf-291A — §9.AH redo (re-verification) — **COMPLETED-via-INLINE-RE-VERIFICATION**

**Expected artefact**: `molmetal/reports/wf291a_section9_ah_redo/` or similar
**Observed**: No dedicated directory on disk; instead the Wf-291A work is captured as an **inline re-verification note** in `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AH (2026-09-29):
> "Wf-291A re-verification (2026-09-29): §9.AH values re-checked verbatim against `wf269c_report.json` fields `G7_decode_ratio_no_wasserstein=1.000` + `G8_decode_ratio_no_closure_bounded=1.000` + `honest_framing_note`. Wf-285A attempt log MISSING on disk. §9.AH already carries the Wf-269C spec — no value replacement required (pre-Wf-284D wrong values no_w=0.987/no_cb=0.992 are NOT present in §9.AH)."

**Verdict**: ✅ COMPLETED-via-re-verification — the original Wf-285A MISSING-DELIVERABLE was for a §9.AH value-replacement that turned out NOT to be needed (the wrong values were in §9.AF, per Wf-277A PENDING §9.AI.6). Wf-291A's re-verification closed the §9.AH gap without requiring a §9.AH edit.

### §1.2 Wf-291B — Stale-read guard retry — **MISSING-DELIVERABLE**

**Expected artefact**: `molmetal/reports/wf291b_stale_read_guard/` or similar
**Observed**: directory does not exist on disk (verified via `find /home/hugo/codes/mol-metal -path "*wf291b*"` — only `wf291c_pipeline_root_cause/` present). No `read_anchor_json(path, max_age_sec=..., required_version=...)` helper landed for `m14_anchor.json` / `cfm_decode_real.json` / `per_metal_pass.json` consumers.
**Cross-ref**: Per Wf-291C §7.4 the stale-read guard retry was queued as "Wf-291H m14 stale-read guard (~45 min CPU)" but the work was not dispatched under the Wf-291B agent name. The Wf-271B + Wf-287A m14 anchor re-use risk pattern remains UNRESOLVED.
**Verdict**: ❌ MISSING-DELIVERABLE — agent did not deposit an on-disk artefact; re-dispatch required.

### §1.3 Wf-291C — Pipeline root-cause investigation — **MEASURED PASS**

**File on disk**: `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` (≈209 lines)
**Findings** (verbatim from §1 + §4):
- **Root cause identified**: When workflow agents are tasked with returning **complex structured JSON** via `StructuredOutput`, the agents return the JSON schema definition itself rather than the actual values. Parent pipeline then tries to extract fields and receives `undefined` for all of them.
- **n_affected_workflows**: **6 workflow waves** (Wf-270, Wf-271, Wf-272, Wf-274, Wf-279, Wf-285) across **~22 individual agents** (~88% of dispatched agents in those waves).
- **Mitigation**: Tool-palette constraint (`Bash + Read` or `Read + Edit` only) + sibling-data reconstruction + auto-fallback-to-text (long-term, NOT yet implemented).
- **Streak break**: Wf-280 (Sept 29) was the first clean 4-of-4 round since Wf-269 per Wf-280E; Wf-282 + Wf-284 maintained the clean streak.
**Verdict**: ✅ MEASURED PASS — root cause fully documented with 5 verbatim synthesis excerpts + 6 affected workflow waves + 22 affected agents + 3 long-term mitigations + recommended Wf-291F (auto-fallback-to-text wrapper).

### §1.4 Wf-291D — LAMBDA + YuelBond flag — **PARTIAL (dual-side land, no synthesis report)**

**Expected artefact**: `molmetal/reports/wf291d_lambda_yuelbond_flag/` or similar
**Observed**: No dedicated directory; but **both halves of Wf-291D landed on disk**:
1. **Documentation side**: `TODO/03_plans/LAMBDA_ROADMAP.md` §9.AF carries the "Wf-291D update note" entry (line ~607) dated 2026-09-29, encoding the YuelBond-conditionality cross-ref + the "DO cite (post-Wf-291D)" / "DO NOT cite (until Wf-277A + Wf-291G ship)" split.
2. **Enforcement side**: `molmetal/scripts/wave_budget_tracker.py` carries the `--cfm-yuelbond-required` CLI flag (file mtime 2026-09-29 00:48, with `cfm_yuelbond_required: bool = False` default + the gate logic that blocks CFM waves when `--no-yuelbond` is set AND `cfm_yuelbond_required=True`).

**Verdict**: ⚠️ PARTIAL — both substantive deliverables landed but no synthesis report was written; the Wf-291D agent failed to deposit a dedicated report under `molmetal/reports/wf291d_*/` documenting the dual-side land. The YuelBond-conditionality contract is now enforced (per `wave_budget_tracker.py`) AND documented (per LAMBDA_ROADMAP §9.AF); the gap is purely in the report-trajectory.

---

## §2 Aggregate counts

- **n_completed** (substantive work landed + documented): **2** — Wf-291A (inline re-verify in 207 v2 §9.AH), Wf-291C (dedicated 209-line root-cause report)
- **n_partial** (substantive work landed, no dedicated report): **1** — Wf-291D (LAMBDA_ROADMAP §9.AF + wave_budget_tracker.py flag both shipped; no `wf291d_*/` directory)
- **n_missing_deliverable** (no on-disk artefact, no recovery path found): **1** — Wf-291B (stale-read guard retry)
- **n_failed** (explicit crash, log evidence of error): **0**

**Substantive deliverables landed**: 4 of 4 (Wf-291A inline verify, Wf-291B missing, Wf-291C report, Wf-291D dual-side land) — only Wf-291B is genuinely outstanding at the artefact level.

---

## §3 Critical remaining (orthogonal to this Wf)

1. **Wf-291B stale-read guard retry MISSING** — `read_anchor_json(path, max_age_sec=..., required_version=...)` helper not shipped. The Wf-271B + Wf-287A m14 anchor re-use risk pattern remains open. Per Wf-291C §7.4 the follow-up Wf-291H is queued (~45 min, CPU-only) but the Wf-291B dispatch name was not used.

2. **Wf-291D synthesis report MISSING** — both substantive deliverables (LAMBDA_ROADMAP §9.AF + wave_budget_tracker.py flag) shipped, but no `molmetal/reports/wf291d_lambda_yuelbond_flag/` report directory was deposited. Recovery: write the synthesis report retrospectively (~15 min CPU) citing the two on-disk artefacts.

3. **Wf-291F Pipeline-harness root-cause fix** (~4 hr, NOT parallelizable per Wf-280E §P2.2) — the highest-ROI long-term fix is the **(c) auto-fallback-to-text** wrapper that detects `{"type":"object","properties":{...}}` shape and falls back to raw text. This closes the Wf-270/271/272/274/279/285 6-occurrence regression class for good.

---

## §4 Next priorities

1. **Wf-291F Pipeline-harness auto-fallback-to-text wrapper** (~4 hr CPU, NOT parallelizable per Wf-280E §P2.2) — top priority because it closes the 6-occurrence pipeline-regression class identified in Wf-291C §1. Implementation: wrap `StructuredOutput` extraction with a `{"type":"object","properties":{...}}` shape detector and fall back to raw text.

2. **Wf-291G Re-emit Wf-285A/D from sibling data** (~30 min CPU, parallelizable per Wf-280E §P2.1) — reconstruct the 2 missing Wf-285 deliverables (Wf-285A §9.AH redo + Wf-285D stale-read guard) from sibling data: Wf-285A's correction is already proven unnecessary per Wf-291A inline re-verify, so Wf-285A's report can simply state "no value replacement required"; Wf-285D's stale-read guard is the same scope as Wf-291B (the original dispatch name) — combine Wf-291G + Wf-291B into one work item.

3. **Write retrospective Wf-291D synthesis report** (~15 min CPU) — file at `molmetal/reports/wf291d_lambda_yuelbond_flag/flag_report.md` citing LAMBDA_ROADMAP §9.AF + `wave_budget_tracker.py` `--cfm-yuelbond-required` flag. Closes the report-trajectory gap with no source code change.

4. **Wf-277A §9.AF source-value mismatch fix** (⏳ PENDING per 207 v2 §9.AI.6, ≤45 min CPU) — edit §9.AF to align verbatim deltas (full=1.000, no_yb=0.000 Δ=−1.000; no_w=0.987 Δ=−0.013; no_cb=0.992 Δ=−0.008) per `wf269c_real_ablation/per_lever.json`. Unblocks the "DO NOT cite" LAMBDA_ROADMAP §9.AF entry.

5. **Defer P3 GPU work**: TODO-180 G4 (14 PENDING pairs, 1.8 hr RTX 5090), TODO-181 G5 (15 PENDING pairs, 7.5 hr Blackwell), TODO-182 G6 (60-run ablation, 7.5 hr Blackwell) to R20+ GPU slot per LAMBDA_ROADMAP §10.4.

---

## §5 Files modified by the Wf-291 wave (no synthesis-side edits)

- `TODO/03_plans/LAMBDA_ROADMAP.md` (Wf-291D documentation side — §9.AF cross-ref + Wf-291D update note)
- `molmetal/scripts/wave_budget_tracker.py` (Wf-291D enforcement side — `--cfm-yuelbond-required` flag)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` (Wf-291A inline re-verification note in §9.AH)
- `molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` (Wf-291C dedicated report, 209 lines)

No new source files under `molmetal/molmetal_lam/` or `molmetal/scripts/` were touched by Wf-291E (synthesis is read-only).

---

## §6 Report paths (4 upstream + 1 synthesis)

- Wf-291A: NO on-disk artefact — substantive work captured inline in `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AH "Wf-291A re-verification" note
- Wf-291B: NO on-disk artefact — MISSING-DELIVERABLE
- Wf-291C: `/home/hugo/codes/mol-metal/molmetal/reports/wf291c_pipeline_root_cause/root_cause.md` (209 lines)
- Wf-291D: NO dedicated report — substantive work captured inline in `TODO/03_plans/LAMBDA_ROADMAP.md` §9.AF "Wf-291D update note" + `molmetal/scripts/wave_budget_tracker.py` `--cfm-yuelbond-required` flag
- Wf-291E: `/home/hugo/codes/mol-metal/molmetal/reports/wf291_synthesis/followup_summary.md` (this file)

---

## §7 Honest framing (party-twenty-spirit §3.5)

- Wf-291C is the only agent to deposit a dedicated on-disk artefact (the 209-line root-cause investigation).
- Wf-291A and Wf-291D's substantive work landed via inline edits to existing artefacts (207 v2 §9.AH re-verify note + LAMBDA_ROADMAP §9.AF cross-ref + wave_budget_tracker.py flag), NOT via dedicated Wf-291A*/Wf-291D* directories.
- Wf-291B is genuinely MISSING — no on-disk artefact and no recovery path was found in this synthesis pass.
- The aggregate substantive-deliverable rate is **3 of 4** (Wf-291A inline, Wf-291C report, Wf-291D dual-side land) with Wf-291B outstanding.
- The aggregate dedicated-report rate is **1 of 4** (Wf-291C only) — a regression-class outcome consistent with the Wf-291C root-cause prediction of "expect 1-2 missing-deliverable HNs" (per §7.1).
- The pipeline-regression class is fully diagnosed but NOT yet patched; Wf-291F remains the P0 follow-up to close the 6-occurrence streak permanently.