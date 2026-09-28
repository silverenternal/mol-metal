# TODO-116 — Wf-74 v2 close-out (HONEST meta-close-out)

**Date**: 2026-09-27
**Workflow**: Wf-74 v2 close-out rewrite (silent-overstatement fix per `party-twenty-spirit` §3.5)
**Status**: ✅ **SHIP — honest meta-close-out; Wf-74 v2 already shipped via `TODO/completed/97c_wf74_reclassification.md`**
**Priority**: **P0 → RESOLVED** (single P0 audit gap closed; per INDEX §17.5 row 3)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan) + `TODO/completed/97b_wetlab_prework_docs.md` (the OVERSTATED v1 — corrected) + `TODO/completed/97c_wf74_reclassification.md` (the AUTHORITATIVE v2 ship)
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 + 不掩盖)

---

## §0 TL;DR (per party-twenty-spirit §3.5)

| Question | Answer |
|---|---|
| Was Wf-74 v2 supposed to run? | **YES** — it was the silent-overstatement fix on the overstated Wf-74 v1 close-out (`97b_wetlab_prework_docs.md`) |
| Did Wf-74 v2 run? | **YES — 2026-09-27** — shipped via `97c_wf74_reclassification.md` |
| Did TODO-116 itself run? | **YES — 2026-09-27** — this doc IS the close-out |
| Is the P0 §3.5 violation resolved? | **YES** — INDEX §17.5 row 3 closed; the audit gap that flagged TODO-116 as the "唯一 P0 未 ship" no longer holds |
| What is open (cannot be Claude-shipped)? | 0 LOI sent / 0 MoU / 0 PI reply / 0 cell culture / 0 IC50 MEASURED — Hugo Lee action (outreach dispatch) |

**One-line verdict (per `party-twenty-spirit` §3.5)**: TODO-116's job was to **flag the Wf-74 v1 close-out overstatement and rewrite the close-out as v2**. Both subtasks ship on disk (`97c` + `116`). The only thing NOT shipped is the actual wet-lab work itself — that requires Hugo Lee outreach + PI reply + 6-month timeline, which is NOT a Claude-task per `TODO/pending/84_no_paper_task_without_authorization.md` §1.2 (outreach = CODE-ONLY-ish, but actual email send requires human coordination).

---

## §1 What was supposed to happen (Wf-74 v2 plan)

Per `TODO/pending/116_wf74_wetlab_outreach_partial.md` §2 + §3:

| Phase | Wall | Output | Status |
|---|---|---|---|
| **P1.1** Audit Wf-74 agent JSONLs + author v2 close-out | 2 hr CPU | `TODO/completed/97c_wf74_reclassification.md` | ✅ **SHIP 2026-09-27** |
| **P1.2** Write 3 LOI drafts | 4 hr CPU | `TODO/pending/wetlab_loi_{ornl_protein_metal,pku_cisplatin_ic50,xiehe_nci60}.md` | ✅ **SHIP 2026-09-27** |
| **P1.3** Write 3 institutional SOP templates | 2 hr CPU | `TODO/pending/wetlab_sop_{cytotoxicity_assay,binding_affinity_assay,cellular_uptake_assay}.md` | ✅ **SHIP 2026-09-27** |
| **P1.4** Outreach dispatch (cold-email 3 PIs) | Hugo Lee action | 3 cold-emails sent | 🚫 **NOT SHIPPED — human action; 0 sent** |

All 3 CPU phases (P1.1 + P1.2 + P1.3) ship on disk today. P1.4 is a Hugo Lee action — by design NOT a Claude task per TODO-116 §2 P1.4 + INDEX §12.3 P3 mitigation.

---

## §2 Why Wf-74 v2 was needed (the §3.5 violation)

Wf-74 v1 close-out (`TODO/completed/97b_wetlab_prework_docs.md`) **silently overstated 6 deliverables** that did not exist on disk:

- ❌ 3 LOI drafts claimed "SHIP" but never written (only Wf-28 generic outreach template existed)
- ❌ 3 institutional-tailored SOPs claimed "SHIP" but were actually Wf-28 generic carry-overs from `molmetal/docs/wetlab_protocols/`

This was a **§3.5 silent-overstatement violation** (party-twenty-spirit: "0 容忍 + 不掩盖"). v1 tabulated Wf-28 carry as Wf-74 SHIP and credited LOI drafts that were never authored.

**The v1 close-out was patched in place** with a correction header (lines 1-19) that cross-links to the v2 authoritative doc at `97c_wf74_reclassification.md`. The v1 doc is preserved as historical record; **v2 is authoritative**.

---

## §3 What we have instead (the on-disk reality, MEASURED-not-PROJECTED)

### §3.1 On-disk SHIP today (Wf-74 v2 artifacts)

| # | File | Lines | Status | Source |
|---:|---|---:|---|---|
| 1 | `TODO/pending/wetlab_ic50_analysis_pipeline.md` | 346 | ✅ SHIP (pre-existed Wf-74 v2) | 97c §1.1 |
| 2 | `molmetal/scripts/template_ic50_pipeline.py` | 653 LOC | ✅ SHIP (pre-existed Wf-74 v2) | 97c §1.1 |
| 3 | `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` | 222 | ✅ SHIP (pre-existed Wf-74 v2) | 97c §1.1 |
| 4 | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | NEW | ✅ SHIP (Wf-74 v2 NEW) | 97c §1.2 |
| 5 | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | NEW | ✅ SHIP (Wf-74 v2 NEW) | 97c §1.2 |
| 6 | `TODO/pending/wetlab_loi_xiehe_nci60.md` | NEW | ✅ SHIP (Wf-74 v2 NEW) | 97c §1.2 |
| 7 | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | NEW | ✅ SHIP (Wf-74 v2 NEW) | 97c §1.3 |
| 8 | `TODO/pending/wetlab_sop_binding_affinity_assay.md` | NEW | ✅ SHIP (Wf-74 v2 NEW) | 97c §1.3 |
| 9 | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | NEW | ✅ SHIP (Wf-74 v2 NEW) | 97c §1.3 |
| 10 | `TODO/completed/97c_wf74_reclassification.md` | 167 | ✅ SHIP (Wf-74 v2 authoritative v2 close-out) | 97c §0 |
| 11 | `TODO/completed/97b_wetlab_prework_docs.md` | patched | ✅ SHIP (v1 + correction header cross-link to 97c) | 97c §3 |
| 12 | `TODO/completed/116_wf74_v2_closeout.md` (THIS FILE) | NEW | ✅ SHIP (TODO-116 meta-close-out) | this doc |

**Total: 12 Wf-74 v2 artifacts on disk, 11 SHIP today (excluding this doc)**.

### §3.2 Carry from Wf-28 (pre-Wf-74, NOT Wf-74 deliverables)

Per 97c §1.4 — these were **explicitly misattributed** in v1 close-out as Wf-74 SHIP:

- 5 generic SOPs at `molmetal/docs/wetlab_protocols/` (cytotoxicity, binding, uptake, data-sharing, equipment requirements)
- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (199-line generic email template, NOT institution-specific LOI)

**v2 close-out (97c) correctly attributes these as Wf-28 carry** — they pre-existed Wf-74 and are NOT part of the v2 SHIP surface.

### §3.3 PROJECTED (NOT YET MEASURED — wet-lab dependent)

| # | Item | Why PROJECTED | When MEASURED |
|---:|---|---|---|
| 1 | 0 → ≥1 LOI sent | Hugo Lee coordination; 3 drafts ready, 0 dispatched | R20-P6 S1 (after user decision) |
| 2 | 0 → 1 MoU signed | Depends on PI reply + counter-sign | R20-P6 S1 (≥1 mo after cold-email) |
| 3 | 0 → wet-lab compound synthesized | PI + 1-month synthesis timeline | R20-P6 S2 (~3 mo) |
| 4 | 0 → cell culture + IC50 panel measured | PI + 3-mo culture + 2-mo assay | R20-P6 S3 (~6 mo) |
| 5 | 0 → in-silico vs wet-lab Pearson r ≥ 0.7 | Cannot MEASURE until P6.2 wet-lab IC50 ships | Q1 2027 earliest |
| 6 | 0 → paper §5 wet-lab validation | TODO-84 paper rule + MEASURED wet-lab | Post-authorization only |

**None of the 6 PROJECTED items is a Claude task.** They depend on PI reply + 6-month wet-lab timeline. The §3.5 honest framing requires we report them as PROJECTED, not MEASURED.

---

## §4 Honest framing (party-twenty-spirit §3.5)

- **No fabricated wet-lab numbers**: IC50 pipeline doc + JSON template carry `pic50=null`, `pearson_r=null`, `n=0`. Template script runs on synthetic fixture; emits `DEFERRED_PENDING_WETLAB`. Bench JSON shell exists with all `null` fields per Wf-58 dry-run.
- **Wet-lab gate NOT active**: `Pearson r ≥ 0.7` cannot MEASURE today. R20-P6 wet-lab cooperation is 100% dependent on PI reply.
- **0 LOI sent**: 3 institutional LOI drafts SHIP today; **0 cold-email / 0 MoU signed / 0 PI contacted**. The actual email *send* requires Hugo's coordination per `TODO/INDEX.md` §12.3 P3 mitigation.
- **SOPs are templates, not validated protocols**: NEW Wf-74 SOPs adapt Wf-28 generic protocols to institutional flavor; **NO wet-lab hands validation, NO institutional QC, NO biological replicate measured**.
- **Cross-metal transfer caveat**: top-20 spans Pt (n=2) / Pd (n=3) / Au (n=1) / Ru (n=3) / Ir (n=11). D-MPNN trains on PlatinAI 4-cell (Pt-only); non-Pt predictions carry distribution-shift — `wetlab_ic50_analysis_pipeline.md` §3.3 explicitly flags this.
- **R20-P6 is a 6-month timeline** (1 mo synthesis + 3 mo culture + 2 mo assay + 1 mo analysis). Wet-lab P6.2 cannot MEASURE before **Q1 2027** at earliest.
- **TODO-116 P0 close-out was docs-only**: this entire Wf-74 v2 was a §3.5 honest-framing fix, not new code or new measurements. Closing this without metric lift is the §3.5 honest pattern.

---

## §5 TODO-116 status reconciliation

| Audit finding (INDEX §17.5 row 3, 2026-09-27 19:20 UTC) | Status as of 2026-09-27 (post-this-doc) |
|---|---|
| "Wf-74 v2 close-out doc 未 ship" | ✅ **RESOLVED** — `97c_wf74_reclassification.md` SHIP 2026-09-27 |
| "唯一 P0 未 ship" | ✅ **RESOLVED** — INDEX §17.5 row 3 closes (this doc is the meta-close-out) |
| "party-twenty-spirit §3.5 violation pending fix" | ✅ **RESOLVED** — 97c §4 + this doc §4 ship the honest framing |
| "Wf-132 MUST enqueue a follow-up Wf-133 to close Wf-74 v2" | ✅ **RESOLVED** — Wf-133 P1-P4 closed 7 close-outs including this lineage; see `molmetal/reports/wf133_orphan_closeout/synthesis.md` |

**Net §17.5 row 3 verdict**: from "OPEN as of 2026-09-27 — 唯一 P0 未 ship" → **✅ CLOSED 2026-09-27**.

The remaining 2 P0 (TODO-119 m14 decode_ratio + TODO-110 m3 Vina cisplatin full 8 d-block) are unchanged — TODO-119 is in-flight on Wf-124, TODO-110 partial-lift ship (cisplatin × 10 pockets MEASURED) per `TODO/completed/121_meko_metal_atom_extension_closeout.md`.

---

## §6 R20/R21 deferred plan (what comes next)

Per `TODO/pending/r20_p1_p2_p3_p4_p5_p6_plan.md` (R20-P6 = wet-lab cooperation):

| Round | What | Owner | Status |
|---|---|---|---|
| **R20-P6 S1** | Cold-email 3 PIs (ORNL/PKU/协和) with the 3 LOI drafts at `TODO/pending/wetlab_loi_*.md` | Hugo Lee (human action) | ⏸️ **DEFERRED until user decision** |
| **R20-P6 S2** | Wet-lab compound synthesis (≥6 compounds for top-20 shortlist) | Wet-lab PI | ⏸️ **BLOCKED on PI reply** |
| **R20-P6 S3** | 9-cell × 3-assay × n=3 IC50 panel + ICP-MS uptake | Wet-lab PI | ⏸️ **BLOCKED on synthesis** |
| **R21-P1** | In-silico vs wet-lab agreement analysis (uses `template_ic50_pipeline.py`) | ultracode agent | ⏸️ **BLOCKED on wet-lab data** |
| **R21-P2+** | Paper §5 wet-lab validation (TODO-84 gated) | ultracode agent | 🚫 **BLOCKED on paper-rule authorization** |

**No additional Claude work** is queued between today and the first PI reply. The 3 LOI drafts + 3 SOPs are SHIP and ready for Hugo Lee dispatch whenever the user decides to launch wet-lab cooperation.

---

## §7 Cross-references

- `TODO/pending/116_wf74_wetlab_outreach_partial.md` (the audit that flagged the v1 overstatement)
- `TODO/completed/97b_wetlab_prework_docs.md` (the OVERSTATED v1 — patched with correction header)
- `TODO/completed/97c_wf74_reclassification.md` (the AUTHORITATIVE v2 close-out — 167 lines)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan parent)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 P6.0+P6.1+P6.2 in-silico layer ship)
- `TODO/pending/84_no_paper_task_without_authorization.md` (TODO-84 paper rule — outreach NOT gated)
- `TODO/completed/81_wetlab_data_prep.md` (Wf-28 wet-lab data tier — 5 generic SOPs + outreach template)
- `TODO/completed/82_metric_gap_filling.md` §5 (R20-P1..P7 plan)
- `TODO/INDEX.md` §12.3 P3 mitigation (outreach dispatch instructions)
- `TODO/INDEX.md` §17.5 row 3 (audit gap that TODO-116 closed)
- `molmetal/reports/wf133_orphan_closeout/synthesis.md` (umbrella synthesis for Wf-133 P4 close-out wave)
- `party-twenty-spirit` §3.5 (0 容忍 + 不掩盖)
- `TODO/pending/r20_p1_p2_p3_p4_p5_p6_plan.md` (R20-P6 deferred plan)

---

## §8 Final verdict

**Date**: 2026-09-27 · **TODO-116 status**: ✅ **SHIP — meta-close-out**; Wf-74 v2 SHIP via 97c; P0 §3.5 violation resolved; INDEX §17.5 row 3 closes; outreach dispatch DEFERRED to Hugo Lee action; 6-mo wet-lab timeline DEFERRED to R20-P6 S1-S3; no additional Claude work queued · **Co-Authored-By**: Claude Code <noreply@anthropic.com>