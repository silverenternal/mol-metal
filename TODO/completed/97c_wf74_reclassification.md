# TODO-97c — Wf-74 wet-lab outreach honest v2 reclassification

**Date**: 2026-09-27
**Workflow**: Wf-74 R20-P6 wet-lab pre-work **honest reclassification** (corrects v1 close-out overstatement)
**Status**: 🚀 **Active — accurate inventory + 3 LOI drafts + 3 SOP templates authored; 0 outreach sent, 0 wet-lab hands**
**Priority**: **P0 CRITICAL** (silent close-out overstatement, paper-rule-adjacent)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan) + `TODO/pending/116_wf74_wetlab_outreach_partial.md` (this audit) + `TODO/completed/97b_wetlab_prework_docs.md` (the OVERSTATED v1 — corrected by this doc)
**Skill alignment**: `mol-metal-wetlab-protocol` (R20 S6 from Wf-62) + `party-twenty-spirit` §3.5

---

## §0 TL;DR

| Item | File | Status |
|---|---|---|
| **IC50 analysis pipeline doc** (Wf-74 P3) | `TODO/pending/wetlab_ic50_analysis_pipeline.md` (346 lines) | ✅ **SHIP** (real, on disk) |
| **Companion IC50 template script** | `molmetal/scripts/template_ic50_pipeline.py` (28 KB, 653 LOC) | ✅ **SHIP** (real, on disk) |
| **Per-compound JSON template** | `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` | ✅ **SHIP** (real, on disk) |
| **LOI-ORNL** (protein-metal binding) | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | ✅ **NEW** (this Wf — ready to send, **0 sent**) |
| **LOI-PKU** (cisplatin IC50 curves) | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | ✅ **NEW** (this Wf — ready to send, **0 sent**) |
| **LOI-协和** (NCI-60 panel) | `TODO/pending/wetlab_loi_xiehe_nci60.md` | ✅ **NEW** (this Wf — ready to send, **0 sent**) |
| **SOP-1 Cytotoxicity** (institutional-tailored) | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | ✅ **NEW** (this Wf — template) |
| **SOP-2 Binding affinity** | `TODO/pending/wetlab_sop_binding_affinity_assay.md` | ✅ **NEW** (this Wf — template) |
| **SOP-3 Cellular uptake** | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | ✅ **NEW** (this Wf — template) |
| **Wf-28 generic protocols** (carry) | `molmetal/docs/wetlab_protocols/{cytotoxicity,binding_affinity,cellular_uptake}_assay.md` (5 protocols) | ✅ CARRY (existed pre-Wf-74) |
| **Paper §5 wet-lab validation** | `paper/sections/05_wetlab_validation.tex` (NOT created) | 🚫 **DEFERRED — TODO-84** |
| **0 LOI sent / 0 MoU / 0 PI reply / 0 cell culture / 0 IC50 MEASURED** | — | 🚫 **NOT STARTED** |

**One-line verdict (per party-twenty-spirit §3.5)**: Wf-74 wet-lab pre-work now ships the **3 IC50 pipeline artifacts + 3 LOI drafts + 3 institutional-tailored SOP templates** on disk, but the **wet-lab gate is NOT active** (0 LOI sent / 0 MoU signed / 0 cell culture started / 0 IC50 MEASURED). The headline `Pearson r ≥ 0.7` cannot MEASURE until ≥1 PI replies + MoU + Phase 1 launches + 6-month timeline completes.

---

## §1 Per-deliverable (accurate inventory)

### §1.1 IC50 analysis pipeline artifacts (3 SHIP — pre-existed before this Wf)

- `TODO/pending/wetlab_ic50_analysis_pipeline.md` — 346 lines, 6 sections (raw data norm → curve fit → pIC50+95%CI → ANOVA → in-silico vs wet-lab agreement → next actions).
- `molmetal/scripts/template_ic50_pipeline.py` — 653 LOC, pure NumPy/SciPy, runs on synthetic fixture only.
- `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` — schema-only template; `pic50=null`, `pearson_r=null`, `qc_flags=["PENDING_WETLAB"]`.

### §1.2 3 LOI drafts (NEW this Wf, ready to send, **0 sent**)

1. `TODO/pending/wetlab_loi_ornl_protein_metal.md` — ORNL protein-metal binding (SPR/ITC + AMBER compute).
2. `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` — PKU med-chem cisplatin IC50 curves (MCF7/A549/A2780/A2780cis + NCI-60).
3. `TODO/pending/wetlab_loi_xiehe_nci60.md` — 协和 hospital NCI-60 9-cell panel + ICP-MS uptake.

### §1.3 3 institutional-tailored SOP templates (NEW this Wf)

1. `TODO/pending/wetlab_sop_cytotoxicity_assay.md` — SRB/MTT/CTG (institutional flavor; references Wf-28 generic protocol).
2. `TODO/pending/wetlab_sop_binding_affinity_assay.md` — SPR/ITC binding (institutional flavor; references Wf-28 generic protocol).
3. `TODO/pending/wetlab_sop_cellular_uptake_assay.md` — ICP-MS uptake (institutional flavor; references Wf-28 generic protocol).

### §1.4 Carry from Wf-28 (existed pre-Wf-74)

- 5 generic SOPs at `molmetal/docs/wetlab_protocols/`: cytotoxicity, binding, uptake, data-sharing, equipment requirements. Plus `collaborator_outreach_template.md` (WLP-OUT-001, 199 lines, generic email template).
- **These 5 are Wf-28 carry, NOT Wf-74 deliverables**. The v1 close-out incorrectly conflated Wf-28 carry with Wf-74 SHIP.

---

## §2 MEASURED vs PROJECTED (corrected)

Per party-twenty-spirit §3.5 ("0 容忍 + 不掩盖"):

### §2.1 MEASURED (today, on disk)

| Item | File | Lines / LOC | Status |
|---|---|---|---|
| IC50 pipeline doc | `TODO/pending/wetlab_ic50_analysis_pipeline.md` | 346 lines | ✅ SHIP |
| IC50 template script | `molmetal/scripts/template_ic50_pipeline.py` | 653 LOC | ✅ SHIP |
| IC50 JSON template | `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` | 222 lines | ✅ SHIP |
| LOI-ORNL draft | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | NEW | ✅ SHIP |
| LOI-PKU draft | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | NEW | ✅ SHIP |
| LOI-协和 draft | `TODO/pending/wetlab_loi_xiehe_nci60.md` | NEW | ✅ SHIP |
| SOP-cytotoxicity | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | NEW | ✅ SHIP |
| SOP-binding | `TODO/pending/wetlab_sop_binding_affinity_assay.md` | NEW | ✅ SHIP |
| SOP-uptake | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | NEW | ✅ SHIP |
| Wf-28 generic SOPs (carry) | `molmetal/docs/wetlab_protocols/*.md` | 5 files | ✅ CARRY (not Wf-74) |
| Wf-28 outreach template (carry) | `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` | 199 lines | ✅ CARRY (not Wf-74) |

### §2.2 PROJECTED (NOT YET MEASURED — wet-lab dependent)

| Item | Why PROJECTED |
|---|---|
| **0 → ≥1 LOI sent** | Hugo Lee coordination required; 3 drafts ready, none dispatched |
| **0 → 1 MoU signed** | Depends on first PI reply + counter-sign |
| **0 → wet-lab compound synthesized** | Depends on PI + 1-month synthesis timeline |
| **0 → cell culture + IC50 panel measured** | Depends on PI + 3-month cell culture + 2-month assay |
| **0 → in-silico vs wet-lab Pearson r ≥ 0.7** | Cannot MEASURE until P6.2 wet-lab IC50 ships (Q1 2027 earliest) |
| **0 → paper §5 wet-lab validation** | TODO-84 paper rule; needs user authorization + MEASURED wet-lab |

---

## §3 Before vs After (v1 overstated → v2 honest)

| Dimension | **v1 (`97b_wetlab_prework_docs.md`) claim** | **v2 reality on disk (this doc)** |
|---|---|---|
| LOI drafts | 3 SHIP claimed | 3 NOT WRITTEN at v1 → 3 NEW this Wf (drafts ready, 0 sent) |
| SOPs (institutional) | 3 SHIP claimed | 3 NOT WRITTEN at v1 → 3 NEW this Wf (templates) |
| IC50 pipeline doc | ✅ SHIP | ✅ SHIP (real, on disk) |
| Wet-lab IC50 MEASURED | ❌ NOT EXIST (gated) | ❌ NOT EXIST (still gated) |
| Paper §5 | 🚫 DEFERRED | 🚫 DEFERRED (TODO-84 unchanged) |
| 3-PI outreach status | ❌ NOT EXIST (0 sent) | ❌ NOT EXIST (3 drafts ready, 0 sent) |

**Net correction**: v1 close-out inflated the LOI + SOP deliverable count from **0 → 3 each**. The v1 doc tabulated the Wf-28 generic protocols as if they were Wf-74 institutional-tailored SOPs. **The v1 close-out also credited LOI drafts that were never authored**; the Wf-28 `collaborator_outreach_template.md` was the only outreach draft on disk (generic email template, not institution-specific LOIs). This v2 closes that gap by authoring 3 institutional LOIs + 3 institutional-tailored SOPs now.

---

## §4 Honest framing (party-twenty-spirit §3.5)

- **No fabricated wet-lab numbers**: IC50 pipeline doc + JSON template carry `pic50=null`, `pearson_r=null`, `n=0`. Template script runs on synthetic fixture; emits `DEFERRED_PENDING_WETLAB`.
- **Wet-lab gate NOT active**: `Pearson r ≥ 0.7` cannot MEASURE today. Bench JSON shell exists (`agreement_per_metal_cell` with all `null` fields) per Wf-58 dry-run.
- **0 LOI sent**: 3 institutional LOI drafts SHIP today; **0 cold-email / 0 MoU signed / 0 PI contacted**. R20-P6 wet-lab cooperation is **100% dependent on PI reply** (per `TODO/completed/81_wetlab_data_prep.md` §4.2).
- **SOPs are templates, not validated protocols**: NEW Wf-74 SOPs adapt Wf-28 generic protocols to institutional flavor; **NO wet-lab hands validation, NO institutional QC, NO biological replicate measured**.
- **Cross-metal transfer caveat**: top-20 spans Pt (n=2) / Pd (n=3) / Au (n=1) / Ru (n=3) / Ir (n=11). D-MPNN trains on PlatinAI 4-cell (Pt-only); non-Pt predictions carry distribution-shift — `wetlab_ic50_analysis_pipeline.md` §3.3 explicitly flags this.
- **R20-P6 is a 6-month timeline** (1 mo synthesis + 3 mo culture + 2 mo assay + 1 mo analysis). Wet-lab P6.2 cannot MEASURE before **Q1 2027** at earliest.
- **Outreach dispatch is a Hugo Lee action**, NOT a Claude task. Claude wrote the 3 LOI drafts + 3 SOPs; the actual email *send* requires Hugo's coordination per `TODO/INDEX.md` §12.3 P3 mitigation.

---

## §5 Paper-rule compliance (TODO-84)

Per `TODO/pending/84_no_paper_task_without_authorization.md`:

### §5.1 P6.4 paper §5 final claim — DEFERRED (unchanged)

- 🚫 `paper/sections/05_wetlab_validation.tex` NOT created.
- P6.4 depends on P6.2 wet-lab IC50 MEASURED, which depends on PI outreach + 6-month timeline.
- Writing §5 needs explicit user OK per TODO-84 §5 template.

### §5.2 Allowed without authorization (this Wf-74 v2 ship)

| File | Status | Category |
|---|---|---|
| 3 IC50 pipeline artifacts (§1.1) | ✅ SHIP | TODO doc + analysis script + JSON template |
| 3 LOI drafts (§1.2) | ✅ SHIP | outreach draft (NOT paper) |
| 3 institutional SOP templates (§1.3) | ✅ SHIP | protocol template (NOT paper) |
| `TODO/completed/97c_wf74_reclassification.md` (this file) | ✅ SHIP | TODO doc (NOT paper) |
| `TODO/completed/97b_wetlab_prework_docs.md` (PATCHED header note cross-linking to 97c) | ✅ SHIP (patched) | TODO doc (NOT paper) |

**🚫 NOT created / NOT modified**:
- `paper/sections/05_wetlab_validation.tex`, `paper/main.tex`, `paper/main.pdf`, rebuttal, supplementary, appendix.
- Any LOI-email send (0 cold-email at this revision).

---

## §6 Cross-references

- `TODO/pending/116_wf74_wetlab_outreach_partial.md` (parent audit)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 overstated close-out — PATCHED with header note → 97c)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (the 1 real SHIP artifact pre-Wf-74)
- `molmetal/scripts/template_ic50_pipeline.py` (the 1 real SHIP script)
- `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` (the 1 real SHIP JSON template)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)
- `TODO/pending/84_no_paper_task_without_authorization.md` (TODO-84 paper rule; outreach NOT gated)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run layer ship; carries the agreement-bench JSON shell with all `null` fields)
- `TODO/completed/81_wetlab_data_prep.md` (Wf-28 wet-lab data tier — 5 generic SOPs + outreach template)
- `TODO/completed/82_metric_gap_filling.md` §5 (R20-P1..P7 plan)
- `TODO/pending/wetlab_loi_*.md` (NEW this Wf, 3 institutional LOIs)
- `TODO/pending/wetlab_sop_*.md` (NEW this Wf, 3 institutional SOPs)
- `TODO/INDEX.md` §12.3 P3 mitigation (outreach dispatch instructions)
- `party-twenty-spirit` §3.5 (honest-negative framing)

---

**Date**: 2026-09-27 · **Verdict**: 🚀 SHIP — 3 IC50 pipeline artifacts (carry) + 3 LOI drafts (NEW) + 3 institutional-tailored SOPs (NEW) on disk; 0 outreach sent; CPU-only docs; wet-lab gate NOT active per TODO-84; 6-month wet-lab timeline dependent; LOI dispatch is Hugo Lee action · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
