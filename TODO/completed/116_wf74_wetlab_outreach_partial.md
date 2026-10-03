# TODO-116 — Wf-74 wet-lab outreach honest PARTIAL reclassification

**Date**: 2026-09-27
**Status**: 🚀 **Active — corrects Wf-74 close-out overstatement (3 LOI + 3 SOP claimed SHIP, only 1 doc actually written)**
**Priority**: **P0 CRITICAL** (silent close-out overstatement — paper-rule-adjacent since wet-lab §5 is gated)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 4 loose-ends research (2026-09-27)
**Skill alignment**: `molmetal-wetlab-protocol` (S7 from Wf-62)

---

## §0 Why this plan exists

Agent 4 audit found Wf-74 wet-lab pre-work close-out overstates SHIP:

**Close-out (`TODO/completed/97b_wetlab_prework_docs.md`) claims**:
- 3 LOI drafts (ORNL / PKU / 协和)
- 3 SOPs (cytotoxicity / binding / cellular uptake)
- IC50 analysis pipeline doc

**Reality on disk**:
- ✅ `TODO/pending/wetlab_ic50_analysis_pipeline.md` (346 lines, EXISTS)
- ✅ `molmetal/scripts/template_ic50_pipeline.py` (28 KB, EXISTS)
- ✅ `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` (EXISTS)
- ❌ `molmetal/docs/wetlab_protocols/loi_ornl_protein_metal.md` — **NOT EXISTS**
- ❌ `molmetal/docs/wetlab_protocols/loi_pku_cisplatin_ic50.md` — **NOT EXISTS**
- ❌ `molmetal/docs/wetlab_protocols/loi_xiehe_nci60.md` — **NOT EXISTS**
- ❌ 3 institutional-tailored SOPs — **NOT EXISTS** (only Wf-28 generic protocols exist)
- ❌ 0 LOIs sent / 0 MoUs signed / 0 PI replies / 0 cell culture started / 0 IC50 MEASURED

**The IC50 pipeline doc was a single artifact, not 7 separate deliverables**.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (documentation fix + outreach) |
| Wall-time | 2 hr CPU (reclassification) + 4 hr (write the 3 LOIs + 3 SOPs) + Hugo Lee action (send LOIs) |
| Output | reclassified close-out v2 + 3 LOIs + 3 SOPs (if Hugo wants) |
| Tests | none (documentation fix) |

## §2 Solution path

### P1.1 — Honest reclassification (mandatory)

1. **Author v2 close-out** (`TODO/completed/97c_wf74_reclassification.md`) with accurate inventory:
   - ✅ IC50 analysis pipeline doc SHIP (1 doc, 346 lines)
   - ✅ template_ic50_pipeline.py SHIP (1 script)
   - ✅ template_per_compound.json SHIP (1 JSON template)
   - ❌ 3 LOI drafts NOT WRITTEN
   - ❌ 3 SOPs NOT WRITTEN
   - ❌ 0 outreach (0 LOI sent, 0 PI contacted)
2. **Cross-link from v1 close-out**: add header note "Corrected by TODO-116 — see v2 at `97c_wf74_reclassification.md`".

### P1.2 — Write the 3 LOIs (4 hr, optional)

1. **`loi_ornl_protein_metal.md`** — Letter of Intent to ORNL PI for protein-metal binding assay
2. **`loi_pku_cisplatin_ic50.md`** — LOI to PKU PI for cisplatin-class IC50 validation
3. **`loi_xiehe_nci60.md`** — LOI to 协和 PI for NCI-60 panel screening

### P1.3 — Write the 3 SOPs (2 hr, optional)

1. **`sop_cytotoxicity_assay.md`** — institutional-tailored cytotoxicity SOP
2. **`sop_binding_affinity_assay.md`** — institutional-tailored binding SOP
3. **`sop_cellular_uptake_assay.md`** — institutional-tailored uptake SOP

### P1.4 — Outreach dispatch (Hugo Lee action, NOT Claude task)

The actual *send* of LOI emails is a Hugo Lee action. The next concrete step is: "Hugo coordinates first cold-email send to ORNL/PKU/协和 PI per `TODO/INDEX.md` §12.3 P3 mitigation". Outreach itself is **NOT gated by TODO-84** (it's outreach/protocol docs, classified as CODE-ONLY per §1.2).

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Audit Wf-74 agent JSONLs + author v2 close-out | 2 hr CPU | `TODO/completed/97c_wf74_reclassification.md` (NEW) | accurate inventory |
| **P1.2** Write 3 LOI drafts | 4 hr CPU | 3 NEW LOI docs | per-LOI: scope + IC50 protocol + expected outputs + timeline + funding |
| **P1.3** Write 3 SOPs | 2 hr CPU | 3 NEW SOP docs | per-SOP: solvent + media + dosing + replicates |
| **P1.4** Outreach dispatch | Hugo Lee action | 3 cold-emails sent | per-PI: confirmation email received |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: 1 IC50 pipeline doc + 1 template script + 1 JSON template SHIP (existing)
- **PROJECTED**: 3 LOIs + 3 SOPs written (P1.2 + P1.3) + 3 cold emails sent (P1.4)

## §5 Honest framing

This is **close-out overstatement** — not full fabrication (the IC50 pipeline is real), but the close-out inflated the deliverable count. The fix is accurate inventory + (optional) completing the 6 missing docs.

**Important**: Outreach is NOT Claude-task — it's a Hugo Lee action. The 3 LOI drafts are docs that Claude CAN write; the actual email *send* requires Hugo's coordination with the PIs.

## §6 Cross-references

- Agent 4 loose-ends research (2026-09-27)
- `TODO/completed/97b_wetlab_prework_docs.md` (the overstated v1 close-out)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (the 1 real SHIP artifact)
- `molmetal/scripts/template_ic50_pipeline.py` (the 1 real SHIP script)
- TODO/pending/97_r20_p6_wetlab_ic50.md (parent plan)
- TODO/pending/wetlab_loi_* (the 7 wet-lab docs created by Wf-74 — verify which actually exist)
- TODO/pending/84_no_paper_task_without_authorization.md (TODO-84 paper rule — outreach NOT gated)
- TODO/INDEX.md §12.3 P3 mitigation (outreach dispatch instructions)
- party-twenty-spirit §3.5 (honest-negative framing)
