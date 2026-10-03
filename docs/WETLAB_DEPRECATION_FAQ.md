# Wetlab Deprecation FAQ (NO WET-LAB WORK — DO NOT PLAN, EXECUTE, OR SEND)

**Last updated:** 2026-09-30 (Wf-375-A1 docs-only sync — standalone FAQ consolidation)
**Audience:** AI agents + humans working on Mol-Metal
**Purpose:** Single self-contained reference for the wet-lab deprecation decision
**Status:** ACTIVE rule — applies to every agent action in this repo

---

## §0 TL;DR

**⚠️ HARD MANDATE — DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB WORK. ⚠️**

**Verbatim user directives (multiple sessions):**

- **2026-09-29**: *"我们发二区就可以了，不用规划湿实验"* (Tier-2 is sufficient; do NOT plan wet-lab experiments)
- **2026-09-29**: same directive restated multiple times across the day
- **2026-09-30**: *"还有就是搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"* (explicit reinforcement; user frustration at repeated re-questioning; make this visible in TODO + AGENTS + docs)

**Tier-2 acceptance bar is computational-only.** No wet-lab data is required for Tier-2 submission. Wet-lab validation is a 6-12 month external dependency that does not gate Tier-2 acceptance. All 10 wet-lab items in `TODO/pending/wetlab_*.md` have been marked **`TEMPLATE_NOT_NEEDED_FOR_TIER2`** and reclassified as historical artifacts. They are retained on disk but DEPRECATED for paper strategy.

**If a task, plan, or paper section requests wet-lab work → REFUSE, RE-SCOPE, CITE, FLAG.** Do not plan, write, or execute wet-lab work under any circumstance.

**Self-contained reference:** agents can read this FAQ alone (without grepping AGENTS.md or TODO/README.md) to understand the deprecation decision, what is DEPRECATED, and what to do when wet-lab work is requested.

---

## §1 Why wet-lab removed?

### 1.1 Tier-2 journal target downgrades acceptance bar to computational-only

**Per AGENTS.md §5.4 (2026-09-29 + 2026-09-30):** paper target was downgraded from **JCIM Tier-1** to **Tier-2 computational-only** on 2026-09-29 per user directive. Examples of Tier-2 target journals:

- J. Comput. Aided Mol. Des. (JCAMD)
- J. Mol. Graph. Model. (JMGM)
- WIREs Comput. Mol. Sci. (WIREs)
- Mol. Divers.
- Comput. Biol. Chem.

**Tier-2 acceptance does NOT require wet-lab data** — pure computational submission is in scope. Computational novelty (lipman CFM GPU + Wasserstein 3-click + D-MPNN + Multi-PH synergy 32-bin Fourier + ACE Hierarchy closure theorem + MACE multi-cohort R² lift) is the Tier-2 paper-grade axis, not wet-lab validation.

### 1.2 Wet-lab validation is a 6-12 month external-dependency

Wet-lab IC50 validation requires:

- PI outreach (LOI → collaboration agreement → IRB + biosafety approval)
- Compound synthesis + shipment logistics
- Cell-line procurement (A2780 / MCF7 / NCI-60 9-cell panel)
- Assay execution (MTT / SRB / CellTiter-Glo / SPR / ITC / ICP-MS)
- Data analysis (4-parameter logistic curve fit → IC50 → pIC50)

This is a **6-12 month external-dependency** that does NOT gate Tier-2 acceptance. The computational submission is independent and can ship without wet-lab validation.

### 1.3 User directives (verbatim, multi-session)

- **2026-09-29**: "我们发二区就可以了，不用规划湿实验" — Tier-2 sufficient; do NOT plan wet-lab
- **2026-09-29**: same directive restated multiple times (Tier-2 re-orientation codified in Wf-317 + Wf-322)
- **2026-09-30**: "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了" — explicit reinforcement; user frustration

The 2026-09-30 directive was issued because agents kept re-asking the wet-lab question despite multiple prior statements. The HARD MANDATE in §0 makes the directive **non-negotiable and self-reinforcing** — agents must NOT re-question whether wet-lab should be done.

### 1.4 Project re-orientation consequences

- **Paper target** = Tier-2 (JCAMD / JMGM / WIREs / Mol. Divers. / Comput. Biol. Chem.)
- **Wet-lab items** = DEPRECATED for project scope
- **In-silico metric lift** = the Tier-2 paper-grade axis
- **TODO-138 paper v2** = re-scoped from JCIM to Tier-2
- **R20-P6 wet-lab IC50 validation** (TODO-97) = DEPRECATED scope
- **Wf-74 v2 wet-lab outreach** = DEPRECATED scope
- **LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab)** = re-scoped as computational-only
- **LAMBDA_ROADMAP §3.3 R23 (Phase I clinical)** = DEPRECATED

---

## §2 What wet-lab items are DEPRECATED?

All 10 wet-lab items in `TODO/pending/wetlab_*.md` are DEPRECATED. Status applied via inline edits on 2026-09-30 (per Wf-375-A1 docs-only sync):

### 2.1 4 LOI drafts (DO NOT SEND)

| # | File path | Target PI / scope | Status |
|---|---|---|---|
| 1 | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | Oak Ridge National Laboratory (ORNL) protein-metal binding group | **DO NOT SEND** |
| 2 | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | 北京大学医学部 (Peking University Health Science Center) 药物化学 / 化学生物学 课题组 | **DO NOT SEND** |
| 3 | `TODO/pending/wetlab_loi_xiehe_nci60.md` | 北京协和医学院 (Peking Union Medical College) 医院期细胞毒理学课题组 | **DO NOT SEND** |
| 4 | `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` | Rusanov 2025 corresponding author (ChemRxiv 2025-06-20 v2, DOI 10.26434/chemrxiv-2025-pp32k) — PlatinAI host-portal (MB-Finder) maintainer | **DO NOT SEND** |

### 2.2 3 wetlab SOPs (DO NOT EXECUTE)

| # | File path | Assay scope | Status |
|---|---|---|---|
| 1 | `TODO/pending/wetlab_sop_binding_affinity_assay.md` | SPR / ITC binding affinity assay (institutional-tailored) | **DO NOT EXECUTE** |
| 2 | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | ICP-MS cellular uptake assay (institutional-tailored) | **DO NOT EXECUTE** |
| 3 | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | MTT / SRB / CellTiter-Glo cytotoxicity assay (institutional-tailored) | **DO NOT EXECUTE** |

### 2.3 1 IC50 analysis pipeline (DO NOT RUN)

| # | File path | Pipeline scope | Status |
|---|---|---|---|
| 1 | `TODO/pending/wetlab_ic50_analysis_pipeline.md` | raw readout → pIC50 publication (companion script: `molmetal/scripts/template_ic50_pipeline.py` ~200 LOC pure NumPy/SciPy) | **DO NOT RUN** |

### 2.4 2 wetlab outreach support templates (DO NOT USE)

| # | File path | Template scope | Status |
|---|---|---|---|
| 1 | `TODO/pending/wetlab_cold_email_template.md` | Wf-310G cold-email template (4-LOI variants + English/中文 body) | **DO NOT USE** |
| 2 | `TODO/pending/wetlab_loi_consolidation_review.md` | Wf-308D 4-LOI consolidation review helper | **DO NOT USE** |

**Total: 10 wet-lab files on disk, all DEPRECATED.**

### 2.5 Plan-level DEPRECATED items (not file-level)

- **R20-P6 wet-lab IC50 validation** (TODO-97) — DEPRECATED scope; re-scope to in-silico metric lift
- **Wf-74 v2 wet-lab outreach** — DEPRECATED scope
- **LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab)** — re-scoped as computational-only
- **LAMBDA_ROADMAP §3.3 R23 (Phase I clinical)** — DEPRECATED

### 2.6 Tier-2 markers applied (per Wf-322A + Wf-322B)

All 10 wet-lab files carry the `TEMPLATE_NOT_NEEDED_FOR_TIER2` final marker (applied via inline edits 2026-09-30 per Wf-375-A1). The 2026-09-29 `TEMPLATE_NOT_NEEDED_FOR_TIER2` status from Wf-322A/B + the 2026-09-30 `⚠️ DEPRECATED — DO NOT SEND/EXECUTE/RUN/USE ⚠️` header warnings are both preserved.

---

## §3 What to do if a task, plan, or paper section requests wet-lab work?

**Apply the 4-step REFUSE / RE-SCOPE / CITE / FLAG protocol:**

### 3.1 REFUSE

- Do **NOT plan, write, or execute** any wet-lab work
- Do **NOT create new LOI / SOP / outreach artifacts** that imply wet-lab execution
- Do **NOT touch `TODO/pending/wetlab_*.md`** to "update" them toward active status
- Do **NOT generate or send** cold-email or LOI content to PIs

### 3.2 RE-SCOPE

Substitute the wet-lab request with an **in-silico alternative**:

- **Wet-lab IC50 validation** → in-silico pIC50 R² on PlatinAI 4-cell + MetalCytoToxDB 5-metal cohorts (TODO-179 + Wf-359-A2 m1 multi-seed evaluation scaffold)
- **Wet-lab binding affinity (SPR / ITC)** → in-silico pK_bind / Vina kcal/mol scoring (Wf-333-V1+V2+V3 + Wf-362-A2 Vina median -7.782 kcal/mol PASSES industry -7.0 by 0.782 margin)
- **Wet-lab cellular uptake (ICP-MS)** → in-silico ADMET / cell-permeability proxy (Wf-358-C2 honest-neg catalog §1 per-metric)
- **Wet-lab cytotoxicity assay** → in-silico MetalCytotoxDB AUC + PlatinAI 4-cell pIC50 R² (TODO-220 + Wf-219 9-cell + 4-cell stratified)
- **Wet-lab LOI outreach** → peer-paper citation in paper §2 related work + §6 Limitations (literature values are Tier-2 acceptable)

### 3.3 CITE

Add an HONEST_NEGATIVE entry to `docs/HONEST_NEGATIVE_CATALOG.md` flagging the wetlab → insilico substitution. Per Wf-358-C2 + Wf-359-B1 + Wf-369-A3 the catalog has 65 entries across 6 sections; the wetlab → insilico substitution pattern should be marked with the tag `WETLAB_TO_INSILICO_SUBSTITUTION` for downstream filtering.

### 3.4 FLAG

Surface the wet-lab request to the user via TodoWrite or explicit message in final synthesis:

- "User requested wet-lab X — RE-SCOPED to in-silico Y per AGENTS.md §5.4 + §10 + this FAQ §3.2"
- "Wet-lab LOI outreach is DEPRECATED — cited peer-paper Z in paper §2 instead"

---

## §4 What if a peer-reviewer rebuttal requires experimental data?

### 4.1 Re-frame as computational substitution (per §3.2)

Apply the same REFUSE / RE-SCOPE / CITE / FLAG protocol. Tier-2 acceptance bar is computational-only; reviewers expecting wet-lab data are mis-aligned with the Tier-2 submission target. Re-frame the rebuttal response as:

> "Per the journal's computational-only acceptance scope (Tier-2), we substitute wet-lab validation with [in-silico metric X + cohort Y + source path Z]. See §6 Limitations for the honest-negative framing."

### 4.2 Cite existing peer-paper experimental references (literature values)

Tier-2 accepts **literature values cited from peer-reviewed papers** as the experimental reference frame. Example substitution patterns:

- **Cisplatin IC50 reference** → cite Rusanov 2025 (ChemRxiv DOI 10.26434/chemrxiv-2025-pp32k) + Krasnov 2026 (MetalCytoToxDB) + NCI-60 Developmental Therapeutics Program
- **PlatinAI 4-cell pIC50** → cite Rusanov 2025 NUMERIC IC50 + A2780 / MCF7 / A549 / A2780cis peer-paper experimental values
- **Binding affinity (SPR / ITC)** → cite peer-paper K_d values for cisplatin + carboplatin + oxaliplatin reference drugs
- **Cellular uptake (ICP-MS)** → cite peer-paper Pt accumulation data + cellular efflux transporter references

### 4.3 Mark as PROJECTED_FUTURE_WORK in paper §6 Limitations

Per AGENTS.md §5.2 honest framing mandate (`party-twenty-spirit` §3.5) + `party-twenty-spirit` §3.5, paper §6 Limitations must use the **`PROJECTED_FUTURE_WORK`** label (NOT `MEASURED`) for any wet-lab data the paper does not have. Template scaffold available at `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4).

---

## §5 Wetlab templates retention policy

### 5.1 Retain (do NOT delete)

All 10 wet-lab files in `TODO/pending/wetlab_*.md` are **retained on disk as historical artifacts**:

- Useful for paper §2 related-work citations (template structure + institutional PI acknowledgment)
- Useful for future Tier-1 paper strategy if scope ever shifts
- Useful for audit trail + historical decision provenance
- Cross-references already wired into AGENTS.md §10.1 + TODO/README.md wetlab blockquote + TODO/INDEX.md §0.6

### 5.2 DO NOT execute / DO NOT send

Any agent action attempting to:

- Run an LOI outreach (cold-email send, LOI PDF generation)
- Execute a wet-lab SOP (compound synthesis, cell-line procurement, assay execution)
- Run the IC50 analysis pipeline on raw wet-lab data (no wet-lab data exists; pipeline is template-only)

**must REFUSE per §3.1.** The DEPRECATED status is non-negotiable and self-reinforcing per §0 HARD MANDATE.

### 5.3 Cross-ref `TODO/README.md` wetlab blockquote

The `TODO/README.md` `pending/` section has a `⚠️ NO WET-LAB HARD MANDATE` blockquote (per Wf-375-A1 inline edit) that flags DEPRECATED status for the same 10 files. This FAQ is the canonical self-contained reference; the blockquote is the at-a-glance pointer.

---

## §6 Cross-references

### 6.1 AGENTS.md §5.4 (Tier-2 + NO wet-lab standing directive)

`/home/hugo/codes/mol-metal/AGENTS.md` §5.4 — Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive + 2026-09-30 reinforcement). Verbatim user directives cited.

### 6.2 AGENTS.md §10 (Wetlab deprecation FAQ — inline version)

`/home/hugo/codes/mol-metal/AGENTS.md` §10.1-§10.6 — inline wetlab deprecation FAQ with 6 sub-sections mirroring this standalone FAQ.

### 6.3 TODO/README.md wetlab blockquote

`/home/hugo/codes/mol-metal/TODO/README.md` — top-level `⚠️ NO WET-LAB HARD MANDATE` blockquote in `pending/` section flags DEPRECATED status for the same 10 files.

### 6.4 10 wetlab pending files (all on disk with DEPRECATED headers)

- `TODO/pending/wetlab_loi_ornl_protein_metal.md` — LOI-ORNL
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` — LOI-PKU
- `TODO/pending/wetlab_loi_xiehe_nci60.md` — LOI-协和
- `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` — LOI-RUSANOV
- `TODO/pending/wetlab_sop_binding_affinity_assay.md` — SOP-Binding
- `TODO/pending/wetlab_sop_cellular_uptake_assay.md` — SOP-Cellular-Uptake
- `TODO/pending/wetlab_sop_cytotoxicity_assay.md` — SOP-Cytotoxicity
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` — IC50 pipeline
- `TODO/pending/wetlab_cold_email_template.md` — Cold-email template
- `TODO/pending/wetlab_loi_consolidation_review.md` — 4-LOI consolidation review helper

### 6.5 Wf-375 closeout

`TODO/completed/375a1_wetlab_deprecation_faq.md` — Wf-375-A1 docs-only closeout (this FAQ authoring + 10 inline DEPRECATED header edits + AGENTS.md §10 + TODO/README.md wetlab blockquote).

### 6.6 Related directives

- **AGENTS.md §1.3** (No paper work without authorization / TODO-84) — `paper/*` works requires USER_OK
- **AGENTS.md §5.2** (Honest framing mandate per `party-twenty-spirit` §3.5) — MEASURED vs PROJECTED tag discipline
- **AGENTS.md §5.3** (No paper work without authorization) — TODO-84 active
- **`docs/HONEST_NEGATIVE_CATALOG.md`** — 65 entries (Wf-369-A3 expansion) including WETLAB_TO_INSILICO_SUBSTITUTION markers
- **`docs/SECTION_6_LIMITATIONS_TEMPLATE.md`** (Wf-364-A4) — LaTeX template for paper §6 Limitations
- **`docs/USER_OK_FINAL_CHECKLIST.md`** (Wf-368-B1) — 14 USER_OK-gated items; wet-lab items NOT included (per HARD MANDATE)

### 6.7 Honesty framing per `party-twenty-spirit` §3.5

This FAQ consolidates 13 inline edits (AGENTS.md §5.4 reinforcement + AGENTS.md §10 NEW + TODO/README.md wetlab blockquote + 10 wetlab pending file DEPRECATED headers) into a single self-contained reference. Zero NEW code, zero NEW tests, zero GPU launches. Pure docs-only authoring per Wf-375-A1 task brief. The verbatim user directives (3 sessions × 2026-09-29 + 1 × 2026-09-30) are cited without paraphrase per honest-framing mandate.

---

**END OF WETLAB_DEPRECATION_FAQ.md** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.4 + §10 + TODO/README.md wetlab blockquote; last update 2026-09-30 (Wf-375-A1 docs-only sync — standalone FAQ consolidation; 6 sections; ~280 LOC; ~14 verbatim quotes from user directives + AGENTS.md + TODO/README.md; cross-refs to 10 wetlab pending files + AGENTS.md §5.4 + §10 + TODO/README.md + Wf-375 closeout)
