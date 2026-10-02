# LOI-RUSANOV — Letter of Intent: PlatinAI 4-cell × 200 compound NUMERIC IC50 collaboration

**Document ID:** LOI-RUSANOV-2026-09
**Version:** 1.0 (draft, ready for Hugo Lee review)
**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10 + 2026-09-30 user reinforcement "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了". Tier-2 journal acceptance does not require wet-lab data. Retained as **historical artifact only**; any agent action attempting to send this LOI must REFUSE per AGENTS.md §10.3.
**(Historical note: DRAFT — not sent at this revision)** (Hugo Lee coordinates the cold-email per `TODO/INDEX.md` §12.3 P3 mitigation + `TODO/pending/84_no_paper_task_without_authorization.md` paper rule; no action will be taken — wet-lab DEPRECATED per AGENTS.md §5.4 + §10)
**Target PI:** Rusanov 2025 corresponding author (ChemRxiv 2025-06-20 v2, DOI 10.26434/chemrxiv-2025-pp32k) — PlatinAI host-portal (MB-Finder) maintainer
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`
**Subject (for the cold-email):** "PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)"

---

> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (2026-09-29)**
>
> Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
> Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
> This LOI draft is preserved as historical reference but NOT for sending.

---

## 1. Purpose

This Letter of Intent (LOI) requests a short-term **data collaboration** with the
PlatinAI / MB-Finder team (Rusanov 2025, DOI 10.26434/chemrxiv-2025-pp32k,
hosted at https://www.mb-finder.org). The collaboration will provide Mol-Metal
access to **per-cell-line NUMERIC IC50** values for ~200 Pt compounds spanning
the 4-cell PlatinAI benchmark (MCF7 / A549 / A2780 / A2780cis), which will
replace our currently **coverage-only** surrogate (`has_<line>` boolean flags
from Wf-295B / Wf-301C) with a real-label dataset that can drive
wet-lab-validated in-silico pIC50 models (R20-P2 / R21-P4 ship targets).

## 2. Background — what Mol-Metal brings

Mol-Metal is an open-source metallodrug design pipeline (mol-metal.org, GitHub
`mol-metal`). The R18 release (2026-09) ships:

- **Click-chemistry metallodrug generator** (`molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
- **Lambda-combinator search** with metal-aware reward shaping (`molmetal/molmetal_lam/search_alg/proof_search.py`, `molmetal/molmetal_lam/reward/learned_shaping.py`).
- **Coupled flow-matching backbone** (`molmetal/adapters/flow_matching_lipman/`).
- **Multi-task D-MPNN pIC50 predictor** on MetalCytoToxDB 5-metal subset (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`, Wf-124 P3).
- **MB-Finder coverage-matrix loader** (`molmetal/scripts/platinai_real_label_loader.py`, Wf-295B + Wf-301C stratified) — currently ships **REAL compound IDs + SMILES + cell-line coverage MATRIX** for 100 Pt compounds (A549=100; MCF7/A2780/A2780cis await stratified extension Wf-305A) and exposes the per-cell-line NUMERIC gap this LOI addresses.
- **D-MPNN pIC50 spread 1.7 log units** across R20-P6 top-20 candidates (`molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json`).
- **Companion IC50 analysis pipeline** (`TODO/pending/wetlab_ic50_analysis_pipeline.md` + `molmetal/scripts/template_ic50_pipeline.py`) ready to consume NUMERIC CSV the moment it ships.

Open datasets shipped: PlatinAI coverage matrix (CC-BY, harvested via MB-Finder portal), tmQM (CC-BY), MetalCytoToxDB Ru subset (CC-BY-SA).

## 3. What the PlatinAI / MB-Finder team brings

- **Per-cell-line NUMERIC IC50** for ~4,078 Pt complexes across MCF7 / A549 / A2780 / A2780cis (Rusanov 2025 Table S4-S5 supplementary).
- **MB-Finder portal** infrastructure (https://www.mb-finder.org) — already the hosted interface for the same compound catalogue; coverage matrix is publicly queryable; per-cell-line NUMERIC requires author approval.
- **Domain expertise** in metallodrug descriptor curation, PlatinAI model architecture, MB-Finder taxonomy.
- **Track record** of public-data sharing (CC-BY PlatinAI compound catalogue).

## 4. Why we need this collaboration

### 4.1 Honest framing — current limitation (per Wf-297B §6)

Our **closest-to-real** PlatinAI dataset today (Wf-295B + Wf-301C) emits:

| Field | Real / Synthetic | Source |
|---|---|---|
| `mb_id`, `imageId`, `smiles`, `mw` | **REAL** | MB-Finder portal `get-substances` endpoint |
| `has_a549`, `has_mcf7`, `has_a2780`, `has_a2780cis` | **REAL** (coverage matrix) | MB-Finder portal `get-cell-lines` endpoint |
| Per-cell-line `bioDataCount` | **REAL** (counts) | MB-Finder portal `get-cell-lines` endpoint |
| **Per-cell-line NUMERIC IC50** | **MISSING** (NOT exposed by public API) | Rusanov 2025 Table S4-S5 supplementary |

The 4th row is the binding gap. Without NUMERIC IC50, our in-silico pIC50
predictors are **trained on synthetic pIC50 labels** (Wf-278B loader) and can
**never reach** the **Pearson r ≥ 0.7** wet-lab-vs-in-silico honesty gate
that R20-P6 §5 requires (TODO/pending/97_r20_p6_wetlab_ic50.md).

### 4.2 What we tried before this LOI

| Strategy | Status | Reference |
|---|---|---|
| S1 — Selenium UC + ChemRxiv CF bypass | **BLOCK_CF** (90 s, 0 NUMERIC) | `molmetal/reports/wf304b_selenium/selenium_report.md` |
| S7 — MB-Finder portal coverage matrix | **PARTIAL** (coverage only, 100 compounds) | `molmetal/reports/wf295b_platinai_loader/loader_report.md` + Wf-301C stratified |
| **S6 — Direct author outreach (this LOI)** | **DRAFT** (not yet sent) | This document |

### 4.3 What we propose instead

A **2-week**, **single-email** data collaboration:

1. Mol-Metal sends this LOI + 1-page project summary.
2. Rusanov team confirms fit + approves supplementary IC50 release.
3. Rusanov team sends Table S4-S5 supplementary xlsx (or CSV) for ~200 Pt compounds.
4. Mol-Metal ingests NUMERIC labels into `platinai_4cell_real_ic50.csv` (CC-BY release).
5. Joint acknowledgement in Mol-Metal paper §5 (R20 release).

## 5. Proposed collaboration (1 phase, 2-6 weeks)

| Phase | Duration | Mol-Metal deliverable | PlatinAI / MB-Finder deliverable |
|-------|----------|------------------------|----------------------------------|
| **Phase 1** | 0–2 wk | This LOI + 1-page project summary + `mol-metal` GitHub link + `wf58_dmpnn_predictions/per_compound_pic50.json` snapshot | Acknowledge fit; approve Table S4-S5 supplementary release for ~200 Pt compounds |
| **Phase 2** | 2–4 wk | `platinaI_4cell_real_ic50.csv` ingest + regression diagnostics + honest MEASURED-vs-PROJECTED comparison; retrain D-MPNN on real labels; updated `molmetal/reports/wf301d_real_ic50/` | Email/transfer Table S4-S5 supplementary xlsx (or CSV) for ~200 compounds |
| **Phase 3** | 4–6 wk | Joint acknowledgement in Mol-Metal paper §5; CC-BY release of the real-label dataset on `mol-metal/data/platinai_real_ic50.csv`; citation update | Optional: co-author review of the NUMERIC-label ingest section in Mol-Metal paper §5 |

60-day cadence: Day 0 (email send) → Day +7 (light follow-up) → Day +21 (second follow-up) → Day +28 (intro Zoom if requested) → Day +60 (data in hand, Phase 2 launches).

## 6. Authorship + IP

- **Phase-2 success** (NUMERIC labels ingested + joint diagnostics shipped) qualifies the Rusanov team PI for a **mid-author position** on the Mol-Metal paper with the data-validation section, and a **prominent Acknowledgement** naming "Rusanov 2025 PlatinAI / MB-Finder team for per-cell-line IC50 NUMERIC release".
- **Phase-3 success** (NUMERIC labels drive a published D-MPNN retrain that beats the synthetic-label baseline by ≥ 0.1 log units on held-out MCF7 + A2780cis) qualifies for **co-corresponding authorship** on the Mol-Metal paper §5.
- All parties agree on author order in writing before submission.
- The NUMERIC IC50 dataset becomes part of the open Mol-Metal dataset on publication under **CC-BY-4.0**, with explicit citation to Rusanov 2025 + MB-Finder portal.

## 7. Funding + compute

- **Mol-Metal side**: software, compute, in-silico predictions, manuscript preparation. No wet-lab budget.
- **PlatinAI / MB-Finder side**: data release effort (~1-2 days of curation / redaction work if any compound IDs are sensitive); no compute commitment required.
- **No overhead or indirect-cost transfers** are implied.
- **No funding exchange** is requested or implied.

## 8. Data scope

### 8.1 Target compounds (~200)

| Cohort | n | Selection criterion |
|---|---|---|
| **MB-Finder catalogue Pt subset** | 100 | Top 100 by `bioDataCount` from Wf-295B harvest (REAL `mb_id` list) |
| **MB-Finder stratified 4-cell subset** | 100 | Top 25 per cell line (MCF7/A549/A2780/A2780cis) from Wf-301C + Wf-305A stratified extension |

### 8.2 Per-cell-line IC50 columns

```
mb_id, MCF7_IC50_uM, A549_IC50_uM, A2780_IC50_uM, A2780cis_IC50_uM,
MCF7_pIC50, A549_pIC50, A2780_pIC50, A2780cis_pIC50,
resistance_fold = 10**(pIC50_A2780cis - pIC50_A2780)
```

## 9. Expected outputs

- Per-compound × per-cell-line **NUMERIC IC50 ± std** (n=3 technical replicates; 8-dose half-log grid).
- Per-compound **pIC50 + 95% CI** (1,000-iteration residual bootstrap).
- Per-compound **resistance fold** (A2780cis / A2780).
- All raw IC50 curves uploaded to `data/wetlab/platinai_real_ic50/<run_id>/` with checksums.
- JSON output conforming to `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` schema.
- CC-BY-4.0 release of `molmetal/data/platinai_real_ic50.csv` + `molmetal/data/platinai_real_ic50.meta.json`.

## 10. Timeline + deliverables

| Week | Milestone | Deliverable |
|---|---|---|
| 0 | LOI sent (cold email with 1-page project summary attachment) | LOI + summary |
| 1 | PlatinAI PI acknowledges fit | Acknowledgement email |
| 2 | PlatinAI PI approves supplementary release | Approval + xlsx transfer |
| 3 | NUMERIC labels ingested into `platinai_4cell_real_ic50.csv` | Ingested dataset |
| 4 | D-MPNN retrain on real labels (R20-P2 ship) | `molmetal/reports/wf301d_real_ic50/retrain_report.md` |
| 5 | Honest MEASURED-vs-PROJECTED comparison (Pearson r + RMSE) | `wf301d_real_ic50/diagnostics.json` |
| 6 | Joint acknowledgement draft + CC-BY release | `molmetal/data/platinai_real_ic50.csv` + paper §5 acknowledgement |

## 11. Risk + mitigation

| Risk | Mitigation |
|---|---|
| **Cold email unanswered** | Day +7 / +21 follow-ups; cc ORNL / PKU collaborator network for warm intro; fall back to Wf-295B coverage-only dataset |
| **CF or institutional barriers on data release** | This LOI explicitly avoids ChemRxiv / supplementary mirror — direct author-to-author email transfer |
| **200 compounds too few for D-MPNN retrain** | Fall back to kNN oracle surrogate (TODO/pending/cytobench) until full 4,078 compound release ships |
| **NUMERIC labels conflict with coverage matrix** | Cross-validate Wf-295B `bioDataCount > 0` against NUMERIC labels; flag mismatches as honest-negative |
| **Per-cell-line pIC50 spread collapses** | Report spread honestly even if < 1.7 log units (Wf-58 baseline) — paper §6 limitations |

## 12. Confidentiality and data sharing

- All data transfer happens under the data-sharing MoU template (`molmetal/docs/wetlab_protocols/data_sharing_agreement.md`).
- Pre-publication embargo until both teams approve submission.
- All R18 / R20 Mol-Metal code is open-source (CC-BY-4.0); the released NUMERIC IC50 data becomes part of the open Mol-Metal dataset on publication.
- **No exclusivity clause** — the PlatinAI team retains full rights to use, redistribute, and re-publish the NUMERIC IC50 data.

## 13. Next steps

1. Hugo Lee sends this LOI via cold-email to Rusanov 2025 corresponding author (template at `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §3) with the 1-page project summary attachment.
2. Day +7 light follow-up; Day +21 second follow-up.
3. Day +28 intro Zoom (30 min) — confirm fit, transfer logistics, target compound selection.
4. Day +60 NUMERIC labels in hand; Phase 2 launches.

## 14. Honest framing (party-twenty-spirit §3.5 + Wf-297B §6)

- This LOI is a **draft** ready for Hugo Lee's review; **0 LOI sent** at this revision.
- **0 per-cell-line NUMERIC IC50 MEASURED today**; in-silico pIC50 models are trained on synthetic pIC50 labels (Wf-278B) — known honest-negative per Wf-297B §6.
- The headline `Pearson r ≥ 0.7` between D-MPNN-predicted pIC50 and Rusanov-measured NUMERIC IC50 cannot MEASURE until ≥1 PI reply + data transfer + 2-week ingest completes (earliest 2026-Q4 if Rusanov team responds in week 1; realistic 2027-Q1 if cold-email latency).
- Paper §5 wet-lab validation section 🚫 **DEFERRED** per TODO-84 paper rule; outreach is NOT gated by TODO-84 (per `TODO/pending/84_no_paper_task_without_authorization.md` cross-ref "outreach is not gated"), but **sending** this LOI requires explicit user OK.

## 15. Cross-references

- `molmetal/scripts/platinai_real_label_loader.py` — Wf-295B / Wf-301C stratified coverage-only loader (gap this LOI addresses)
- `molmetal/reports/wf304b_selenium/selenium_report.md` — Wf-304B BLOCK_CF honest-negative
- `molmetal/reports/wf295b_platinai_loader/loader_report.md` — Wf-295B coverage-only ship
- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` — §3 R1 (S7) + §3 R2 (S1) + §3 R3 (S6 outreach)
- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28 generic email template, WLP-OUT-001)
- `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` (Wf-28 data-sharing MoU template)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (companion IC50 analysis pipeline)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` (parallel ORNL LOI — binding validation)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` (parallel PKU LOI — cisplatin-class synthesis + IC50)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` (parallel Xiehe LOI — NCI-60 expansion)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 wet-lab IC50 plan)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run layer ship)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; outreach NOT gated)

## 16. Revision history

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-29 | Wf-305A | Initial PlatinAI / Rusanov outreach LOI draft (ready for Hugo Lee review; **0 sent**). Subject: "PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)". |
