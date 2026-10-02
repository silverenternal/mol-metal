# Wf-308D — 4-LOI Consolidation Review Helper

**⚠️ DEPRECATED — DO NOT USE (2026-09-30) ⚠️** — see AGENTS.md §5.4 + §10 + 2026-09-30 user reinforcement "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了". This consolidation review helper is preserved as historical reference only; any agent action attempting to use this helper for active LOI outreach must REFUSE per AGENTS.md §10.3.
**(Historical note: TEMPLATE_NOT_NEEDED_FOR_TIER2 per AGENTS.md §5.4, 2026-09-29)** — wet-lab LOI outreach DEPRECATED for project scope; Tier-2 journal acceptance does not require wet-lab data.

**Document ID:** TODO-308D / Wf-308D PART 2
**Version:** 1.0 (review helper, NOT a sent letter)
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`
**Scope:** Review-and-decide helper only. **NO send** — cold-email is user's
responsibility per `TODO/pending/84_no_paper_task_without_authorization.md`.

---

> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (2026-09-29)**
>
> Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
> Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
> This review helper is preserved as historical reference but NOT for
> sending the 4 LOIs.

---

## 1. Why this helper exists

As of 2026-09-29 there are **4 LOI drafts** sitting in `TODO/pending/`:

1. `wetlab_loi_ornl_protein_metal.md` (Wf-74 / WLP-OUT-ORNL)
2. `wetlab_loi_pku_cisplatin_ic50.md` (Wf-74 / WLP-OUT-PKU)
3. `wetlab_loi_xiehe_nci60.md` (Wf-74 / WLP-OUT-XIEHE)
4. `wetlab_loi_rusanov_2025_platinai.md` (Wf-305A NEW; WLP-OUT-RUSANOV)

All four are DRAFT — **0 LOI sent at this revision**. Per `TODO/INDEX.md`
§12.3 P3 mitigation + `TODO/pending/84_no_paper_task_without_authorization.md` paper
rule + `TODO/INDEX.md` P0 TODO-116 (Wf-74 v2 close-out rewrite), Hugo Lee personally
coordinates the cold-email send and the PI outreach is **NOT** delegated to
subagents / scripts. This helper gives Hugo a single-page review surface to
decide send order, fill in missing emails, and tick off the per-LOI next-action
checklist.

## 2. 4-row LOI summary table

| # | File (path) | Target PI / Lab | PI Email (Hugo's TODO) | Status | What's asked | What's offered | Time / risk |
|---|---|---|---|---|---|---|---|
| 1 | `wetlab_loi_rusanov_2025_platinai.md` (Wf-305A) | Rusanov 2025 corresponding author (PlatinAI / MB-Finder portal maintainer); ChemRxiv DOI 10.26434/chemrxiv-2025-pp32k | **TBD — Hugo to look up from ChemRxiv submission metadata** | DRAFT v1 | Per-cell-line NUMERIC IC50 release for ~200 Pt compounds across MCF7 / A549 / A2780 / A2780cis (Rusanov 2025 Table S4-S5 supplementary xlsx/CSV) | mid-author + co-corresponding authorship on Mol-Metal paper §5; CC-BY release of `platinai_4cell_real_ic50.csv`; prominent Acknowledgement naming Rusanov team | **2-6 weeks** (single email + transfer). Low risk: data-only request, no wet-lab, no funding. **Replaces coverage-only synthetic pIC50 surrogate** — closes the Wf-297B §6 gap that gates the Pearson r ≥ 0.7 honesty rule for R20-P6 paper §5. |
| 2 | `wetlab_loi_pku_cisplatin_ic50.md` (Wf-74 v1) | 北京大学医学部 (Peking University Health Science Center) 药物化学 / 化学生物学 课题组 | **TBD — Hugo to identify from PKU faculty page or mutual contact** | DRAFT v1 | Cisplatin-class 4-cell IC50 validation (MCF7 / A549 / A2780 / A2780cis) on top-20 R20-P6 candidates (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11); optional NCI-60 9-cell extension | mid-author + co-corresponding on data-validation section; joint manuscript; model retrain on measured IC50 | **6-12 months** (3-phase). Medium risk: cisplatin analog synthesis + 4-cell culture + cisplatin-resistant A2780cis line maintenance. |
| 3 | `wetlab_loi_xiehe_nci60.md` (Wf-74 v1) | 北京协和医学院 (Peking Union Medical College) 医院期细胞毒理学 课题组 | **TBD — Hugo to identify from 协和 faculty page** | DRAFT v1 | NCI-60 9-cell **完整 panel** (SRB/MTT) + ICP-MS cellular uptake on top-20 candidates; lead compound 临床转化 prep | co-corresponding on data-validation section + cover-sheet hospital name; mid-author on partial 9-cell | **6-12 months** (3-phase). Higher risk: IRB + mycoplasma + 9-cell culture + ICP-MS instrument access (Thermo iCAP RQ / Agilent 7900) + GCP-grade clinical-prep workflow. |
| 4 | `wetlab_loi_ornl_protein_metal.md` (Wf-74 v1) | Oak Ridge National Laboratory (ORNL) protein-metal binding group (SPR / ITC) | **TBD — Hugo to identify from ORNL directory (Cn/journals)** | DRAFT v1 | SPR (Biacore) + ITC (MicroCal PEAQ) K_d validation on top-20 candidates × 4 proteins (GST / cisplatin-DNA lesion / p53 DBD / metallothionein MT-2A) | co-corresponding + mid-author; joint grant (NIH R01 / DOE BER / NSF) — ORNL as fiscal host | **9-12 months** (3-phase). Highest risk: AMBER license + SPR regeneration conditions + 4 protein-expression pipeline + ORNL cluster access. |

## 3. Recommended send order

The order below is **NOT** random; it is the lowest-friction → highest-friction
ladder, optimized for **early signal** (data-flow + paper-traction uplift) before
slow wet-lab commitments.

### Send 1 — LOI-RUSANOV (data collaboration, 2-6 weeks)

**Why first:**
- **Pure data request** (no wet-lab cost, no facility, no funding exchange).
- Closes the **Pearson r ≥ 0.7** honesty gap for **paper §5** (R20-P6 ship
  target) — currently blocked on Wf-297B §6 NUMERIC gap.
- 2-6 weeks turnaround matches R20 paper timeline.
- Single-email cadence (Day 0 / +7 / +21 / +28 / +60) — does not consume Hugo's
  bandwidth on long-horizon wet-lab review cycles.
- High data-value-per-unit-effort — even a partial release (just MCF7 + A549
  per-cell IC50) unblocks Wf-301D real-label retrain.
- **Rusanov team has prior track record of CC-BY PlatinAI release** — lowers
  acceptance friction.

### Send 2 — LOI-PKU (cisplatin 4-cell, 6-12 months)

**Why second:**
- **Lower cell-line complexity** (4 cells vs 9 cells) and well-known
  cisplatin / carboplatin / oxaliplatin synthesis routes (PKU strength).
- 4-cell baseline data also unlocks paper §5 sub-claim (cisplatin-class
  benchmarking) before the higher-cost 协和 contract kicks in.
- Establishes a domestic Chinese wet-lab partnership before the
  hospital-phase (协和) complication (IRB / mycoplasma / GCP).

### Send 3 — LOI-协和 (NCI-60 9-cell + ICP-MS, 6-12 months)

**Why third:**
- **Hospital-phase + IRB + mycoplasma + GCP workflow** — heaviest wet-lab
  compliance overhead; want this in flight while PKU 4-cell data is rolling.
- 协和-PI optional PKU/PD xenograft (NSG 小鼠) at Phase 3 is the gold-standard
  clinical-prep ramp — worth the slower cadence.
- NCI-60 9-cell + ICP-MS uptake is the paper-§5 figure-tier data; do not
  start before PKU 4-cell baseline is in hand (so Phase-1 deliverables are
  staggered, not stacked).

### Send 4 — LOI-ORNL (protein-metal binding, 9-12 months)

**Why last:**
- **Highest risk** (AMBER license, SPR regeneration conditions, 4-protein
  expression pipeline, ORNL cluster access).
- **Longest horizon** (12 months to first K_d; joint manuscript).
- **No direct R20 paper §5 gate** — paper §5 only requires wet-lab IC50, not
  protein K_d. ORNL data is R22-P4 (TODO-97) future-work territory.
- Defer until R21-P4 triangle closure R² lift ships, AND until Rusanov /
  PKU / 协和 wet-lab data is at least Phase-1 active. The protein-K_d axis is
  the most publishable but the least time-critical for R20 ship.

## 4. Per-LOI next-action checklist (for Hugo)

### 4.1 LOI-RUSANOV (Wf-305A)

- [ ] **Look up Rusanov corresponding-author email** from the ChemRxiv v2
      submission page (DOI 10.26434/chemrxiv-2025-pp32k) — typically the
      `Corresponding author` block at the top of the ChemRxiv page. **Save
      the email** at `TODO/pending/wetlab_loi_rusanov_2025_platinai.md`
      line ~6 (`**Target PI:** ...`) — replace the placeholder with a
      concrete email.
- [ ] **Decide** send / defer / drop. Per `TODO/INDEX.md` §12.3 P3 mitigation,
      **send is the user's call** — this helper does NOT send.
- [ ] **If send:** run `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md`
      §3 (Wf-28 generic email template, WLP-OUT-001) → fill Rusanov-specific
      1-page project summary → cold-email Day 0 → Day +7 light follow-up →
      Day +21 second follow-up → Day +28 intro Zoom (if requested) → Day
      +60 data in hand.
- [ ] **If defer:** add a `**Defer-until:** YYYY-MM-DD` line and move the
      LOI back to `TODO/pending/`. **If drop:** archive to `TODO/archive/`
      with the rationale.
- [ ] After the LOI goes out, set `**Status:** SENT YYYY-MM-DD` in the LOI
      header (line 5).
- [ ] On Rusanov reply, transition through SENT → REPLIED → DECLINED /
      ACTIVE → SUCCESS (data in hand) per Wf-305A §5.

### 4.2 LOI-PKU (Wf-74 v1)

- [ ] **Identify PKU PI email** from PKU 医学部 faculty page or mutual
      contact. The LOI header (line 6) names **北京大学医学部 药物化学 /
      化学生物学 课题组** — Hugo picks the exact PI.
- [ ] **Resolve PKU vs 协和分工** — LOI-PKU §7.2 says NCI-60 9-cell is
      optional for PKU + delegated to 协和. Confirm this boundary before
      send; if PKU wants 9-cell too, rewrite §7.2 to clarify.
- [ ] **Decide** send / defer / drop (paper-§5-blocking on cisplatin 4-cell
      IC50 — recommended SEND).
- [ ] **If send:** use Wf-28 generic template (WLP-OUT-001) with the
      `wetlab_protocols/collaborator_outreach_template.md` §3 1-page
      summary. Cold-email Day 0 → +7 / +21 / +28 / +60 cadence.
- [ ] On PKU reply, transition status SENT → REPLIED → ACTIVE / DECLINED.

### 4.3 LOI-协和 (Wf-74 v1)

- [ ] **Identify 协和 PI email** from 协和医学院 faculty page or mutual
      contact. The LOI header (line 6) names **北京协和医学院 医院期细胞
      毒理学 课题组** — Hugo picks the exact PI.
- [ ] **Pre-flight 6 compliance items before send:**
  - IRB approval pathway confirmed (Phase 1 deliverable).
  - Mycoplasma test capability confirmed.
  - NCI-60 9-cell panel **all 9 cell lines** sourced and authenticated.
  - ICP-MS instrument (Thermo iCAP RQ / Agilent 7900) + microwave digestion
    (CEM MARS 6) confirmed available.
  - GCP-grade clinical-prep workflow acknowledged.
  - Case-report authorship path agreed in writing.
- [ ] **Decide** send / defer / drop (recommended DEFER until PKU 4-cell
      baseline + Rusanov NUMERIC are in hand — staggering prevents
      bandwidth collapse).
- [ ] **If send:** same Wf-28 generic template (WLP-OUT-001).
- [ ] On 协和 reply, transition SENT → REPLIED → ACTIVE / DECLINED.

### 4.4 LOI-ORNL (Wf-74 v1)

- [ ] **Identify ORNL PI email** from ORNL directory (e.g.
      https://www.ornl.gov/group/... or recent cisplatin-DNA lesion
      binding / GST / metallothionein / p53 DBD publications — Hugo picks).
- [ ] **Pre-flight 4 ORNL-specific items:**
  - AMBER license + ORNL cluster access (LOI §10 risk #1) — confirm path
    before send.
  - SPR (Biacore X100 / T200) instrument time — Phase-1 calibration.
  - 4 protein-expression pipeline (GST / cisplatin-DNA lesion / p53 DBD /
    MT-2A) — Phase-1 protein prep.
  - Joint grant fiscal-host path (NIH R01 / DOE BER / NSF) — Phase-3.
- [ ] **Decide** send / defer / drop (recommended DEFER until R21-P4
      triangle closure R² lift ships + Rusanov / PKU / 协和 wet-lab data
      is at least Phase-1 active; ORNL is R22-P4 future-work, not R20
      paper-§5-gating).
- [ ] **If send:** Wf-28 generic template (WLP-OUT-001).
- [ ] On ORNL reply, transition SENT → REPLIED → ACTIVE / DECLINED.

## 5. Cross-references

- `TODO/INDEX.md` §12.3 P3 mitigation (cold-email is user's responsibility)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule;
  outreach NOT gated but send is user's call)
- `TODO/pending/116_wf74_wetlab_outreach_partial.md` (P0 TODO — Wf-74 v2
  close-out rewrite; the **🔴 single P0 not yet shipped** per TODO/INDEX
  §873; party-twenty-spirit §3.5 violation pending fix)
- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28
  generic email template, WLP-OUT-001)
- `molmetal/docs/wetlab_protocols/binding_affinity_assay.md` (Wf-28
  generic ITC protocol, WLP-BIND-001)
- `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` (Wf-28
  data-sharing MoU template)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (companion IC50 analysis pipeline)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run)
- `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` (top-20
  candidates referenced in all 4 LOIs)
- `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` (D-MPNN
  pIC50 spread 1.7 log units referenced in all 4 LOIs)
- `TODO/completed/100_r20_r21_consolidation.md` (R20-R21 plan)
- `TODO/completed/106_r23_plan_l17_l20.md` (R23 plan, references
  ORNL/PKU/协和 outreach 0 progress today — TODO-97 P6.1)
- `molmetal/reports/wf295b_platinai_loader/loader_report.md` (Wf-295B
  coverage-only MB-Finder harvest)
- `molmetal/reports/wf304b_selenium/selenium_report.md` (Wf-304B
  Selenium UC + ChemRxiv CF bypass — BLOCK_CF, 0 NUMERIC labels harvested)
- `molmetal/scripts/platinai_real_label_loader.py` (Wf-295B + Wf-301C
  stratified MB-Finder coverage loader — currently coverage-only)

## 6. Honest framing (party-twenty-spirit §3.5)

- This is a **review helper**, NOT an LOI send tool. **No send happens.**
- **0 LOI sent at this revision** (2026-09-29). Send is the user's call.
- All 4 LOI drafts are pre-PI-review; none have been forwarded to a
  scientific reader outside the project.
- **Recommended send order is a recommendation, not a mandate.** Hugo may
  reorder per strategic preference (e.g. prioritize hospital-phase over
  data-collaboration).
- The LOI-RUSANOV send does NOT depend on paper §5 authorization per
  TODO-84 — outreach is NOT gated. **Send if/when Hugo chooses.**
- The LOI-ORNL protein-K_d axis is **not R20 paper-§5-gating**; defer is
  risk-free for paper timeline.
- Wet-lab pipeline (`template_ic50_pipeline.py`) is ready to ingest NUMERIC
  CSV / IC50 curves the moment wet-lab data ships — no additional plumbing
  needed.

## 7. Revision history

| Date | Author | Change |
|---|---|---|
| 2026-09-29 | Wf-308D PART 2 (Claude) | Initial 4-LOI consolidation review helper drafted (Wf-308D workflow). |