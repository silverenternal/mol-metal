# Wf-310G — 4-LOI Send-Ready Package (consolidation + send protocol)

**Document ID:** TODO-310G / Wf-310G
**Version:** 1.0 (send-ready package, NOT a sent letter)
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`
**Scope:** Send-ready consolidation across 4 LOI drafts + cold-email sending
protocol. **NO send happens** — cold-email is user's responsibility per
`TODO/pending/84_no_paper_task_without_authorization.md` paper rule and
`TODO/INDEX.md` §12.3 P3 mitigation.

---

## 1. Why this package exists

As of 2026-09-29 there are **4 LOI drafts** in `TODO/pending/`:

1. `wetlab_loi_ornl_protein_metal.md` (Wf-74 / WLP-OUT-ORNL)
2. `wetlab_loi_pku_cisplatin_ic50.md` (Wf-74 / WLP-OUT-PKU)
3. `wetlab_loi_xiehe_nci60.md` (Wf-74 / WLP-OUT-XIEHE)
4. `wetlab_loi_rusanov_2025_platinai.md` (Wf-305A NEW; WLP-OUT-RUSANOV)

This Wf-310G package consolidates the 4 LOIs into a single review-and-send
surface so Hugo can decide send order, fill in the missing PI emails, and tick
off per-LOI next-action items. Per `TODO/INDEX.md` §12.3 P3 mitigation, Hugo
personally coordinates the cold-email send and the PI outreach is **NOT**
delegated to subagents / scripts.

## 2. Audit checklist (per-LOI verification)

| Check item                          | LOI-ORNL                       | LOI-PKU                        | LOI-协和                       | LOI-RUSANOV                    |
|-------------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| 1. Subject line (clear + specific)  | DRAFT (see cold-email §1)      | DRAFT (see cold-email §1)      | DRAFT (see cold-email §1)      | READY ("PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)") |
| 2. Cold-email body 200–300 words    | DRAFT (template §2)            | DRAFT (template §2)            | DRAFT (template §2)            | DRAFT (template §2)            |
| 3. PI contact info                  | **TBD — Hugo to look up** (ORNL directory, recent cisplatin-DNA / GST / MT / p53 DBD pubs) | **TBD — Hugo to look up** (PKU 医学部 faculty page or mutual contact) | **TBD — Hugo to look up** (协和医学院 faculty page) | **TBD — Hugo to look up** (ChemRxiv v2 submission metadata, DOI 10.26434/chemrxiv-2025-pp32k) |
| 4. 16-section LOI body              | 15 sections parallel           | 15 sections parallel           | 16 sections parallel (+ICP-MS panel) | 16 sections parallel (+Why-we-need +Data-scope) |
| 5. Asks + Offers + Timeline + Risk  | OK (asks §4 + offers §5 + timeline §9 + risk §10) | OK (asks §4 + offers §5 + timeline §9 + risk §10) | OK (asks §4 + offers §5 + timeline §10 + risk §11) | OK (asks §5 + offers §6 + timeline §10 + risk §11) |
| 6. Attachments (1-pager + refs)     | OK (see §3 below)              | OK (see §3 below)              | OK (see §3 below)              | OK (see §3 below)              |
| 7. Signature (Hugo Lee + title + ORCID) | **TBD ORCID**                 | **TBD ORCID**                 | **TBD ORCID**                 | **TBD ORCID**                 |

**Honest verdict (Wf-310G audit):**

- All 4 LOIs are **internally consistent** (parallel structure, complete
  ask/offer/timeline/risk).
- All 4 LOIs are **send-ready in draft form**: subject line, body
  template, attachments, and signature line are now standardised
  (this package + `wetlab_cold_email_template.md`).
- All 4 LOIs are **BLOCKED on actual send** by exactly two missing fields:
  (a) PI email (4 × TBD, Hugo's responsibility) and
  (b) Hugo's ORCID (1 × TBD, fill once in cold-email §4 signature block).

## 3. 4-row LOI summary table

| # | File (path) | Target PI / Lab | Status | What's asked | What's offered |
|---|---|---|---|---|---|
| 1 | `wetlab_loi_rusanov_2025_platinai.md` (Wf-305A) | Rusanov 2025 corresponding author (PlatinAI / MB-Finder portal maintainer); ChemRxiv DOI 10.26434/chemrxiv-2025-pp32k | DRAFT v1 — body + subject READY | Per-cell-line NUMERIC IC50 release for ~200 Pt compounds across MCF7 / A549 / A2780 / A2780cis (Rusanov 2025 Table S4-S5 supplementary xlsx/CSV) | mid-author + co-corresponding authorship on Mol-Metal paper §5; CC-BY release of `platinai_4cell_real_ic50.csv`; prominent Acknowledgement naming Rusanov team |
| 2 | `wetlab_loi_pku_cisplatin_ic50.md` (Wf-74 v1) | 北京大学医学部 (Peking University Health Science Center) 药物化学 / 化学生物学 课题组 | DRAFT v1 — body + subject DRAFTED | Cisplatin-class 4-cell IC50 validation (MCF7 / A549 / A2780 / A2780cis) on top-20 R20-P6 candidates (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11); optional NCI-60 9-cell extension | mid-author + co-corresponding on data-validation section; joint manuscript; model retrain on measured IC50 |
| 3 | `wetlab_loi_xiehe_nci60.md` (Wf-74 v1) | 北京协和医学院 (Peking Union Medical College) 医院期细胞毒理学 课题组 | DRAFT v1 — body + subject DRAFTED | NCI-60 9-cell **完整 panel** (SRB/MTT) + ICP-MS cellular uptake on top-20 candidates; lead compound 临床转化 prep | co-corresponding on data-validation section + cover-sheet hospital name; mid-author on partial 9-cell |
| 4 | `wetlab_loi_ornl_protein_metal.md` (Wf-74 v1) | Oak Ridge National Laboratory (ORNL) protein-metal binding group (SPR / ITC) | DRAFT v1 — body + subject DRAFTED | SPR (Biacore) + ITC (MicroCal PEAQ) K_d validation on top-20 candidates × 4 proteins (GST / cisplatin-DNA lesion / p53 DBD / metallothionein MT-2A) | co-corresponding + mid-author; joint grant (NIH R01 / DOE BER / NSF) — ORNL as fiscal host |

**Time / risk / paper-§5-gating summary:**

| # | LOI | Time horizon | Risk tier | Paper §5 gating? |
|---|---|---|---|---|
| 1 | LOI-RUSANOV | 2–6 wk | Low (data-only) | YES — closes Wf-297B §6 NUMERIC gap |
| 2 | LOI-PKU | 6–12 mo | Medium (synth + 4-cell) | YES — cisplatin-class 4-cell IC50 |
| 3 | LOI-协和 | 6–12 mo | High (IRB + 9-cell + ICP-MS) | YES — NCI-60 9-cell + ICP-MS uptake |
| 4 | LOI-ORNL | 9–12 mo | Highest (AMBER + SPR + 4-protein) | NO — protein K_d is R22-P4 future-work |

**Recommended send order (low-friction → high-friction):**
**Send 1 → LOI-RUSANOV** (data-only, 2–6 wk, closes paper §5 NUMERIC gap).
**Send 2 → LOI-PKU** (4-cell cisplatin baseline, lower cell-line complexity).
**Send 3 → LOI-协和** (NCI-60 9-cell + ICP-MS, IRB + hospital-phase).
**Send 4 → LOI-ORNL** (protein K_d, defer until R21-P4 ships — NOT R20-paper-§5-gating).

## 4. Cold-email sending protocol

### 4.1 Files referenced

- **Send-ready package** (this file): `TODO/pending/wetlab_loi_send_ready_package.md`
- **Cold-email template** (subject + body + signature): `TODO/pending/wetlab_cold_email_template.md`
- **Generic outreach protocol** (institutional): `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (WLP-OUT-001, Wf-28)
- **Follow-up cadence**: §5 of `collaborator_outreach_template.md` (Day 0 / +7 / +21 / +28 / +60 / Month 2 / Month 6 / Month 9)
- **Per-LOI long-form body**: the 4 `wetlab_loi_*.md` files themselves (attached as PDF).

### 4.2 Pre-send checklist (Hugo runs once per LOI)

1. **PI email**: look up the PI email from the source named in §3 row.
   Update `**Target PI:** ...` line in the LOI header (line ~6 of each
   LOI) with the concrete email + name. This is the **single biggest blocker** —
   4 LOIs are blocked on this.
2. **Subject line**: copy the LOI-specific subject from
   `wetlab_cold_email_template.md` §1.
3. **Body customization**: copy the LOI-specific body from
   `wetlab_cold_email_template.md` §2, then fill the 4 placeholder
   fields: `{PI_name}`, `{institution}`, `{specific_ask_2_sentences}`,
   `{specific_offer_2_sentences}`.
4. **Word count check**: the customized body should land in **200–300 words**.
   Trim any one paragraph if it exceeds.
5. **Attachments**: see §3 of `wetlab_cold_email_template.md`. All 4 LOIs
   share the same 4 attachments, with 1 LOI-specific 5th attachment per row.
6. **Signature**: copy from `wetlab_cold_email_template.md` §4. **ORCID is
   the only Hugo-personal field** — fill once and reuse for all 4 sends.
7. **CC**: see §4.4 below. CC-list is LOI-specific.
8. **Send Day 0**: log the send in
   `TODO/pending/wetlab_outreach_log.csv` (TODO — Hugo to create or
   confirm column header). Update LOI header line 5 from
   `Status: DRAFT — not sent` → `Status: SENT YYYY-MM-DD`.
9. **Day +7 light follow-up**: short ping ("checking if my note landed,
   happy to Zoom if useful").
10. **Day +21 second follow-up**: substantive follow-up with 1 new
    paragraph (e.g. "since my first note, we've also shipped
    `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` —
    attaching snapshot").
11. **Day +28 intro Zoom (if requested)**: 30 min, agenda in
    `collaborator_outreach_template.md` §2 framework.
12. **Day +60**: if no reply, mark `**Status:** DECLINED YYYY-MM-DD` and
    move to next PI on the fallback list.

### 4.3 Pre-flight checks (LOI-specific blockers)

- **LOI-RUSANOV**: pre-flight 0 (pure data request, no compliance gates).
- **LOI-PKU**: pre-flight 1 item — confirm PKU vs 协和 NCI-60 division of
  labour (LOI-PKU §7.2 says NCI-60 9-cell is optional for PKU + delegated to 协和).
  If PKU wants 9-cell too, rewrite §7.2 boundary before send.
- **LOI-协和**: pre-flight 6 compliance items (per Wf-308D §4.3):
  IRB pathway + mycoplasma test + NCI-60 9-cell sourcing + ICP-MS instrument
  (Thermo iCAP RQ / Agilent 7900) + CEM MARS 6 microwave + GCP-grade workflow.
  These are Phase-1 deliverables (per LOI §4) but Hugo should verbally confirm
  fit on intro Zoom (Day +28) before MoU.
- **LOI-ORNL**: pre-flight 4 ORNL-specific items (per Wf-308D §4.4):
  AMBER license + ORNL cluster access + SPR (Biacore X100/T200) instrument
  time + 4-protein expression pipeline. These are Phase-1 but Hugo should
  verbally confirm on intro Zoom.

### 4.4 CC + cover sheet protocol

| LOI | Primary recipient | CC | Cover sheet |
|---|---|---|---|
| LOI-RUSANOV | Rusanov 2025 corresponding author | none (single-author email is professional norm for data request) | none |
| LOI-PKU | PKU 医学部 药物化学 / 化学生物学 PI | Hugo's mutual contact at PKU (TBD — Hugo to identify) | none |
| LOI-协和 | 协和医学院 医院期细胞毒理学 PI | Hugo's mutual contact at 协和 (TBD) + 协和 科研处 联系人 (TBD) | none |
| LOI-ORNL | ORNL protein-metal binding PI | Hugo's mutual contact at ORNL (TBD — Hugo to identify via Cn/journals or ORNL directory) | none |

**Subject prefix discipline**: All 4 subjects start with the specific ask
noun (`PlatinAI NUMERIC IC50`, `顺铂 4-cell IC50`, `NCI-60 9-cell + ICP-MS`,
`SPR + ITC K_d`), so the recipient can scan-triage at inbox-zero depth.

### 4.5 Status state machine (per LOI)

```
DRAFT (current state, all 4)
  └─→ READY_TO_SEND (after Hugo fills PI email + ORCID; 0 of 4 today)
       └─→ SENT (Day 0 cold-email send)
            ├─→ REPLIED (Day +1 to +60) → ACTIVE (intro Zoom + MoU signed)
            │                              └─→ SUCCESS (data / Phase-1 in hand)
            │                              └─→ DECLINED (Phase-1 lost)
            └─→ DECLINED (no reply by Day +60)
```

## 5. Per-LOI next-action checklist (for Hugo)

### 5.1 LOI-RUSANOV (Wf-305A) — RECOMMENDED SEND #1

- [ ] **Look up Rusanov corresponding-author email** from ChemRxiv v2
      submission metadata (DOI 10.26434/chemrxiv-2025-pp32k).
      Replace `**Target PI:**` placeholder line ~6 of
      `wetlab_loi_rusanov_2025_platinai.md` with the concrete email + name.
- [ ] **Fill Hugo's ORCID** in the cold-email signature block
      (`wetlab_cold_email_template.md` §4).
- [ ] **Customize cold-email body**: use §2 RUSANOV-specific template
      (200–300 words target).
- [ ] **Decide** send / defer / drop. Recommended: SEND (closes Wf-297B §6
      NUMERIC gap for paper §5 R20 ship).
- [ ] **If send**: run cold-email §1 subject, attach 5 attachments
      (LOI-RUSANOV §3 of `wetlab_cold_email_template.md`),
      send Day 0, log in `wetlab_outreach_log.csv`.
- [ ] On Rusanov reply, transition DRAFT → SENT → REPLIED → ACTIVE / DECLINED.

### 5.2 LOI-PKU (Wf-74 v1) — RECOMMENDED SEND #2

- [ ] **Identify PKU PI email** from PKU 医学部 faculty page or mutual
      contact. The LOI header (line 6) names **北京大学医学部 药物化学 /
      化学生物学 课题组** — Hugo picks the exact PI.
- [ ] **Resolve PKU vs 协和 NCI-60 division of labour** — LOI-PKU §7.2
      boundary (optional NCI-60 9-cell). Confirm with PI on intro Zoom
      if PI wants 9-cell too, rewrite §7.2 to clarify.
- [ ] **Fill Hugo's ORCID** (already in cold-email template; reuse from
      RUSANOV send).
- [ ] **Customize cold-email body**: use §2 PKU-specific template (中文).
- [ ] **Decide** send / defer / drop. Recommended: SEND (paper-§5-blocking
      on cisplatin 4-cell IC50).
- [ ] **If send**: use 中文 cold-email subject from §1 PKU row, attach 4
      attachments (LOI-PKU §3 of cold-email template).
- [ ] On PKU reply, transition status DRAFT → SENT → REPLIED → ACTIVE / DECLINED.

### 5.3 LOI-协和 (Wf-74 v1) — RECOMMENDED SEND #3

- [ ] **Identify 协和 PI email** from 协和医学院 faculty page or mutual
      contact. The LOI header (line 6) names **北京协和医学院 医院期
      细胞毒理学 课题组** — Hugo picks the exact PI.
- [ ] **Pre-flight 6 compliance items** (per §4.3 above) — verbally
      confirm fit on intro Zoom before MoU.
- [ ] **Fill Hugo's ORCID** (reuse).
- [ ] **Customize cold-email body**: use §2 协和-specific template (中文).
- [ ] **Decide** send / defer / drop. Recommended: DEFER until PKU 4-cell
      baseline + Rusanov NUMERIC are in hand (staggering prevents bandwidth
      collapse).
- [ ] **If send**: use 中文 cold-email subject from §1 协和 row, attach 4
      attachments (LOI-协和 §3 of cold-email template).
- [ ] On 协和 reply, transition DRAFT → SENT → REPLIED → ACTIVE / DECLINED.

### 5.4 LOI-ORNL (Wf-74 v1) — RECOMMENDED SEND #4 (DEFER)

- [ ] **Identify ORNL PI email** from ORNL directory
      (https://www.ornl.gov/group/... or recent cisplatin-DNA lesion
      binding / GST / metallothionein / p53 DBD publications).
- [ ] **Pre-flight 4 ORNL-specific items** (per §4.3 above).
- [ ] **Fill Hugo's ORCID** (reuse).
- [ ] **Customize cold-email body**: use §2 ORNL-specific template (English).
- [ ] **Decide** send / defer / drop. Recommended: **DEFER** until
      R21-P4 triangle closure R² lift ships + Rusanov / PKU / 协和
      wet-lab data is at least Phase-1 active. ORNL protein-K_d axis
      is R22-P4 future-work (TODO-97 P6.1), NOT R20 paper-§5-gating.
- [ ] **If send**: use English cold-email subject from §1 ORNL row,
      attach 4 attachments (LOI-ORNL §3 of cold-email template).
- [ ] On ORNL reply, transition DRAFT → SENT → REPLIED → ACTIVE / DECLINED.

## 6. Honest framing (party-twenty-spirit §3.5)

- This is a **send-ready consolidation**, NOT a send tool. **0 LOI sent at
  this revision** (2026-09-29). Cold-email is user's responsibility per
  `TODO/INDEX.md` §12.3 P3 mitigation.
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
- All 4 LOIs are now **send-ready in draft form**; the only remaining
  blockers for actual send are (a) PI email lookup (4 × TBD) and
  (b) Hugo's ORCID (1 × TBD, fill once and reuse).

## 7. Cross-references

- `TODO/pending/wetlab_cold_email_template.md` — concrete cold-email
  subject + body + signature template (companion to this file, Wf-310G PART 2)
- `TODO/pending/wetlab_loi_consolidation_review.md` — Wf-308D PART 2
  pre-existing 4-row summary table (this file supersedes / refines it
  with the audit checklist + send protocol)
- `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` (Wf-305A LOI)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` (Wf-74 LOI)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` (Wf-74 LOI)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` (Wf-74 LOI)
- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28
  generic email template, WLP-OUT-001)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule;
  outreach NOT gated but send is user's call)
- `TODO/pending/116_wf74_wetlab_outreach_partial.md` (P0 TODO — Wf-74 v2
  close-out rewrite; the **🔴 single P0 not yet shipped** per TODO/INDEX
  §873; party-twenty-spirit §3.5 violation pending fix)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run)
- `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` (top-20
  candidates referenced in all 4 LOIs)
- `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` (D-MPNN
  pIC50 spread 1.7 log units referenced in all 4 LOIs)

## 8. Revision history

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-29 | Wf-310G | Initial 4-LOI send-ready package (consolidation + send protocol + per-LOI checklist). **0 LOI sent.** |
