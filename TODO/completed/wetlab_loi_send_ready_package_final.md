# Wf-313B — 4-LOI Send-Ready Package FINAL + Hugo ORCID Placeholder

**Document ID:** TODO-313B PART 2 / Wf-313B
**Version:** 1.0 (FINAL consolidation, NOT a sent letter)
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`
**Scope:** FINAL consolidation of 4 LOI drafts (Rusanov / PKU / 协和 / ORNL)
into a send-ready surface. **NO send happens** — cold-email is user's
responsibility per `TODO/pending/84_no_paper_task_without_authorization.md`
paper rule and `TODO/INDEX.md` §12.3 P3 mitigation.

---

## 1. Why this FINAL package exists

Wf-313B PART 2 closes the Wf-310G PART 1 + Wf-308D PART 2 chain with a
**single review-and-send surface** for Hugo Lee. It consolidates:

- **Wf-308D PART 2** (`TODO/pending/wetlab_loi_consolidation_review.md`) —
  the 4-row summary table + send-order ladder.
- **Wf-310G PART 1** (`TODO/pending/wetlab_loi_send_ready_package.md`) —
  the audit checklist + sending protocol.
- **Wf-310G PART 2** (`TODO/pending/wetlab_cold_email_template.md`) —
  the cold-email subject + body templates + signature block.
- **4 LOI drafts** (`TODO/pending/wetlab_loi_*.md`) — the long-form asks.

This FINAL package provides:
1. **Send order**: low-friction → high-friction ladder (Rusanov → PKU → 协和 → ORNL).
2. **Per-LOI checklist**: 6 items per LOI Hugo ticks off before sending.
3. **4 PI email placeholders**: **TBD — Hugo to fill from named source**.
4. **Hugo ORCID placeholder**: `0000-0000-0000-0000` (TBD — Hugo to fill once and reuse).

---

## 2. Send order (low-friction → high-friction)

The order below is the **recommended ladder** (per Wf-308D §3 + Wf-310G §3
cohort reasoning). Hugo may reorder per strategic preference.

| # | LOI | Time | Risk tier | Paper §5 gating | PI Email |
|---|---|---|---|---|---|
| **Send 1** | **LOI-RUSANOV** (PlatinAI / MB-Finder, data-only) | 2–6 wk | Low | **YES** (closes Wf-297B §6 NUMERIC gap) | **TBD — Hugo** |
| **Send 2** | **LOI-PKU** (cisplatin 4-cell) | 6–12 mo | Medium | **YES** (cisplatin-class 4-cell IC50) | **TBD — Hugo** |
| **Send 3** | **LOI-协和** (NCI-60 9-cell + ICP-MS) | 6–12 mo | High | **YES** (NCI-60 9-cell + uptake) | **TBD — Hugo** |
| **Send 4** | **LOI-ORNL** (protein K_d, SPR + ITC) | 9–12 mo | Highest | **NO** (R22-P4 future-work) | **TBD — Hugo** |

**Per-party send timing rationale** (per Wf-308D §3):

- **Send 1 → LOI-RUSANOV first** because it is a pure data request (no wet-lab
  cost, no facility, no funding exchange). Closes the **Pearson r ≥ 0.7**
  honesty gap for paper §5; 2-6 wk turnaround matches R20 timeline; high
  data-value-per-unit-effort.
- **Send 2 → LOI-PKU second** because cisplatin / carboplatin / oxaliplatin
  synthesis routes are well-established at PKU; 4-cell baseline unblocks
  paper §5 sub-claim before the higher-cost 协和 contract kicks in.
- **Send 3 → LOI-协和 third** because IRB + mycoplasma + GCP compliance
  overhead is the heaviest; want this in flight while PKU 4-cell data is
  rolling. NCI-60 9-cell + ICP-MS is the paper-§5 figure-tier data.
- **Send 4 → LOI-ORNL last** because AMBER license + SPR regeneration
  conditions + 4-protein expression pipeline + ORNL cluster access is
  the highest-risk axis. ORNL data is **NOT** R20-paper-§5-gating; defer
  until R21-P4 triangle closure R² lift ships.

---

## 3. 4 PI email placeholders (TBD — Hugo to fill)

**Critical reminder**: per `TODO/INDEX.md` §12.3 P3 mitigation, the PI
outreach is **NOT** delegated to subagents / scripts. Hugo personally
fills the 4 PI emails from the named source below. Once filled, the
emails live in the LOI header (line ~6 of each `wetlab_loi_*.md` file).

| LOI | PI / Lab | Email placeholder | Source for Hugo to look up |
|---|---|---|---|
| LOI-RUSANOV | Rusanov 2025 corresponding author (PlatinAI / MB-Finder portal maintainer) | `rusanov.{corresponding_author}@mb-finder.org` **[TBD — Hugo]** | ChemRxiv v2 submission page (DOI 10.26434/chemrxiv-2025-pp32k) — `Corresponding author` block at the top |
| LOI-PKU | 北京大学医学部 药物化学 / 化学生物学 课题组 PI | `pi.{pku_med_chem}@bjmu.edu.cn` **[TBD — Hugo]** | PKU 医学部 faculty page or mutual contact |
| LOI-协和 | 北京协和医学院 医院期细胞毒理学 课题组 PI | `pi.{xiehe_tox}@pumc.edu.cn` **[TBD — Hugo]** | 协和医学院 faculty page or mutual contact |
| LOI-ORNL | ORNL protein-metal binding group PI | `pi.{ornl_binding}@ornl.gov` **[TBD — Hugo]** | ORNL directory (https://www.ornl.gov/group/...) or recent cisplatin-DNA lesion / GST / MT / p53 DBD publications |

---

## 4. Hugo ORCID placeholder

Hugo's ORCID must be filled **once** and reused across all 4 LOIs + the
cold-email signature block (per `wetlab_cold_email_template.md` §4 +
§4.1 中文 signature). The placeholder below is set to the **ORCID
all-zeros placeholder** (per ISO 27723 ID convention) and must be
replaced before any send:

```
ORCID: 0000-0000-0000-0000  [PLACEHOLDER — Hugo to replace with real ORCID]
```

**Real ORCID format**: `0000-0000-0000-0000` (16 digits, ISO 27729).
Hugo registers at https://orcid.org/register if not already registered;
the ORCID then lives in the cold-email signature block (English §4 and
中文 §4.1 of `wetlab_cold_email_template.md`) and in the cover-sheet
author block of the Mol-Metal paper §5.

**Workflow once Hugo has the real ORCID**:
1. Replace `0000-0000-0000-0000` in this file (§4).
2. Replace `0000-0000-0000-0000` in `TODO/pending/wetlab_cold_email_template.md` §4 + §4.1.
3. (Optional) Replace in each `wetlab_loi_*.md` if Hugo wants the ORCID on the cover page.

---

## 5. Per-LOI checklist (Hugo ticks off before each send)

### 5.1 LOI-RUSANOV checklist

- [ ] **PI email**: look up from ChemRxiv v2 submission metadata (DOI
      10.26434/chemrxiv-2025-pp32k). Save the email at
      `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` line 6.
- [ ] **Hugo ORCID**: replace the `0000-0000-0000-0000` placeholder in
      `wetlab_cold_email_template.md` §4 (English signature).
- [ ] **Subject line**: copy verbatim from `wetlab_cold_email_template.md`
      §1.1 ("PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)").
- [ ] **Body customization**: copy from §2.1; fill the 4 placeholder
      fields: `{Rusanov_corresponding_author_name}`,
      `{institution}`, `{specific_ask_2_sentences}`, `{specific_offer_2_sentences}`.
- [ ] **Word count**: body lands in 200–300 words (target ~270).
- [ ] **Attachments**: §3 — 4 common (1-pager + LOI PDF + top-20 CSV +
      D-MPNN pIC50 snapshot) + 1 RUSANOV-specific (Wf-295B MB-Finder
      coverage matrix report PDF).
- [ ] **Send Day 0**: log the send in `TODO/pending/wetlab_outreach_log.csv`
      (TODO — Hugo to create with columns:
      `loi_id, target_pi, send_date, status, reply_date, reply_type, notes`).
      Update LOI header line 5 from `DRAFT — not sent` → `SENT YYYY-MM-DD`.
- [ ] **Day +7 light follow-up** + **Day +21 second follow-up** per
      `wetlab_cold_email_template.md` §6 cadence.

### 5.2 LOI-PKU checklist

- [ ] **PI email**: look up from PKU 医学部 faculty page (药物化学 / 化生
      物学 课题组). Save at `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md`
      line 6.
- [ ] **Hugo ORCID**: already filled (after LOI-RUSANOV first send).
- [ ] **Subject line**: copy from `wetlab_cold_email_template.md` §1.2
      (主题：顺铂类 4-cell IC50 湿实验验证合作邀请 — Mol-Metal R18 + PKU 药物化学).
- [ ] **Body customization**: copy from §2.2 (中文); fill 4 placeholders.
- [ ] **Word count**: body lands in 200–300 字 (target ~290 字).
- [ ] **Resolve PKU vs 协和分工**: LOI-PKU §7.2 says NCI-60 9-cell is
      optional for PKU + delegated to 协和. Confirm before send.
- [ ] **Attachments**: 4 common + 1 PKU-specific (NCI-60 9-cell sourcing
      matrix — Hugo to draft).
- [ ] **Send Day 0** + log + cadence.

### 5.3 LOI-协和 checklist

- [ ] **PI email**: look up from 协和医学院 faculty page (医院期细胞毒理学).
      Save at `TODO/pending/wetlab_loi_xiehe_nci60.md` line 6.
- [ ] **Pre-flight 6 compliance items before send** (per Wf-308D §4.3):
  - [ ] IRB approval pathway confirmed (Phase 1 deliverable).
  - [ ] Mycoplasma test capability confirmed.
  - [ ] NCI-60 9-cell panel **all 9 cell lines** sourced and authenticated.
  - [ ] ICP-MS instrument (Thermo iCAP RQ / Agilent 7900) + microwave
        digestion (CEM MARS 6) confirmed available.
  - [ ] GCP-grade clinical-prep workflow acknowledged.
  - [ ] Case-report authorship path agreed in writing.
- [ ] **Decision: send vs defer** (Wf-308D §3 recommends DEFER until
      PKU 4-cell baseline + Rusanov NUMERIC are in hand).
- [ ] **Subject line**: copy from §1.3 (主题：NCI-60 9-cell panel +
      ICP-MS 摄取合作邀请 — Mol-Metal R18 + 协和医院期细胞毒理学).
- [ ] **Body customization**: copy from §2.3 (中文); fill 4 placeholders.
- [ ] **Attachments**: 4 common + 1 协和-specific (NCI-60 9-cell sourcing
      matrix + ICP-MS instrument spec sheet — Hugo to draft).
- [ ] **Send Day 0** + log + cadence.

### 5.4 LOI-ORNL checklist

- [ ] **PI email**: look up from ORNL directory (protein-metal binding
      group) or recent cisplatin-DNA lesion / GST / MT / p53 DBD
      publications. Save at `TODO/pending/wetlab_loi_ornl_protein_metal.md`
      line 6.
- [ ] **Pre-flight 4 ORNL-specific items** (per Wf-308D §4.4):
  - [ ] AMBER license + ORNL cluster access (LOI §10 risk #1).
  - [ ] SPR (Biacore X100 / T200) instrument time — Phase-1 calibration.
  - [ ] 4 protein-expression pipeline (GST / cisplatin-DNA lesion /
        p53 DBD / MT-2A) — Phase-1 protein prep.
  - [ ] Joint grant fiscal-host path (NIH R01 / DOE BER / NSF) — Phase-3.
- [ ] **Decision: send vs defer** (Wf-308D §3 recommends DEFER until
      R21-P4 triangle closure R² lift ships + Rusanov / PKU / 协和 wet-lab
      data is at least Phase-1 active).
- [ ] **Subject line**: copy from §1.4 ("Protein-metal binding validation
      collaboration request (Mol-Metal R18 + ORNL SPR/ITC)").
- [ ] **Body customization**: copy from §2.4 (English); fill 4 placeholders.
- [ ] **Attachments**: 4 common + 1 ORNL-specific (4-protein sourcing matrix
      — Hugo to draft).
- [ ] **Send Day 0** + log + cadence.

---

## 6. Audit-trail summary (per Wf-310G §2)

| Audit check | Status |
|---|---|
| 1. Subject line (clear + specific) | READY (4 LOIs; verbatim in `wetlab_cold_email_template.md` §1) |
| 2. Cold-email body 200–300 words | READY (templates §2 with 4 placeholders to fill) |
| 3. PI contact info | **BLOCKED on Hugo lookup (4 × TBD)** |
| 4. 16-section LOI body | READY (4 LOIs internally consistent) |
| 5. Asks + Offers + Timeline + Risk | READY (parallel structure) |
| 6. Attachments (1-pager + refs) | **BLOCKED on 1-page project summary PDF (TODO — Hugo)** |
| 7. Signature (Hugo Lee + ORCID) | **BLOCKED on Hugo's real ORCID (placeholder: 0000-0000-0000-0000)** |

**Audit items summary** (per Wf-313B §OUTPUT): **3 audit items BLOCKED on Hugo lookup**: (a) 4 PI emails, (b) Hugo's ORCID, (c) 1-page project summary PDF.

---

## 7. Cross-references

- Wf-308D PART 2 — `TODO/pending/wetlab_loi_consolidation_review.md`
- Wf-310G PART 1 — `TODO/pending/wetlab_loi_send_ready_package.md` (this is the v1 predecessor)
- Wf-310G PART 2 — `TODO/pending/wetlab_cold_email_template.md`
- 4 LOI drafts — `TODO/pending/wetlab_loi_*.md`
- Generic email template — `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (WLP-OUT-001, Wf-28)
- Data-sharing MoU template — `molmetal/docs/wetlab_protocols/data_sharing_agreement.md`
- 3 institutional SOPs — `TODO/pending/wetlab_sop_*.md` (Wf-313B PART 1 enhanced)
- IC50 analysis pipeline — `TODO/pending/wetlab_ic50_analysis_pipeline.md` + `molmetal/scripts/template_ic50_pipeline.py`
- Wf-58 P6 dry-run — `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` (top-20 referenced in all 4 LOIs)
- R20-P6 plan — `TODO/pending/97_r20_p6_wetlab_ic50.md`
- Wf-58 ship — `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md`
- Paper rule — `TODO/pending/84_no_paper_task_without_authorization.md` (outreach NOT gated; send is user's call)
- P0 unblocking — `TODO/pending/116_wf74_wetlab_outreach_partial.md` (Wf-74 v2 close-out rewrite — single P0 not yet shipped)

---

## 8. Honest framing (party-twenty-spirit §3.5)

- This package is a **send-ready consolidation**, NOT a sent letter.
- **0 LOI sent at this revision** (2026-09-29). Send is the user's call.
- All 4 LOI drafts are pre-PI-review; none have been forwarded to a
  scientific reader outside the project.
- The recommended send order is a recommendation, not a mandate.
- **Send is NOT gated by TODO-84 paper rule** (per `TODO/pending/84_no_paper_task_without_authorization.md` — outreach is NOT gated; only paper §1-§7 work is gated).
- Wet-lab pipeline (`template_ic50_pipeline.py`) is ready to ingest NUMERIC
  CSV / IC50 curves the moment wet-lab data ships.
- The ORCID `0000-0000-0000-0000` is a **placeholder**, not a real ORCID.
  Hugo must replace before send.

---

## 9. Revision history

| Rev | Date | Author | Notes |
|---|---|---|---|
| 1.0 | 2026-09-29 | Wf-313B PART 2 (Claude) | Final consolidation of Wf-308D + Wf-310G + 4 LOIs into a single review-and-send surface; 4 PI email placeholders (TBD — Hugo); Hugo ORCID placeholder `0000-0000-0000-0000` (TBD). |