# Wf-310G — Wet-Lab LOI Cold-Email Template (4-LOI variants)

**⚠️ DEPRECATED — DO NOT USE (2026-09-30) ⚠️** — see AGENTS.md §5.4 + §10 + 2026-09-30 user reinforcement "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了". This cold-email template is preserved as historical artifact only; any agent action attempting to use this template for active outreach must REFUSE per AGENTS.md §10.3.

**Document ID:** TODO-310G PART 2 / Wf-310G
**Version:** 1.0 (concrete cold-email template, NOT a sent email)
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`
**Scope:** Cold-email subject line variants + 3-paragraph body template
(English + 中文) + LOI-specific attachments + signature block, for the
4 LOI drafts in `TODO/pending/wetlab_loi_*.md`. **NO send happens** —
cold-email is user's responsibility per `TODO/pending/84_no_paper_task_without_authorization.md`
paper rule and `TODO/INDEX.md` §12.3 P3 mitigation.

---

> **TEMPLATE_NOT_NEEDED_FOR_TIER2 (2026-09-29)**
>
> Per AGENTS.md §5.4 user directive "我们发二区就可以了，不用规划湿实验",
> Tier-2 journal is sufficient and wet-lab validation is OUT OF SCOPE.
> This cold-email template is preserved as historical reference but NOT
> for sending.

---

## 1. Subject line variants (4 LOIs)

Subject discipline: every subject starts with the **specific ask noun**
(platinum / cisplatin / NCI-60 / SPR) so the recipient can scan-triage at
inbox-zero depth.

### 1.1 LOI-RUSANOV — PlatinAI NUMERIC IC50 release (data-only, English)

> **Subject:** PlatinAI 4-cell × 200 comp NUMERIC IC50 collaboration request (Mol-Metal)

Rationale: this subject is **already present** in `wetlab_loi_rusanov_2025_platinai.md`
line 9 (`**Subject (for the cold-email):** ...`). Wf-310G inherits it
verbatim — no change needed.

### 1.2 LOI-PKU — cisplatin-class 4-cell IC50 validation (中文)

> **主题：** 顺铂类 4-cell IC50 湿实验验证合作邀请 — Mol-Metal R18 + PKU 药物化学

Rationale: noun-first ("顺铂类 4-cell IC50"), institution-second ("PKU 药物化学"),
project-third ("Mol-Metal R18"). Mirrors the structure of the
institutional `collaborator_outreach_template.md` §4 中文 template.

### 1.3 LOI-协和 — NCI-60 9-cell + ICP-MS (中文)

> **主题：** NCI-60 9-cell panel + ICP-MS 摄取合作邀请 — Mol-Metal R18 + 协和医院期细胞毒理学

Rationale: noun-first ("NCI-60 9-cell panel + ICP-MS 摄取"), institution-second
("协和医院期细胞毒理学"), project-third ("Mol-Metal R18").

### 1.4 LOI-ORNL — protein-metal binding SPR + ITC (English)

> **Subject:** Protein-metal binding validation collaboration request (Mol-Metal R18 + ORNL SPR/ITC)

Rationale: noun-first ("Protein-metal binding validation"), method-second
("SPR/ITC"), project-third ("Mol-Metal R18 + ORNL").

## 2. Body template (3 paragraphs: purpose / what we need / what we offer)

Word-count target: **200–300 words** for the customized body. The template
below is the cover-note body; the full LOI document is the attachment (not
in the body).

### 2.1 LOI-RUSANOV — English

> Dear Prof. {Rusanov_corresponding_author_name},
>
> I am writing on behalf of the Mol-Metal open-source project
> (mol-metal.org, GitHub `mol-metal`) to request a **2-week data
> collaboration** with the PlatinAI / MB-Finder team. We have built a
> multi-task D-MPNN pIC50 predictor on the PlatinAI compound catalogue
> and a flow-matching backbone for metallodrug design; both are open
> source. We are now looking for **per-cell-line NUMERIC IC50 labels** for
> ~200 Pt compounds across MCF7 / A549 / A2780 / A2780cis so we can
> retrain our predictor on real (not synthetic) labels and close the
> Wf-297B §6 honesty gap that currently gates our paper §5
> `Pearson r ≥ 0.7` claim.
>
> **What we need from your team:** approval to release Table S4-S5
> supplementary NUMERIC IC50 (xlsx or CSV) for ~200 Pt compounds,
> scoped to your existing 4-cell PlatinAI benchmark. The MB-Finder
> portal already exposes compound IDs + coverage matrix publicly;
> this request only adds the per-cell NUMERIC column.
>
> **What we offer in return:** (a) prominent Acknowledgement naming the
> Rusanov 2025 PlatinAI / MB-Finder team on the Mol-Metal paper §5;
> (b) mid-author position on the data-validation section; (c)
> co-corresponding authorship if NUMERIC release drives a published
> D-MPNN retrain that beats our synthetic-label baseline by ≥ 0.1 log
> units on held-out MCF7 + A2780cis; (d) CC-BY-4.0 release of the joint
> `platinai_4cell_real_ic50.csv` on the Mol-Metal data portal with
> explicit citation to Rusanov 2025 + MB-Finder. There is no funding
> exchange, no wet-lab cost, and no exclusivity clause — your team
> retains full rights to the NUMERIC data.
>
> Would you be open to a 30-minute Zoom in the next 2–3 weeks to
> confirm fit? I am attaching the 1-page project summary, our top-20
> R20-P6 candidate list, and the D-MPNN per-compound pIC50 snapshot.
>
> Best regards,
> [signature block — see §4]

**Customized word count target: ~270 words** (3-paragraph body).

### 2.2 LOI-PKU — 中文

> {PI_name} 教授，您好：
>
> 我代表 Mol-Metal 开源项目 (mol-metal.org, GitHub `mol-metal`) 与您
> 联系，希望探讨一个 **6–12 个月的湿实验验证合作**。我们刚刚发布了
> R18 版本管线，整合了点击化学金属基药物生成器、λ-组合子搜索与
> flow-matching 主干网络；针对若干金属酶口袋，我们已经对 Pt / Pd /
> Au / Ru / Ir 跨 5 种金属共 **20 个候选分子** 进行了多目标加权
> 排序 (dmpnn_pic50 0.40 + qed 0.20 + sa 0.15 + vina 0.15 + astr 0.10)。
>
> **我们希望贵组提供：** 顺铂类 (cisplatin-class) 4 个细胞系
> (MCF7 / A549 / A2780 / A2780cis) 的 IC50 验证，覆盖 top-20 候选
> 分子 (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11)，每 (compound × cell) 8
> 个 dose × 3 个 biological replicate = 1,920 个数据点。可选扩展
> 5–9 个 NCI-60 cell (建议与协和分工，贵组负责 4-cell 基准 + 部分
> NCI-60)。顺铂作为阳性对照。
>
> **我们提供：** (a) 全部 in-silico 预测 (3D 结构、predicted IC50、
> logP、redox、HSA K_d、cellular uptake pg/cell)；(b) IC50 分析 pipeline
> 模板 (template_ic50_pipeline.py) 即时接收湿实验室数据；
> (c) 联合 manuscript + model retrain；(d) Phase-2 验证发现 ≥1 sub-µM
> IC50 候选时，贵组 PI 获 **共同通讯作者 (co-corresponding)** 署名。
> 本合作不涉及经费划拨、间接成本或管理费；可配合在贵校财务体系下
> 共同申报 NSFC / 科技部基金。
>
> 请问您未来 2–3 周内是否有 30 分钟的 Zoom 会议时间讨论契合度？我
> 已附上 1 页项目摘要、top-20 候选分子清单 (Pt 2 / Pd 3 / Au 1 /
> Ru 3 / Ir 11) 以及 D-MPNN per-compound pIC50 快照。
>
> 此致
>
> 敬礼
>
> [signature block — 见 §4]

**Customized word count target: ~290 字** (中文 paragraph-count differs).

### 2.3 LOI-协和 — 中文

> {PI_name} 教授，您好：
>
> 我代表 Mol-Metal 开源项目 (mol-metal.org, GitHub `mol-metal`) 与您
> 联系，希望探讨一个 **6–12 个月的医院期湿实验合作**。我们刚刚
> 发布了 R18 版本管线；针对 R20-P6 top-20 金属基药物候选分子
> (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11)，我们希望由贵组开展 **NCI-60
> 9-cell 完整 panel** 的 SRB / MTT 细胞毒评价 + ICP-MS 细胞内金属
> 摄取定量，为 lead compound 临床转化奠基。
>
> **我们希望贵组提供：** (i) NCI-60 9-cell 完整 panel (Leukemia
> CCRF-CEM / Lung A549 / Colon HCT-116 / CNS SF-539 / Melanoma
> LOX-IMVI / Ovarian OVCAR-3 / Renal A498 / Prostate PC-3 / Breast
> MCF7) 的 SRB/MTT 全 9-cell × 20 化合物 × 8 dose × 3 replicate =
> 4,320 个 IC50 数据点；(ii) ICP-MS (Thermo iCAP RQ / Agilent 7900)
> 摄取 (Pt / Pd / Au / Ru / Ir) 在 ≥2 cell × 3 time-point = 180 个
> ng/cell 数据点；(iii) IRB + mycoplasma + 临床前 GCP 流程；
> (iv) Phase-3 lead compound 临床转化准备 (PK/PD 桥接实验；可选
> NSG 小鼠 xenograft)。
>
> **我们提供：** (a) 全部 in-silico 预测；(b) IC50 + 摄取分析 pipeline
> 模板；(c) 联合 manuscript + clinical case report；(d) 协和医院名称
> 列入 cover sheet；Phase-2 完成 NCI-60 9-cell 完整 panel 时，贵组 PI
> 获 **共同通讯作者 (co-corresponding)** 署名。**建议分工**：贵组负责
> NCI-60 9-cell + ICP-MS 摄取；PKU LOI 负责 4-cell 基准 + 部分 NCI-60
> (避免工作量冲突)。本合作不涉及经费划拨。
>
> 请问您未来 2–3 周内是否有 30 分钟的 Zoom 会议时间讨论契合度？我
> 已附上 1 页项目摘要、top-20 候选分子清单以及 D-MPNN per-compound
> pIC50 快照。
>
> 此致
>
> 敬礼
>
> [signature block — 见 §4]

**Customized word count target: ~310 字** (slightly over the 200–300 range;
trim the optional NSG xenograft mention if it lands over 300 字).

### 2.4 LOI-ORNL — English

> Dear Prof. {ORNL_PI_name},
>
> I am writing on behalf of the Mol-Metal open-source project
> (mol-metal.org, GitHub `mol-metal`) to explore a **9–12 month
> protein-metal binding validation collaboration** with your group at
> ORNL. We have just released **R18** of our pipeline, which couples a
> click-chemistry metallodrug generator with a lambda-combinator search
> and a flow-matching backbone; we have ranked the top 30–50 Pt / Ru / Ir
> candidates for several metalloenzyme pockets, including the 4-protein
> panel we propose to validate with you (GST, cisplatin-DNA lesion,
> p53 DBD, metallothionein MT-2A).
>
> **What we need from your group:** SPR (Biacore X100 / T200) + ITC
> (MicroCal PEAQ) K_d validation on the top-20 R20-P6 candidates × 4
> proteins = 80 K_d measurements (each n=3 technical replicates),
> delivered as per-compound × per-protein JSON conforming to
> `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json`. AMBER
> force-field + ORNL cluster access for MD refinement is the natural
> supporting capability.
>
> **What we offer in return:** (a) joint manuscript with the data-validation
> section; (b) co-corresponding authorship if Phase-2 validation surfaces
> ≥ 1 sub-µM K_d candidate for a clinically-relevant target (e.g. p53 DBD
> rescue, metallothionein displacement); (c) mid-author position with the
> data-validation section otherwise; (d) joint grant proposal under ORNL
> as fiscal host (NIH R01 / DOE BER / NSF) — we can supply a letter of
> support and a 1-page research plan on request. No funding transfer,
> no overhead, no exclusivity.
>
> Would you be open to a 30-minute Zoom in the next 2–3 weeks to confirm
> fit and protein availability? I am attaching the 1-page project summary,
> the top-20 candidate list (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11), and the
> D-MPNN per-compound pIC50 snapshot.
>
> Best regards,
> [signature block — see §4]

**Customized word count target: ~290 words** (3-paragraph body).

## 3. Attachments list (per-LOI)

All 4 LOIs share a common 4-attachment baseline; some LOIs add a 5th
LOI-specific attachment.

### 3.1 Common attachments (all 4 LOIs)

| # | File | Description | Source |
|---|---|---|---|
| 1 | **1-page project summary PDF** | Project 1-pager (NEW — TODO — Hugo to draft or delegate to a sub-agent). Should fit on 1 page: title, R18 release bullets, top-20 candidate list, ask, offer, timeline. | TODO — Hugo to create `molmetal/docs/wetlab_protocols/project_summary_1page.pdf` |
| 2 | **Full LOI document (PDF)** | The 4 `wetlab_loi_*.md` file converted to PDF. Use `pandoc -t latex` or `md-to-pdf`. | `TODO/pending/wetlab_loi_*.md` (4 files) |
| 3 | **Top-20 candidates CSV** | R20-P6 top-20 ranked candidates (Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11) with multi-objective score breakdown. | `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` |
| 4 | **D-MPNN per-compound pIC50 snapshot** | Per-compound × per-cell-line pIC50 spread 1.7 log units, JSON. | `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` |

### 3.2 LOI-specific 5th attachment (1 of 4 only)

| LOI | 5th attachment | Source |
|---|---|---|
| LOI-RUSANOV | Wf-295B MB-Finder coverage matrix harvest report | `molmetal/reports/wf295b_platinai_loader/loader_report.md` (PDF) |
| LOI-PKU | NCI-60 9-cell sourcing matrix (which cell lines available where) | TODO — Hugo to draft or pull from `wf58_dryrun/` |
| LOI-协和 | NCI-60 9-cell sourcing matrix + ICP-MS instrument spec sheet | TODO — Hugo to draft |
| LOI-ORNL | 4-protein (GST / cisplatin-DNA lesion / p53 DBD / MT-2A) sourcing matrix | TODO — Hugo to draft |

### 3.3 File format discipline

- All attachments as **PDF** (not .docx / .md) — preserves formatting across
  recipient mail clients.
- Filename convention: `Mol-Metal_LOI-{INSTITUTION}_{VERSION}_{YYYY-MM-DD}.pdf`
  for the LOI document; `Mol-Metal_project_summary_{YYYY-MM-DD}.pdf` for
  the 1-pager.
- File size budget: each PDF ≤ 5 MB; total email ≤ 20 MB.

## 4. Signature block

> Best regards,
>
> ---
> **Hugo Lee**
> Mol-Metal Project Lead
> Open-source metallodrug design pipeline (R18 release, 2026-09)
> mol-metal.org · GitHub: `mol-metal`
> Email: hugo.lee@mol-metal.org
> ORCID: {Hugo_ORCID — TBD, fill once and reuse}
> ---
>
> *This message and any attachments are intended solely for the named
> recipient. If you have received it in error, please notify the sender
> and delete the message. The Mol-Metal project is open-source under
> CC-BY-4.0; the LOI document is shared under the data-sharing MoU
> template (`molmetal/docs/wetlab_protocols/data_sharing_agreement.md`).*

### 4.1 中文 signature (for LOI-PKU + LOI-协和)

> 此致
>
> 敬礼
>
> ---
> **Hugo Lee**
> Mol-Metal 项目负责人
> 开源金属基药物设计流水线 (R18 版本，2026-09)
> mol-metal.org · GitHub: `mol-metal`
> Email: hugo.lee@mol-metal.org
> ORCID: {Hugo_ORCID — TBD, fill once and reuse}
> ---
>
> *本邮件及附件仅供指定收件人阅悉。如误收，请通知发件人并删除。Mol-Metal
> 项目采用 CC-BY-4.0 开源；LOI 文件按数据共享 MoU 模板
> (`molmetal/docs/wetlab_protocols/data_sharing_agreement.md`) 共享。*

## 5. CC + cover sheet protocol

| LOI | Primary recipient | CC | Reply-to |
|---|---|---|---|
| LOI-RUSANOV | Rusanov 2025 corresponding author | none (single-author email is professional norm for data request) | hugo.lee@mol-metal.org |
| LOI-PKU | PKU 医学部 药物化学 / 化学生物学 PI | Hugo's mutual contact at PKU (TBD — Hugo to identify) | hugo.lee@mol-metal.org |
| LOI-协和 | 协和医学院 医院期细胞毒理学 PI | Hugo's mutual contact at 协和 (TBD) + 协和 科研处 联系人 (TBD) | hugo.lee@mol-metal.org |
| LOI-ORNL | ORNL protein-metal binding PI | Hugo's mutual contact at ORNL (TBD — Hugo to identify via Cn/journals or ORNL directory) | hugo.lee@mol-metal.org |

## 6. Follow-up cadence

Per `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §5
(WLP-OUT-001 standard cadence):

| Touch-point          | Timing            | Channel   | Owner             |
|----------------------|-------------------|-----------|-------------------|
| First email          | Day 0             | Email     | Hugo              |
| Light follow-up      | Day +7            | Email     | Hugo              |
| Second follow-up     | Day +21           | Email     | Hugo              |
| Intro Zoom (if yes)  | Day +28           | Video     | Both PIs          |
| MoU signed           | Day +60           | DocuSign  | Both institutions |
| Phase-1 ack          | Month +2          | Email     | Hugo              |
| Phase-2 review       | Month +6          | Video     | Both PIs          |
| Manuscript kickoff   | Month +9          | Video     | Both PIs          |

If the Day +21 second follow-up receives no response, mark the contact as
**DECLINED** in `TODO/pending/wetlab_outreach_log.csv` (TODO — Hugo to
create) and move on. Do **not** send a third follow-up — that crosses into
pushy-territory per `TODO/INDEX.md` §12.3 P3 mitigation.

## 7. Customization checklist (Hugo runs per LOI)

- [ ] **PI email + name**: look up from source noted in `wetlab_loi_send_ready_package.md` §3 row.
- [ ] **Subject line**: copy from §1 above (LOI-specific).
- [ ] **Body customization**: copy the LOI-specific body from §2 above;
      fill the 4 placeholder fields: `{PI_name}`, `{institution}`,
      `{specific_ask_2_sentences}`, `{specific_offer_2_sentences}`.
- [ ] **Word count check**: body should land in **200–300 words / 字**.
      Trim any one paragraph if it exceeds.
- [ ] **Attachments**: §3 — common 4 attachments + 1 LOI-specific 5th.
- [ ] **Signature**: copy from §4. **ORCID is the only Hugo-personal
      field** — fill once and reuse for all 4 sends.
- [ ] **CC + cover sheet**: §5 — LOI-specific CC list.
- [ ] **Send Day 0**: log the send in `TODO/pending/wetlab_outreach_log.csv`
      (TODO — Hugo to create). Update LOI header line 5 from
      `Status: DRAFT — not sent` → `Status: SENT YYYY-MM-DD`.
- [ ] **Day +7 light follow-up**: short ping ("checking if my note landed,
      happy to Zoom if useful").
- [ ] **Day +21 second follow-up**: substantive follow-up with 1 new
      paragraph (e.g. "since my first note, we've also shipped
      `wf58_dmpnn_predictions/per_compound_pic50.json` — attaching snapshot").
- [ ] **Day +28 intro Zoom** (if requested): 30 min, agenda per
      `collaborator_outreach_template.md` §2 framework.
- [ ] **Day +60**: if no reply, mark `**Status:** DECLINED YYYY-MM-DD`
      and move to next PI on the fallback list.

## 8. Honest framing (party-twenty-spirit §3.5)

- This is a **cold-email template**, NOT a send tool. **0 email sent at
  this revision** (2026-09-29).
- All 4 LOIs are now **send-ready in draft form** — subject + body +
  attachments + signature template standardised.
- All 4 LOIs are still **BLOCKED on actual send** by exactly two missing
  fields: (a) PI email (4 × TBD, Hugo's responsibility) and
  (b) Hugo's ORCID (1 × TBD, fill once and reuse).
- The 4 customized bodies above are **target word counts**, not hard caps.
  Trim if a paragraph exceeds; expand if the LOI is unusually short.
- **The 1-page project summary PDF is NOT yet created** — this is a
  remaining TODO that gates actual send for all 4 LOIs. Recommend Hugo
  draft it (or delegate to a sub-agent) before any send.
- The `wetlab_outreach_log.csv` is **NOT yet created** — Hugo should
  create it on first send (or before) with columns:
  `loi_id, target_pi, send_date, status, reply_date, reply_type, notes`.

## 9. Cross-references

- `TODO/pending/wetlab_loi_send_ready_package.md` — companion Wf-310G
  PART 1 file: 4-row summary table + per-LOI next-action checklist +
  cold-email sending protocol.
- `TODO/pending/wetlab_loi_consolidation_review.md` — Wf-308D PART 2
  pre-existing 4-row summary table.
- `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` (Wf-305A LOI)
- `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` (Wf-74 LOI)
- `TODO/pending/wetlab_loi_xiehe_nci60.md` (Wf-74 LOI)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` (Wf-74 LOI)
- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` (Wf-28
  generic email template, WLP-OUT-001) — §3 English + §4 中文
- `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` (Wf-28
  data-sharing MoU template)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule;
  outreach NOT gated but send is user's call)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run)
- `molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv` (top-20
  candidates referenced in all 4 LOIs)
- `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json` (D-MPNN
  pIC50 spread 1.7 log units referenced in all 4 LOIs)
- `molmetal/reports/wf295b_platinai_loader/loader_report.md` (Wf-295B
  coverage-only MB-Finder harvest — RUSANOV-specific 5th attachment)

## 10. Revision history

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-29 | Wf-310G | Initial 4-LOI cold-email template (subject variants + 3-paragraph body EN/中文 + attachments + signature). **0 email sent.** |
