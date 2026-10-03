# Wet-Lab Collaborator Outreach — Email Templates and Collaboration Framework

**Document ID:** WLP-OUT-001
**Version:** 1.0
**Status:** Template (drafts to be sent by the Mol-Metal coordinator; no
contact has been initiated at this revision)
**Owner:** Mol-Metal wet-lab coordinator (TBD)
**Last reviewed:** 2026-09-26

---

## 1. Purpose

This document provides ready-to-send outreach email templates (中文 + English)
plus a written collaboration framework covering deliverables, data sharing,
authorship, and resourcing. It is intended for **internal use only** — the
Mol-Metal coordinator customizes the text for the specific PI and sends it
from a personal / institutional account. Do **not** send the template
verbatim without review.

The target audience is **principal investigators** (PIs) running a BSL-2
medicinal-chemistry / cancer-biology lab with access to cell culture,
SRB / MTT cytotoxicity, and (ideally) ICP-MS. Optional capabilities that
strengthen the fit: ITC, X-ray crystallography, NMR.

## 2. Collaboration Framework

The Mol-Metal / wet-lab collaboration is structured as a **three-phase
joint program**:

| Phase | Duration    | Mol-Metal deliverable                    | Wet-lab deliverable                  |
|-------|-------------|------------------------------------------|--------------------------------------|
| 1     | 0–2 months  | 5–10 ranked Pt / Ru / Ir candidates + structures, in-silico predicted IC50, logP, redox potential, predicted HSA K_d, predicted uptake (pg/cell) | Acknowledge fit; counter-sign MoU; share cell-line authentication and mycoplasma-test records |
| 2     | 2–6 months  | Compound aliquots shipped (or synthesis recipe if MoU allows in-house resynthesis); updated ranking after each round | Run SRB cytotoxicity (≥ 4 cell lines), ICP-MS uptake (≥ 2 cell lines, ≥ 3 time-points); optional ITC on serum protein |
| 3     | 6–12 months | Joint manuscript draft; updated Mol-Metal model retrained on measured data; co-design of next round | Repeat protocols on top candidates; optional X-ray / NMR speciation |

After each phase the partners reconvene for a 60-min review and decide
whether to proceed.

### 2.1 What Mol-Metal brings to the partnership

- **R18 ship modules** (the production Mol-Metal release as of 2026-09):
  - Click-chemistry metallodrug generator
    (`molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
  - Lambda-combinator search with metal-aware reward shaping
    (`molmetal/molmetal_lam/search_alg/proof_search.py`,
    `molmetal/molmetal_lam/reward/learned_shaping.py`).
  - Coupled flow-matching backbone
    (`molmetal/adapters/flow_matching_lipman/`).
- **Pre-computed candidate libraries**:
  - 100-pocket Pt / Ru / Ir ligand-set scored against the PlatinAI
    training corpus.
  - 30–50 top hits per pocket, ranked by composite score.
- **In-silico prediction panel**:
  - Predicted IC50 (with 80 % CI), predicted logP, redox potential, HSA K_d.
  - Counter-screen predictions for off-target metalloproteins.
- **Open datasets**: PlatinAI (CC0), tmQM (CC-BY), MetalCytoToxDB Ru subset
  (CC-BY-SA).
- **Open-source pipeline** so the wet-lab team can audit every prediction.

### 2.2 What the wet-lab partner contributes

- BSL-2 cell-culture facility and trained operator.
- SRB (or MTT) cytotoxicity assay — at least 4 cell lines including one
  cisplatin-resistant line.
- ICP-MS quantification of Pt / Ru / Ir.
- (Optional, value-adding) ITC, X-ray crystallography, NMR, or
  speciation LC-ICP-MS.
- Author-time on the joint manuscript, including figure generation and
  statistical analysis.

### 2.3 Authorship

- Phase-2 validation that surfaces ≥ 1 sub-µM IC50 candidate qualifies the
  wet-lab PI for **co-corresponding authorship**.
- Phase-2 validation alone (without a hit) qualifies for a mid-author
  position with the data-validation section of the paper.
- Both teams agree on author order in writing before submission.
- Pre-print / conference abstracts: dry-lab PI and wet-lab PI present
  jointly; first authorship on the abstract follows experimental
  contribution to the underlying figure.

### 2.4 Funding and equipment

- Mol-Metal side: software, compute, in-silico predictions, manuscript
  preparation. No wet-lab budget.
- Wet-lab side: reagents, cell-line maintenance, plate-reader and
  ICP-MS time, operator hours.
- Joint grant applications encouraged; submit under the wet-lab
  institution as the fiscal host. The Mol-Metal team can supply a
  letter of support and a 1-page research plan on request.
- No overhead or indirect-cost transfers are implied.

## 3. Email Template (English, formal)

> **Subject:** Joint metallodrug validation collaboration — Mol-Metal open
> pipeline + wet-lab cytotoxicity / ICP-MS
>
> Dear Prof. {{LastName}},
>
> I am writing on behalf of the Mol-Metal open-source project
> (mol-metal.org) to explore a short-term validation collaboration with
> your group. We have just released **R18** of our pipeline, which couples
> a click-chemistry metallodrug generator with a lambda-combinator search
> and a flow-matching backbone; we have ranked the top 30–50 Pt / Ru / Ir
> candidates against {{target_protein_or_cell_line}} for a number of
> metalloenzyme pockets.
>
> We are looking for one (and at most two) wet-lab groups to validate the
> top candidates via (i) SRB / MTT cytotoxicity on a panel of 4 cell
> lines, and (ii) ICP-MS cellular-uptake quantification. Your recent
> paper on {{topic}} and your group’s experience with {{assay}} make
> this an unusually good fit.
>
> The collaboration is structured as three short phases over roughly 12
> months; details are in the attached one-page framework document. In
> summary: we ship the compounds (or synthesis recipes) and the
> in-silico predictions; your team runs the assays; we co-write the
> resulting paper. Compounds with sub-µM IC50 + measurable uptake qualify
> for joint corresponding authorship.
>
> There is no funding transfer and no overhead. We are happy to support
> a joint grant proposal under your institution’s fiscal administration.
>
> Would you be available for a 30-minute Zoom in the next 2–3 weeks to
> discuss fit? I will follow up by email next week if I do not hear back.
>
> Best regards,
>
> {{Coordinator Name}}
> Mol-Metal coordinator
> {{institutional_email}}

## 4. Email Template (中文)

> **主题：** 金属基药物 (metallodrug) 湿实验验证合作邀请 — Mol-Metal
> 开源管线 + SRB 细胞毒 / ICP-MS 摄取
>
> {{LastName}} 教授，您好：
>
> 我代表 Mol-Metal 开源项目 (mol-metal.org) 与您联系，希望探讨一个
> 短期验证合作的可行性。我们刚刚发布了 R18 版本管线，整合了
> 点击化学金属基药物生成器、λ-组合子搜索与 flow-matching 主干网络；
> 针对若干金属酶口袋，我们已经对 Pt / Ru / Ir 候选分子进行排序，
> 并给出每个口袋的前 30–50 名。
>
> 我们希望寻找 1 个（最多 2 个）湿实验室小组，对这些候选分子进行
> (i) 4 个细胞系的 SRB / MTT 细胞毒测试，以及 (ii) 基于 ICP-MS 的
> 细胞内金属摄取定量。您课题组在 {{topic}} 上的近期工作以及在
> {{assay}} 上的经验，与本项目契合度极高。
>
> 合作分为三个短阶段，预计 12 个月。详情见附件一页的合作框架。简而言之：
> 我们提供化合物（或合成路线）和全部 in-silico 预测；贵组执行实验；
> 双方共同撰写论文。IC50 < 1 µM 且具有可测量摄取的候选分子，其
> 湿实验负责人可获共同通讯作者署名。
>
> 本合作不涉及经费划拨、间接成本或管理费。如有需要，我们可配合
> 在贵校财务体系下共同申报基金，并提供 1 页研究计划与支持函。
>
> 请问您未来 2–3 周内是否有 30 分钟的 Zoom 会议时间可以讨论合作
> 契合度？若一周内未收到回复，我会再次邮件跟进。
>
> 此致
>
> 敬礼
>
> {{Coordinator Name}}
> Mol-Metal 项目协调人
> {{institutional_email}}

## 5. Follow-up Cadence

| Touch-point          | Timing            | Channel   | Owner             |
|----------------------|-------------------|-----------|-------------------|
| First email          | Day 0             | Email     | Coordinator       |
| Light follow-up      | Day +7            | Email     | Coordinator       |
| Second follow-up     | Day +21           | Email     | Coordinator       |
| Intro Zoom (if yes)  | Day +28           | Video     | Both PIs          |
| MoU signed           | Day +60           | DocuSign  | Both institutions |
| Phase-1 ack          | Month +2          | Email     | Coordinator       |
| Phase-2 review       | Month +6          | Video     | Both PIs          |
| Manuscript kickoff   | Month +9          | Video     | Both PIs          |

If the second follow-up receives no response, mark the contact as
**declined** in `logs/wetlab_outreach.csv` and move on.

## 6. References

- Mol-Metal repository: https://www.mol-metal.org
- PlatinAI dataset (CC0): https://github.com/mol-metal/platinai
- tmQM dataset (CC-BY): https://github.com/aimat-lab/tmQM
- NCI-60 SRB protocol — Skehan P. et al., *J. Natl. Cancer Inst.* 82, 1107
  (1990).
- ICP-MS EPA Method 6020B (current revision).

## 7. Revision History

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 0.1 | 2026-09-26 | Wf-28  | Initial template (no outreach sent yet).            |
