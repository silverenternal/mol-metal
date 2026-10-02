# LOI-PKU — Letter of Intent: Cisplatin-Class IC50 Validation

**Document ID:** LOI-PKU-2026-09
**Version:** 1.0 (draft, ready for review)
**Status:** **TEMPLATE_NOT_NEEDED_FOR_TIER2** — see AGENTS.md §5.4 + §10 + 2026-09-30 user reinforcement "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了". Tier-2 journal acceptance does not require wet-lab data. Retained as **historical artifact only**; any agent action attempting to send this LOI must REFUSE per AGENTS.md §10.3.
**(Historical note: DRAFT — not sent at this revision)** (Hugo Lee coordinates the cold-email per `TODO/INDEX.md` §12.3 P3 mitigation; no action will be taken — wet-lab DEPRECATED per AGENTS.md §5.4 + §10)
**Target PI:** 北京大学医学部 (Peking University Health Science Center) 药物化学 / 化学生物学 课题组
**Authoring date:** 2026-09-27
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`

---

## 1. 目的 (Purpose)

本意向书 (Letter of Intent, LOI) 提议 Mol-Metal 开源项目与北京大学医学部
一个具有金属基药物合成与细胞活性评价能力的课题组开展短期合作验证。
合作将对 Mol-Metal R18 版本筛选出的前 20 名金属基药物候选分子
(Pt / Pd / Au / Ru / Ir 跨 5 种金属) 进行顺铂类 (cisplatin-class)
体外抗癌活性 (IC50) 评价，覆盖 MCF7 / A549 / A2780 / A2780cis 四系
基准 + NCI-60 9-cell 扩展面板。

## 2. 背景 — Mol-Metal 提供的资源

Mol-Metal 是一个开源金属基药物设计流水线 (mol-metal.org)，
R18 版本 (2026-09) 已经发布：

- **点击化学金属基药物生成器** (`molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
- **λ-组合子搜索 + 金属感知奖励塑形** (`molmetal/molmetal_lam/search_alg/proof_search.py`).
- **耦合 flow-matching 主干网络** (`molmetal/adapters/flow_matching_lipman/`).
- **R20-P6 前 20 名候选分子** (`molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv`):
  Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11，多目标加权排序
  (dmpnn_pic50 0.40 + qed 0.20 + sa 0.15 + vina 0.15 + astr 0.10)。
- **D-MPNN 跨 4 个细胞系 pIC50 预测**，spread 1.7 log units
  (Wf-58 P6.1 `molmetal/reports/wf58_dmpnn_predictions/per_compound_pic50.json`).
- **IC50 分析 pipeline 模板** (`TODO/pending/wetlab_ic50_analysis_pipeline.md`
  + `molmetal/scripts/template_ic50_pipeline.py`) 准备好接收湿实验室数据。

开放数据集：PlatinAI (CC0)、tmQM (CC-BY)、MetalCytoToxDB Ru 子集 (CC-BY-SA)。

## 3. PKU 提供的资源

北大医学部在金属基药物合成与化学生物学方面具有深厚积累：

- 顺铂 (cisplatin)、卡铂 (carboplatin)、奥沙利铂 (oxaliplatin) 类似物的成熟合成路线。
- 多个金属基药物专利与发表经验。
- BSL-2 细胞培养设施 + 受训操作员。
- NMR + X-ray 单晶衍射表征能力。
- NCI-60 9-cell 面板来源 (Leukemia CCRF-CEM / Lung A549 / Colon HCT-116 / CNS SF-539 / Melanoma LOX-IMVI / Ovarian OVCAR-3 / Renal A498 / Prostate PC-3 / Breast MCF7)。

## 4. 提议合作框架 (3 阶段，6-12 个月)

| 阶段 | 时长 | Mol-Metal 提供 | PKU 提供 |
|---|---|---|---|
| **Phase 1** | 0–2 月 | 5–10 个 Pt/Ru/Ir 候选分子 (3D 结构、in-silico IC50/logP/redox/HSA K_d/摄取); 分析 pipeline 模板 | 化合物 aliquots 发货 (或合成 recipe); 反向签 MoU; 4-cell 验证 (MCF7/A549/A2780/A2780cis) 启动 |
| **Phase 2** | 2–6 月 | 化合物 aliquots (或合成路线); 每轮更新排序 | 顺铂 IC50 曲线: MCF7/A549/A2780/A2780cis + 5 个 NCI-60 cell = 9 cell × 20 化合物 × 8 dose × 3 replicate = 4,320 个数据点; 顺铂作为阳性对照 |
| **Phase 3** | 6–12 月 | 联合 manuscript; 模型用实测数据 retrain | Lead compound 活性验证 (≥1 sub-µM IC50); 可选 X-ray/NMR speciation |

60-day cadence: Day 0/7/21 (邮件跟进) → Day 28 (intro Zoom) → Day 60 (MoU signed) → Month 2/6/9 (阶段评审)。

## 5. 署名 (Authorship)

- Phase-2 验证发现 **≥1 sub-µM IC50 候选** → PKU PI 获 **共同通讯作者 (co-corresponding)** 署名。
- Phase-2 验证无 sub-µM 命中 → PKU PI 获 **中位作者** + 数据验证章节署名。
- 双方在投稿前书面约定作者顺序。
- 预印本 / 会议摘要：dry-lab PI 与 PKU PI 联合展示；摘要第一作者按实验贡献决定。

## 6. 经费与设备

- **Mol-Metal 一方**：软件、算力、in-silico 预测、manuscript 准备。**无湿实验室预算**。
- **PKU 一方**：合成试剂、NMR / X-ray 时间、细胞培养耗材、plate-reader 时间、操作员人力。
- **联合基金申请鼓励** (NSFC / 科技部)。以 PKU 为财务主体；Mol-Metal 可提供支持信 + 1 页研究计划。
- **无管理费 / 间接成本划拨**。

## 7. 目标细胞系 (4-cell 基准 + NCI-60 扩展)

### 7.1 4-cell 基准 (Phase 2 必须)

| Cell line | Tissue | 备注 |
|---|---|---|
| **MCF7** | Breast (ER+) | 顺铂敏感 |
| **A549** | Lung (NSCLC) | 顺铂中等敏感 |
| **A2780** | Ovarian (顺铂敏感) | 顺铂敏感参考 |
| **A2780cis** | Ovarian (顺铂耐药) | 顺铂耐药模型 |

### 7.2 NCI-60 扩展 (Phase 2 可选，与协和分工)

| Cell line | Tissue |
|---|---|
| **CCRF-CEM** | Leukemia |
| **HCT-116** | Colon |
| **SF-539** | CNS |
| **LOX-IMVI** | Melanoma |
| **OVCAR-3** | Ovarian |
| **A498** | Renal |
| **PC-3** | Prostate |

## 8. 期望输出 (Expected outputs)

- 每个 (compound × cell) IC50 ± std (n=3 biological replicates, 8-dose 半对数网格)。
- 每个 (compound × cell) pIC50 + 95% CI (1,000-iteration residual bootstrap)。
- Hill slope + 4-parameter logistic 拟合参数。
- 顺铂耐药比 (resistance fold) = 10 ** (pic50_A2780cis - pic50_A2780)。
- 原始 plate-reader 文件上传至 `data/wetlab/cytotox/<run_id>/` 含 checksum。
- JSON 输出符合 `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` schema。

## 9. 时间表 + 交付物

| 月份 | Milestone | 交付物 |
|---|---|---|
| 0 | MoU signed; 化合物 aliquots 发货 | MoU + 化合物清单 |
| 1 | 4-cell 验证启动; 顺铂阳性对照 IC50 校准 | QC report |
| 2 | 第一个 5 个候选 IC50 测定完成 | 5 × 4 = 20 IC50 |
| 3 | 第二个 5 个候选 IC50 测定完成 | + 20 IC50 |
| 4 | 第三个 5 个候选 IC50 测定完成 | + 20 IC50 |
| 5 | 第四个 5 个候选 IC50 测定完成 | + 20 IC50 |
| 6 | Phase-2 评审; lead compound 选定 | IC50 summary report |
| 6–9 | Phase-3 lead compound 活性验证; X-ray/NMR 可选 | Speciation data |
| 9–12 | 联合 manuscript; 用实测数据 retrain 模型 | Joint paper |

## 10. 风险与缓解 (Risk and mitigation)

- **化合物溶解度**：顺铂类化合物在 DMSO 中溶解良好；Pd / Au / Ir 类似物可能需要超声助溶或共溶剂 (Tween-80 ≤ 0.1% v/v)。
- **顺铂耐药基线**：A2780cis 的顺铂 IC50 应在 10–30 µM 范围；若偏离，flag 该细胞系可能污染或失耐药性。
- **NCI-60 9-cell panel 工作量**：若 PKU 课题组工作量大，可与 协和 LOI 分工 (见 `wetlab_loi_xiehe_nci60.md`)。

## 11. 保密与数据共享 (Confidentiality and data sharing)

- 所有湿实验室数据按 `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` MoU 模板共享。
- 联合预印本 embargo 至双方 PI 同意投稿。
- Mol-Metal 所有 R18 / R20 代码均开源 (CC-BY-4.0); 湿实验室 IC50 数据在论文发表后成为 Mol-Metal 开放数据集的一部分。

## 12. 下一步 (Next steps)

1. Hugo Lee 用 `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §4 中文模板发送本 LOI 给 PKU PI。
2. Day +7 轻量跟进；Day +21 第二次跟进。
3. Day +28 intro Zoom (30 分钟) — 确认契合度、合成能力、细胞系可用性。
4. Day +60 MoU 双方签字；Phase 1 启动。

## 13. 诚实标注 (Honest framing per party-twenty-spirit §3.5)

- 本 LOI 是 **草案** (draft)，等待 Hugo Lee 审阅；**本版本 0 LOI 已发送**。
- **当前 0 个湿实验室 IC50 实测**；in-silico D-MPNN pIC50 来自合成 IC50 labels (Rusanov 2025 tables S4-S5 受 Cloudflare 限制)，作为 **rank-only**。
- 头条 `Pearson r ≥ 0.7` (D-MPNN 预测 vs PKU 实测) **当前无法 MEASURE**；需 ≥1 PI 回复 + MoU + Phase 1 启动 + 6 个月时间线完成 (最早 2027 Q1)。
- Paper §5 湿实验室验证章节 🚫 **DEFERRED** per TODO-84 paper rule; 需要 explicit user authorization。

## 14. 交叉引用 (Cross-references)

- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §4 中文模板
- `molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` (Wf-28 通用 SRB 协议, WLP-CYTO-001)
- `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` (Wf-28 数据共享 MoU 模板)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (配套 IC50 分析 pipeline)
- `TODO/pending/wetlab_sop_cytotoxicity_assay.md` (NEW 机构定制 SOP)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 夸大 close-out — 由 97c 修正)
- `TODO/completed/97c_wf74_reclassification.md` (本 Wf-74 v2 close-out)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 计划)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run layer ship)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper 规则; outreach 不受 gating)

## 15. 修订历史 (Revision history)

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-27 | Wf-74  | 机构定制 LOI 草案 (等待 Hugo Lee 审阅; **0 已发送**)。 |
