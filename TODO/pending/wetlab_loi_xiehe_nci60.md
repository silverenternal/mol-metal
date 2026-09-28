# LOI-协和 — Letter of Intent: NCI-60 9-Cell Panel Hospital-Phase Cytotoxicity

**Document ID:** LOI-协和-2026-09
**Version:** 1.0 (draft, ready for review)
**Status:** **DRAFT — not sent at this revision** (Hugo Lee coordinates the cold-email per `TODO/INDEX.md` §12.3 P3 mitigation)
**Target PI:** 北京协和医学院 (Peking Union Medical College) 医院期细胞毒理学课题组
**Authoring date:** 2026-09-27
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`

---

## 1. 目的 (Purpose)

本意向书 (Letter of Intent, LOI) 提议 Mol-Metal 开源项目与北京协和医学院
医院期细胞毒理学课题组开展短期合作验证。合作将对 Mol-Metal R18 版本
筛选出的前 20 名金属基药物候选分子进行 **NCI-60 9-cell 完整面板**
医院期细胞毒性 (SRB) + ICP-MS 摄取定量评价。

## 2. 背景 — Mol-Metal 提供的资源

Mol-Metal 是一个开源金属基药物设计流水线 (mol-metal.org)，
R18 版本 (2026-09) 已经发布：

- **点击化学金属基药物生成器** (`molmetal/molmetal_lam/lam_chem/pt_click_compat.py`).
- **λ-组合子搜索 + 金属感知奖励塑形** (`molmetal/molmetal_lam/search_alg/proof_search.py`).
- **耦合 flow-matching 主干网络** (`molmetal/adapters/flow_matching_lipman/`).
- **R20-P6 前 20 名候选分子** (`molmetal/reports/wf58_wetlab_dryrun/top20_candidates.csv`):
  Pt 2 / Pd 3 / Au 1 / Ru 3 / Ir 11。
- **跨 4-cell 基准 D-MPNN pIC50 预测** (Wf-58 P6.1)。
- **NCI-60 9-cell 测点 SOP** (`molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` WLP-CYTO-001) +
  **ICP-MS 摄取 SOP** (`molmetal/docs/wetlab_protocols/cellular_uptake_assay.md` WLP-UP-001)。
- **IC50 分析 pipeline 模板** 准备好接收湿实验室数据。

## 3. 协和提供的资源

协和医院具备医院期 (hospital-phase) 临床前研究能力：

- **NCI-60 9-cell 完整面板**：Leukemia CCRF-CEM / Lung A549 / Colon HCT-116 / CNS SF-539 / Melanoma LOX-IMVI / Ovarian OVCAR-3 / Renal A498 / Prostate PC-3 / Breast MCF7。
- 医院 IRB + 临床前 GCP 流程 + mycoplasma 检测能力。
- ICP-MS 仪器 (Thermo iCAP RQ / Agilent 7900) + 微波消解 (CEM MARS 6)。
- BSL-2 细胞培养设施 + 受训操作员。
- 临床案例报告 (case report) 预备能力，为 lead compound 临床转化奠基。

## 4. 提议合作框架 (3 阶段，6-12 个月)

| 阶段 | 时长 | Mol-Metal 提供 | 协和提供 |
|---|---|---|---|
| **Phase 1** | 0–2 月 | 5–10 个 Pt/Ru/Ir 候选分子 (3D 结构、in-silico IC50/logP/redox/HSA K_d/摄取); 分析 pipeline 模板 | IRB approval + 细胞系认证 + mycoplasma 测试; 反向签 MoU; 9-cell panel 启动 |
| **Phase 2** | 2–6 月 | 化合物 aliquots 发货; 每轮更新排序 | SRB/MTT 全 9-cell panel: 9 cell × 20 化合物 × 8 dose × 3 replicate = 4,320 数据点; ICP-MS 摄取 (Pt/Pd/Au/Ru/Ir) 在 ≥2 cell × 3 time-point = 180 个 ng/cell 数据点 |
| **Phase 3** | 6–12 月 | 联合 manuscript + clinical case-report 预备; 模型 retrain | Lead compound 临床转化准备 (PK/PD 桥接实验); 可选 in-vivo xenograft (NSG 小鼠) |

60-day cadence: Day 0/7/21 (邮件跟进) → Day 28 (intro Zoom) → Day 60 (MoU signed) → Month 2/6/9 (阶段评审)。

## 5. 署名 (Authorship)

- Phase-2 验证完成 **NCI-60 9-cell 完整 panel** → 协和 PI 获 **共同通讯作者 (co-corresponding)** 署名 + **医院名称列入 cover sheet**。
- Phase-2 验证完成 9-cell 子集 (≥4 cell) → 协和 PI 获 **中位作者** + 数据验证章节署名。
- 双方在投稿前书面约定作者顺序。
- 预印本 / 会议摘要：dry-lab PI 与 协和 PI 联合展示；摘要第一作者按实验贡献决定。

## 6. 经费与设备

- **Mol-Metal 一方**：软件、算力、in-silico 预测、manuscript 准备。**无湿实验室预算**。
- **协和一方**：医院期试剂、IRB + 临床前 GCP 流程、9-cell panel + 培养耗材、ICP-MS + 微波消解时间、操作员人力。
- **联合基金申请鼓励** (NSFC / 科技部 / 国家自然科学基金重大研究计划)。以 协和医院 为财务主体；Mol-Metal 可提供支持信 + 1 页研究计划。
- **无管理费 / 间接成本划拨**。

## 7. NCI-60 9-cell 面板 (Phase 2 完整)

| Cell line | Tissue | 备注 |
|---|---|---|
| **CCRF-CEM** | Leukemia | 顺铂高度敏感 |
| **A549** | Lung (NSCLC) | 顺铂中等敏感 |
| **HCT-116** | Colon | MSI-high 模型 |
| **SF-539** | CNS | 跨 BBB 候选筛选 |
| **LOX-IMVI** | Melanoma | 顺铂耐药参考 |
| **OVCAR-3** | Ovarian | 顺铂敏感 |
| **A498** | Renal | 顺铂耐药参考 |
| **PC-3** | Prostate | 顺铂耐药参考 |
| **MCF7** | Breast (ER+) | 顺铂敏感 |

## 8. ICP-MS 摄取面板 (Phase 2 完整)

| Cell line | Time-point | 检测目标 |
|---|---|---|
| **A549** | 1h / 6h / 24h | Pt / Pd / Au / Ru / Ir ng/cell |
| **MCF7** | 1h / 6h / 24h | 同上 |
| (可选) **A2780cis** | 1h / 6h / 24h | 顺铂耐药 vs 敏感对比 |

## 9. 期望输出 (Expected outputs)

- 每个 (compound × cell) IC50 ± std (n=3 biological replicates, 8-dose 半对数网格)。
- 每个 (compound × cell × time-point) ICP-MS ng-metal / 1e6 cells。
- per-metal 跨 9-cell 的 GI50 profile (与 NCI-60 DTP 数据库公开数据对比)。
- 跨 panel COMPARE analysis (mean graph)。
- 原始 plate-reader + ICP-MS 文件上传至 `data/wetlab/{cytotox,uptake}/<run_id>/` 含 checksum。
- JSON 输出符合 `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` schema。

## 10. 时间表 + 交付物

| 月份 | Milestone | 交付物 |
|---|---|---|
| 0 | IRB 审批 + MoU 签署 | IRB + MoU |
| 1 | 细胞系认证 + mycoplasma 测试 + 9-cell 启动 | 认证报告 |
| 2 | 第一个 5 个候选 9-cell IC50 测定 | 5 × 9 = 45 IC50 |
| 3 | 第二个 5 个候选 9-cell IC50 测定 | + 45 IC50 |
| 4 | 第三个 5 个候选 9-cell IC50 测定 | + 45 IC50 |
| 5 | 第四个 5 个候选 9-cell IC50 测定 | + 45 IC50 |
| 5–6 | ICP-MS 摄取 60 个样品测定 | 60 ng/cell 数据点 |
| 6 | Phase-2 评审; lead compound 选定 | 9-cell summary report |
| 6–9 | Phase-3 lead compound 临床转化准备 | 临床案例 + xenograft 可选 |
| 9–12 | 联合 manuscript + clinical case report | Joint paper |

## 11. 风险与缓解 (Risk and mitigation)

- **NCI-60 9-cell 工作量大**：4,320 数据点 + 180 ICP-MS 数据点 ≈ 6 个月 wet-lab 工作。若 协和 工作量大，可与 PKU LOI 分工 (PKU 负责 4-cell 基准 + 部分 NCI-60；协和 负责 NCI-60 全 panel + ICP-MS 摄取)。
- **医院 IRB 周期**：中国医院 IRB 通常需 4–8 周；建议 Day 0 同步启动 IRB + MoU。
- **ICP-MS 灵敏度**：Ir 与 Ru 的 LOD ≤ 0.01 ng/mL；如低于此值，flag 候选分子可能低摄取。

## 12. 保密与数据共享 (Confidentiality and data sharing)

- 所有湿实验室数据按 `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` MoU 模板共享。
- 联合预印本 embargo 至双方 PI 同意投稿。
- Mol-Metal 所有 R18 / R20 代码均开源 (CC-BY-4.0); 湿实验室 IC50 + 摄取数据在论文发表后成为 Mol-Metal 开放数据集的一部分。

## 13. 下一步 (Next steps)

1. Hugo Lee 用 `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §4 中文模板发送本 LOI 给 协和 PI。
2. Day +7 轻量跟进；Day +21 第二次跟进。
3. Day +28 intro Zoom (30 分钟) — 确认契合度、IRB 流程、9-cell panel 可用性。
4. Day +60 MoU 双方签字；Phase 1 启动 (同步 IRB 启动)。

## 14. 诚实标注 (Honest framing per party-twenty-spirit §3.5)

- 本 LOI 是 **草案** (draft)，等待 Hugo Lee 审阅；**本版本 0 LOI 已发送**。
- **当前 0 个湿实验室 IC50 / 摄取实测**；in-silico D-MPNN pIC50 来自合成 IC50 labels (Rusanov 2025)，作为 **rank-only**。
- 头条 `Pearson r ≥ 0.7` (D-MPNN 预测 vs 协和 实测) **当前无法 MEASURE**；需 ≥1 PI 回复 + IRB + MoU + Phase 1 启动 + 6 个月时间线完成 (最早 2027 Q1)。
- Paper §5 湿实验室验证章节 🚫 **DEFERRED** per TODO-84 paper rule; 需要 explicit user authorization。

## 15. 交叉引用 (Cross-references)

- `molmetal/docs/wetlab_protocols/collaborator_outreach_template.md` §4 中文模板
- `molmetal/docs/wetlab_protocols/cytotoxicity_assay.md` (Wf-28 通用 SRB 协议, WLP-CYTO-001)
- `molmetal/docs/wetlab_protocols/cellular_uptake_assay.md` (Wf-28 通用 ICP-MS 摄取协议, WLP-UP-001)
- `molmetal/docs/wetlab_protocols/data_sharing_agreement.md` (Wf-28 数据共享 MoU 模板)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (配套 IC50 分析 pipeline)
- `TODO/pending/wetlab_sop_cytotoxicity_assay.md` (NEW 机构定制 SOP)
- `TODO/pending/wetlab_sop_cellular_uptake_assay.md` (NEW 机构定制摄取 SOP)
- `TODO/completed/97b_wetlab_prework_docs.md` (v1 夸大 close-out — 由 97c 修正)
- `TODO/completed/97c_wf74_reclassification.md` (本 Wf-74 v2 close-out)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 计划)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run layer ship)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper 规则; outreach 不受 gating)

## 16. 修订历史 (Revision history)

| Rev | Date       | Author | Notes                                              |
|-----|------------|--------|----------------------------------------------------|
| 1.0 | 2026-09-27 | Wf-74  | 机构定制 LOI 草案 (等待 Hugo Lee 审阅; **0 已发送**)。 |
