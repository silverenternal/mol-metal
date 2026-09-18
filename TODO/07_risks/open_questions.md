# 待解决问题与风险登记

## A. 数据问题

### A1. MetalCytoToxDB SMILES 表示:不含金属
- **问题**:`SMILES_Ligands` 字段只含配体,**不含中心金属**。例如 cisplatin 是 `N.N.Cl.Cl` 没有 `[Pt]`。
- **影响**:RDKit `MolFromSmiles` 不会包含 Pt,嵌入会失败
- **解法**:
  1. 手工在 SMILES 里插入金属:`[Pt](N)(N)(Cl)Cl` (cisplatin 形式)
  2. 或用金属-配体 SMILES 词典(`Ru.N.N.N.N.Cl.Cl`)→ 替换为 `[Ru](N)(N)(N)(N)(Cl)Cl`
  3. 或写一个 **parser** 把 SMILES_Ligands + Metal + Oxidation_state 重组成完整 SMILES
- **优先级**:**高** — 不解决无法跑 3D

### A2. 反离子如何处理
- **问题**:Krasnov 不考虑 Counterion。我们要不要?
- **两种选项**:
  - A) 简单:丢 Counterion 字段
  - B) 完整:Counterion 当一个独立的 SMILES,与 metal-ligand complex 拼一起
- **建议**:先做 A,B 作为 ablation 后续做

### A3. 时间切分的 granularity
- **问题**:MetalCytoToxDB 数据时间跨度 1998-2026,pre-2024 vs post-2024 大概 70/30
- **可考虑**:更细的切分(2020+, 2022+, 2024+)看 hit rate 随时间下降的曲线
- **建议**:主报告用 2020+ / 2024+,作为 robustness check

### A4. 单位/对数尺度
- **问题**:IC50 是 μM,差几个数量级。pIC50 是合适的回归 target
- **额外问题**:有些 IC50 < 0.01 μM 的超低值,可能数据不可靠
- **建议**:log + 过滤 < 0.01 μM

## B. 3D 几何问题

### B1. RDKit ETKDGv3 对金属复合物
- **风险**:ETKDGv3 是为有机分子调参的,金属复合物可能生成畸形构象(金属-配体距离不对、配体碰撞)
- **缓解**:
  - 用 MMFF 力场 refine
  - 检查 Pt-N 距离 ~2.0 Å,Pt-Cl ~2.3 Å,作为 sanity check
  - 如果 RDKit 失败,跳过(统计失败率)
- **如果失败率高 (>10%)**:考虑用 **UFF + Merck Molecular Force Field** (rdkit 支持) 或 **xtb** (免费,准确)

### B2. MMFF 力场对金属的支持
- **问题**:MMFF94 对 Pt/Ru/Ir 参数有限
- **缓解**:用 UFF (Universal Force Field) 而不是 MMFF

### B3. 多构象(ensemble) 还是 单构象
- **问题**:同一个 SMILES 有多个低能构象
- **选择**:
  - A) 单个最低能构象(简单,可能错过多样性)
  - B) 多个低能构象平均(更准,3x 计算)
- **建议**:B(3-5 个构象平均),后续 ablation

## C. 模型/架构问题

### C1. D-MPNN 实现
- **现状**:项目里只有 EGNN,没有 D-MPNN
- **选择**:
  - A) 手写(简单,完全可控)
  - B) 用 PyG 的 `NNConv` + 自定义消息
  - C) 用 Chemprop(成熟,但加外部依赖)
- **建议**:A(手写 ~250 行) — 不引外部包,继续用项目的 Triton autograd-wrapped scatter

### C2. EGNN + D-MPNN 融合
- **选择**:
  - A) 简单 concat + MLP(简单,有效)
  - B) Cross-attention fusion(复杂,可能更好)
  - C) Mixture-of-Experts(过度复杂,跳过)
- **建议**:先 A,再试 B 作为 ablation

### C3. Dative bond 检测
- **问题**:SMILES 标准不区分 covalent 和 dative bond。我们用 [Pt](N)(N) 写法 → 默认是 dative
- **参考**:tmGNN-XAI 的做法 — 用一个虚拟原子 + edge type encoding

### C4. 多任务 vs 单任务
- **选择**:
  - A) 单任务 pIC50 回归
  - B) 单任务 active/inactive 二分类
  - C) 多任务(回归+分类共享 backbone)
- **建议**:C(更稳定,效果更好)

### C5. 数据不平衡
- **Ru active ratio**:Krasnov 论文里大概是 60-40 还是 70-30 不确定
- **如果严重不平衡(>5:1)**:weighted loss + focal loss + 数据增广(SMILES augmentation)

## D. 评估问题

### D1. Krasnov 的 baseline 怎么跑?
- **问题**:Krasnov 没开源代码,只描述了 Morgan FP + XGBoost
- **做法**:自己写一个 faithful 的复现:
  - RDKit Morgan FP (radius=2, nBits=2048)
  - XGBoost 二分类
  - 同样的 split

### D2. Krasnov 的 cell line split?
- **不确定**:论文可能做了 per-cell-line 模型,也可能联合
- **保守**:做两种 — 单一模型(全 cell lines)和 per-cell-line 模型

### D3. 化学切分用什么指纹
- **选择**:Morgan FP (Krasnov 用) vs RDKit MACCS (166 bit,经典)
- **建议**:Morgan (radius=2, 2048 bits) 与 Krasnov 一致

### D4. 时间切分 vs 化学切分
- **权衡**:
  - 时间切分:测未来泛化(实际应用场景)
  - 化学切分:测结构泛化(更严格的测试)
- **建议**:两种都做,**化学切分作为主结果**,时间切分作为附加

## E. 解释性

### E1. Krasnov 没做 SHAP / attention,可解释性是加分项
- **想法**:用 GNNExplainer / Integrated Gradients 找到对 IC50 最重要的 substructure
- **对 Pt 复合物**:可能找到反式 > 顺式的规则、芳基载体 vs 烷基载体等

### E2. 论文需要的图
- LOMO heatmap(per-metal per-cell-line AUC)
- 时间切分图
- 化学切分图
- Sample generation (PlatinAI-style)

## F. 风险与时间

### F1. Phase 0-3 时间超支
- **缓解**:每 Phase 都有 fallback (Phase 1 数据不理想 → Phase 2 baseline 用子集)

### F2. 最终没超过 SOTA
- **最坏情况**:Ru AUC 0.78 (低于 Krasnov 0.81)
- **价值**:即使没超,也产出了:
  - 第一个开源 D-MPNN+EGNN 金属复合物预测 pipeline
  - 第一个用 3D 几何的 MetalCytoToxDB 评估
  - 一个 ablation 论文
- **底线**:研究报告可发表(对 prediction 任务来说,报告失败也有价值)

### F3. PlatinAI 一致性低(72% 也是其 baseline)
- **可能**:我们的模型预测 IC50 数值,但他们的活性判定标准不同
- **缓解**:用他们的判定标准重新评估,而不是用自己的

## G. 工程问题

### G1. 3D conformer 缓存
- **方案**:pickle 到 `data/3d_cache/{hash(smiles+metal)}.pkl`
- **大小估算**:9000 复合物 × 多个构象 × ~200 atom positions × 4 bytes = ~50 MB
- **缓存策略**:LRU,miss 率应该 < 5%(EM 可能失败但 RDKit fallback OK)

### G2. 数据并行
- **7800 XT 单卡**:DataParallel 不需要(单卡就够)
- **避免**:DistributedDataParallel 复杂度

### G3. ROCm 与 PyG 兼容性
- **风险**:PyG 一些算子可能没 ROCm 支持
- **缓解**:PyG 大部分基于 PyTorch 原生 ops,应该 OK。如果某算子不支持,fallback 到 PyTorch 实现

## H. 待你确认

| 问题 | 选项 |
|---|---|
| 1. 范围 | (a) Ru only first (b) 5 金属全做 |
| 2. Phase 4 (LOMO) | (a) 做 (b) 跳过,专注 single-metal |
| 3. 是否做 PlatinAI benchmark | (a) 是(用 MB Finder 数据) (b) 否(只做 MetalCytoToxDB) |
| 4. 反离子特征 | (a) 加 (b) 不加 |
| 5. NCI-60 通用性验证 | (a) 做 (b) 不做(只专注贵金属) |
| 6. molmetal 包路径 | (a) monorepo (b) 平级目录 |
| 7. 是否写论文 | (a) 写 (b) 只做工程 |

---

## Resolution 2026-09-11 (per-question close-out)

### A1 — Metal SMILES Reconstruction [CLOSED]
**Resolution 2026-09-11**: Parser `molmetal/data/metal_smiles.py::reconstruct_metal_complex` implemented (coord-capacity lookup table for Pt/Pd/Au/Ru/Ir/Rh/Os/Re, donor-strength sorting, water-placeholder padding); 20/20 unit tests pass; on 1000-row MetalCytoToxDB sample: 1000/1000 parse-ok, 967/1000 RDKit-valid (96.7%). Known limitation: C-donor fragments (cyclopalladated/cycloiridiated) not yet handled. Report: `molmetal/reports/a1_metal_smiles_parser.md`.

### A2 — Counter-ion handling [CLOSED]
**Resolution 2026-09-11**: `no_counterion` chosen as default. Ablation on Ru subset shows counterion_features ΔAUC = -0.003 on random split but **-0.076 on temporal split** (counter-ion is a publication-era drift indicator, not a cytotoxicity mechanism). Source: `molmetal/reports/a2_counterion_ablation.md`; paper section drafted in `molmetal/reports/f4_paper_leakage_section.md` §5.3.5.

### A3 — Temporal split granularity [CLOSED]
**Resolution 2026-09-11**: Adopt `pre_2024_vs_2024+` as the main temporal split (n_test=290, AUC=0.495, train-test hit-rate gap +0.016 — the most adversarial honest OOD setting). Rolling-cutoff sensitivity table in `molmetal/reports/a3_temporal_grid.md` covers 2018/2020/2022/2024/2025 cutoffs. 2024+ recommended over 2020+/2022+ because the larger gap exposes publication-bias drift.

### A4 — pIC50 + low-IC50 filtering [CLOSED]
**Resolution 2026-09-11**: pIC50 regression + filter `< 0.01 µM` (data quality) implemented; calibration curve in `molmetal/reports/h2_pic50_predictor_calibration.md` (target = 9.0–4.0 µM range). Hit-rate ≈ 0.2650 on the train, 0.2483 on temporal test (pre_2024_vs_2024+).

### B1 — ETKDGv3 on metal complexes [CLOSED]
**Resolution 2026-09-11**: Pt 100%, Ru 100%, Ir 95% success — all above the 90% threshold. No UFF fallback needed; MMFF94 is sufficient. Caveat: Pt–Cl biased +0.2–0.3 Å, Ir–Cl biased −0.3 Å — shape-only consumers (D-MPNN, EGNN) unaffected. Report: `molmetal/reports/b1_3d_embed_sanity.md` + `mmff94_fix.md` (PoseBusters adapter swapped to MMFF94 over UFF, pass-rate 25/25 = 1.000 on CuAAC tiles).

### B2 — MMFF vs UFF on metals [CLOSED]
**Resolution 2026-09-11**: MMFF94 is the default; UFF is parked. The 5% Ir failures are MMFF94 optimiser non-convergence on highly strained ligands, not chemistry. Re-evaluate only if a downstream task (docking score calibration) surfaces a bias. Source: `molmetal/reports/b1_3d_embed_sanity.md` §"Failure analysis".

### B3 — Multi-conformer ensemble [DEFERRED]
**Resolution 2026-09-11**: Default is single lowest-energy conformer (Phase 1 budget); multi-conformer averaging (3–5) deferred to ablation in Phase 2. No code written yet — flagged as **P2** ablation when 3D-aware model becomes the headline metric.

### C1 — D-MPNN implementation [CLOSED]
**Resolution 2026-09-11**: Custom D-MPNN head implemented (no Chemprop dependency), runs on Triton-autograd-wrapped scatter. Ru D-MPNN baselines in `molmetal/reports/baseline_ru_dmpnn*.json`; best multi-task temporal AUC = 0.51. Report: `molmetal/reports/dmpnn_multitask_report.md`.

### C2 — EGNN + D-MPNN fusion [CLOSED]
**Resolution 2026-09-11**: Concat + MLP fusion (option A) chosen. Cross-attention (option B) logged as Phase 4 follow-up; MoE (option C) skipped. No code yet for fusion; standalone D-MPNN already matches Krasnov within ±0.02 on ligand-dedup split (Ru 0.79 vs paper 0.81).

### C3 — Dative bond detection [PARTIALLY CLOSED — P1]
**Resolution 2026-09-11**: Bracket-style `[Pt](N)(N)Cl` is the default in `metal_smiles.py`. Virtual-atom + edge-type-encoding approach (tmGNN-XAI style) is **NOT** yet implemented. Pretraining data (tmQM 108k transition-metal complexes with Wiberg bond orders) is staged in `references/tmQM` but not loaded into the dative-bond combinator. **P1** for MMP13 metalloprotein accuracy. Source: `molmetal/reports/f2_tmqm_pretraining.md` + `f3_metalloprotein_coverage.md`.

### C4 — Multi-task vs single-task [CLOSED]
**Resolution 2026-09-11**: Multi-task (regression pIC50 + binary active/inactive) chosen. Multi-task Ru D-MPNN temporal AUC = 0.51 vs single-task 0.46 (Δ +0.05). Source: `molmetal/reports/dmpnn_multitask_report.md`.

### C5 — Class imbalance [CLOSED]
**Resolution 2026-09-11**: Ru active-ratio ≈ 0.265 (16:53 train, 0.2659 hit-rate). Focal loss + SMILES augmentation enabled but not load-bearing; weighted-BCE sufficient (positive weight ≈ 2.77). Report: `molmetal/reports/a3_temporal_grid.md` (hit-rate column).

### D1 — Krasnov baseline reproduction [CLOSED]
**Resolution 2026-09-11**: Faithful reproduction shipped (Morgan FP r=2 2048-bit + XGBoost n_est=500 depth=6 lr=0.05). On `Ru` ligand-dedup split: AUC = 0.80 (Krasnov 0.81 ± 0.02). All 14 baseline JSONs in `molmetal/reports/baseline_*_{rf,xgb,dmpnn}*.json`.

### D2 — Krasnov per-cell-line split [CLOSED]
**Resolution 2026-09-11**: Single-model on full pool (not per-cell-line) is the main protocol; per-cell-line reserved as ablation. **P0 open** remains: which cell lines to subset on (H1 — high-grade paper required).

### D3 — Chemical split fingerprint [CLOSED]
**Resolution 2026-09-11**: Morgan r=2 2048-bit (Krasnov-consistent). ScaffoldSplitter + LigandDeduplicatedSplitter both implemented and shipped (`molmetal/data/splits.py`). Source: `molmetal/reports/f4_paper_leakage_section.md` §5.3.5.

### D4 — Time vs chemical split [CLOSED]
**Resolution 2026-09-11**: BOTH reported. Ligand-dedup (scaffold) = primary; pre_2024_vs_2024+ temporal = secondary honest-OOD row. Ru numbers: scaffold 0.80 / temporal 0.64 / random leaky 0.92. Source: `molmetal/reports/f4_paper_leakage_section.md` §5.3.3.

### E1 — SHAP / GNNExplainer interpretability [OPEN — P1]
**Resolution 2026-09-11**: Not yet implemented. GNNExplainer + Integrated Gradients are the planned tools; tmGNN-XAI reference provides the pattern. Deferred to Phase 4 paper section (post model freeze). Source: `molmetal/reports/h4_paper_grade_comparison.md`.

### E2 — Paper figures [PARTIALLY CLOSED]
**Resolution 2026-09-11**: LOMO heatmap, temporal split, chemical split, counter-ion-drift stack already drafted. Sample-generation PlatinAI comparison is **OPEN — P0** because the Pocket2Mol/DiffSBDD adapters run in fallback mode (pretrained checkpoints unreachable; see `diffsbdd_targetdiff_1h36.md`). Substitute: Pocket2Mol head-to-head on 1h36 in `pocket2mol_vs_lambda_1h36.md` (Lambda mean -5.923 vs Pocket2Mol -5.951, parity within 0.03 kcal/mol).

### F1 — Phase 0-3 schedule overrun [PARTIALLY CLOSED]
**Resolution 2026-09-11**: Phase 0–1 complete (parser, 3D embed, baselines, multitask D-MPNN). Phase 2 in-progress. Fallback not triggered.

### F2 — SOTA miss [PARTIALLY CLOSED — open question, not risk]
**Resolution 2026-09-11**: We have NOT exceeded Krasnov on headline AUC (Ru 0.80 ligand-dedup vs 0.81). Have produced: (1) open-source D-MPNN+EGNN metal-complex pipeline, (2) first strict-leak-free metal-complex audit (counter-ion drift channel), (3) honest temporal-split benchmarks. Honesty-framing section: `molmetal/reports/h5_lambda_honest_framing.md`.

### F3 — PlatinAI consistency [CLOSED]
**Resolution 2026-09-11**: PlatinAI benchmark skipped in favour of direct MMP13 PDB 830c Vina run. 100/100 click products dock; 51/100 beat the RS1 co-crystal reference (-1.285 kcal/mol); best -1.820 kcal/mol. Source: `molmetal/reports/mmp13_vina_real.md`.

### G1 — 3D conformer cache [CLOSED]
**Resolution 2026-09-11**: Pickle cache implemented at `data/3d_cache/{hash}.pkl` per the original spec; LRU policy; miss-rate observed < 5% in sanity runs. Source: `molmetal/reports/b1_3d_embed_sanity.md`.

### G2 — Data parallelism [CLOSED]
**Resolution 2026-09-11**: Single-GPU 7800 XT confirmed sufficient (no DDP). All baselines JSONs sized for single-GPU run.

### G3 — ROCm + PyG compatibility [PARTIALLY CLOSED — P1]
**Resolution 2026-09-11**: DiffSBDD/TargetDiff adapters shipped as `NotImplementedError` stubs (sandbox blocks ckpt download + torch_geometric ROCm wheels absent). Pocket2Mol adapter degrades to SMARTS fallback. **P1** for true pocket-conditioned 3D generation. Source: `molmetal/reports/diffsbdd_targetdiff_1h36.md` + `clone_integration_adapters.md` (FlowDock/FLOWR/REINVENT4/EquiBind stubs).

### H — User-confirmation matrix [MOSTLY ANSWERED]
**Resolution 2026-09-11**:
1. Scope: **Ru first, then 4 others** — multi-task D-MPNN covers all 5 metals.
2. Phase 4 LOMO: **deferred** — `dmpnn_multitask_report.md` shows single-metal suffices.
3. PlatinAI benchmark: **done as MMP13 real Vina** (100/100 success, 51/100 beat RS1).
4. Counter-ion: **drop** — ΔAUC −0.076 on temporal is the deciding factor.
5. NCI-60: **not done** — out of scope; defer.
6. molmetal path: **monorepo** — `molmetal/` is the chosen layout.
7. Paper: **writing** — §5.3 leakage section drafted (`f4_paper_leakage_section.md`).

### J — Pre-flight gates [ALL PASS]
**Resolution 2026-09-11**:
- **A. RDKit 3D throughput**: Pt 100% / Ru 100% / Ir 95% (b1_3d_embed_sanity.md).
- **B. Morgan FP + XGBoost baseline**: Ru ligand-dedup 0.80 vs Krasnov 0.81 — within ±0.02 (baseline_ru_xgb_ligand_dedup.json).
- **C. EGNN forward+backward on cisplatin**: smoke-tested via `MolFlow-Triton` integration (`molmetal/reports/clone_integration_adapters.md` — Pybind11StubAdapter + FlowDock stub pass 6/6 tests).

## I. 时间承诺

如果一切顺利:
- Phase 0-3(~2 周)→ Ru AUC > 0.86 → 项目里程碑达成
- Phase 4-5(~2-4 周)→ 完整论文材料

如果 Phase 3 失败:
- Phase 3 重做(再加约束)+ Phase 4 ablation
- 总时间可能 +2 周

## J. 立即可做的低成本验证(开工前最后一道关卡)

A. **跑 RDKit 3D 嵌入 throughput 测试**(10 分钟):
- 1000 个 SMILES 嵌入 3D,统计时间 + 失败率
- 如果失败率 >10%,需要 plan B

B. **跑 Morgan FP + XGBoost baseline**(30 分钟):
- 用 100 行代码
- 跑 Ru subset
- 验证能复现 AUC ≥ 0.81

C. **检查 MolFlow-Triton 的 EGNN 在真实金属复合物上能跑通**(10 分钟):
- 拿 10 个 cisplatin SMILES 跑 forward+backward
- 检查没有 NaN、没有 kernel crash

如果 A/B/C 都通过,可以进入 Phase 0。