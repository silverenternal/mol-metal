# TODO-133: Paper §6 honest-negative drop list (Q2 paper-ready)

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P0** — Q2 paper §6 limitations 必须诚实 ship 8 个 honest-negative
**Parent**: TODO-131 weak metric lift master plan §4 + TODO-132 paper §4 rewrite §6
**Target**: JCIM IF 5.6 / Bioinformatics IF 4.8 — Q2 中科院二区
**Skill alignment**: `party-twenty-spirit` §3.5 (honest negative 0-tolerance)

---

## §0 Why this TODO exists

Q2 paper §6 limitations 接受 honest-negative(数量 5-8),但需要显式 ship 不可隐含。本文档规划 8 个 honest-negative 怎么写。

**不假装 ship 的原则**(per party-twenty-spirit §3.5):
- 任何 R²/RMSD/Vina lift 没真正 ship, 不假装在 paper §4 表中 lead claim
- 任何 metric 没真正写 paper §, 不假装 in §4
- 任何 GPU retrain 没收敛, 不假装 ship

## §1 8 个 honest-negative 总览

| # | Metric | 数值 | Reason honest-negative | Wf source |
|---:|---|---:|---|---|
| 1 | **PlatinAI 4-cell combined R²** | -5.66 ± 1.36 | 12-d RDKit signal floor 架构性 ceiling | Wf-77/79/83/122/127/128/129 |
| 2 | **L5 Hilbert+ASTSR R²** | -4.04 ± 4.31 (mcf7) | Wf-127 P4 ship 后仍 negative | Wf-127 |
| 3 | **L7 multi-PH + PLL R²** | -0.078 ± 0.015 | Wf-129 P3 ship 后仍 negative | Wf-129 |
| 4 | **L10 3-way R²** | -0.36 (Wf-91 synthetic) / -40.24 (Wf-128 P3 29-element aggregate) | Wf-128 172-d + multi-task 后仍 negative | Wf-41 / Wf-91 / Wf-128 |
| 5 | **m3 Vina cisplatin** | -3.103 ± 0.351 (10 pockets × 3 seeds) | cisplatin 是 covalent binder, Vina 物理模型看不到 covalent bond | Wf-121 |
| 6 | **diversity_tanimoto** | 0.137 vs TargetDiff 0.860 (gap -0.723) | 引 n_distinct 替代; diversity_tanimoto 受 pool-saturated ceiling | R12 + EV-3 |
| 7 | **m15 RMSD (旧定义)** | 1.241 (self-RMSD between 2 conformers of same molecule) | metric 设计问题,被 Phase D 替换为 docked_pose_rmsd | Wf-126 P1 honest finding |
| 8 | **Hilbert 4-cell R² (Wf-83 original)** | -6.71 | Wf-127 P4 部分改善但仍 negative; Wf-129 guards ship 后仍 honest-negative | Wf-83 / Wf-127 / Wf-129 |

## §2 §6 写作模板(每个 honest-negative 一段)

### §6.1 PlatinAI 4-cell combined R²

**§6 writing**:
> "On the PlatinAI 4-cell synthetic pIC50 dataset, our combined Hilbert+ASTSR evolutionary regression achieves combined R² = -5.66 ± 1.36 across 4 cell lines (mcf7 / a549 / a2780 / a2780cis). Three rescue attempts with Wasserstein regularization (Wf-129), Hilbert L2 + ASTSR scale guards (Wf-127, Wf-122), and 172-d Multi-PH + PLL feature augmentation (Wf-128) all yielded honest-negative results (-4.04, -0.078, -40.24 respectively). Root cause is the 12-d RDKit descriptor signal floor — none of our 13 mathematical theorems (Wasserstein gradient flow, Hilbert primal-dual, ASTSR evolutionary search, Multi-PH Fourier, PLL signature) provides sufficient signal lift to break the architectural ceiling. Future work: real wet-lab pIC50 (R23-P6) + AlphaFold-Multimer integration (R22-P1) for protein-aware descriptors."

### §6.2 L5 Hilbert+ASTSR R² (mcf7)

**§6 writing**:
> "Adding L2 regularization to Hilbert primal-dual and scale-clip to ASTSR signature (Wf-127 Phase 4) prevents divergence but does not lift PlatinAI mcf7 R² from the baseline of -6.71. With guards, mcf7 R² = -4.04 ± 4.31; a549 R² = -1.57 ± 1.18. Notably, ASTSR scale-clip with default range (-10, 10) **regresses** ASTSR R² on mcf7 from -0.014 (no clip, Wf-86 P3 baseline) to -3.54 (with clip) — the clip forces atypical scaffold predictions to the boundary, hurting honest fits."

### §6.3 L7 multi-PH + PLL R²

**§6 writing**:
> "End-to-end Multi-PH + PLL feature extraction combined with Hilbert L2 guard + ASTSR scale guard (Wf-129 Phase 3) achieves PlatinAI 4-cell combined R² = -0.078 ± 0.015 (5 seeds × 5 folds × 4 cell lines), vs the Wf-79 P2 baseline of +0.087 (NCI-60 real pIC50, different cohort). The honest-negative result confirms that Wf-127 + Wf-122 + Wf-131 guards prevent divergence (no `r2 = -inf`; all L2 penalties finite ≥ 0; ASTSR guard never fires on widened band [0.05, 20.0]) but cannot lift R² ≥ +0.30 at the architectural level. The Wf-128 172-d Multi-PH + PLL feature augmentation with multi-task backbone (118,789 params) is the only architectural lever that may break the 12-d signal floor; Wf-128 P3 29-element aggregate R² = -40.24 confirms this lever also fails."

### §6.4 L10 3-way R²

**§6 writing**:
> "The 3-way Deflexpressor + Multi-PH + ASTSR integration on real MetalCytoToxDB 5-metal pIC50 (Wf-128 P3) yields honest-negative results across 5 precious metals (Ru/Ir/Os/Rh/Re) with 172-d feature stack (RDKit 12-d + Multi-PH 32-d + PLL 128-d) and multi-task backbone (118,789 params). On the Wf-41 P1 3-way baseline (PlatinAI synthetic `qed * qed` target), R² = -0.36 ± 0.28; on the Wf-128 P3 29-element aggregate, R² = -40.24 (uncomputable target signal). Both targets confirm the architectural signal floor — 12-d RDKit descriptor cannot be lifted to R² ≥ 0 by any combination of Wasserstein + L2 + scale-guard + 172-d features + multi-task backbone."

### §6.5 m3 Vina cisplatin

**§6 writing**:
> "Cisplatin real Vina docking sweep (Wf-121 Phase 3) on 10 pockets × 3 seeds yields mean -3.103 ± 0.351 kcal/mol (best -3.734 on 3ny8, worst -2.549 on 4rn0). All 10 pockets show kcal/mol > -5.0, well above the industry binding threshold of ≥ -7.0 kcal/mol. Root cause: cisplatin is a covalent binder (aquation + DNA N7 guanine crosslink); Vina's non-covalent physics model cannot represent the covalent mechanism. Additionally, our Vina 1.2.7 setup substitutes 'Mg' atom-type for 'Pt' to bypass strict-parser rejection (P1 SMARTS 'M' rejected), introducing generic VDW underestimation. True cisplatin binding requires covalent docking tools (DockCov, CovDock) and wet-lab IC50 — both deferred to R23-P6. Paper §4 main m3 metric uses protein-binder pool (organic drug-like, non-covalent), not cisplatin."

### §6.6 diversity_tanimoto

**§6 writing**:
> "Pool-wide Morgan-ECFP4 Tanimoto distance on Lambda-MCTS generated molecules reaches 0.137 (EV-3, 30 cells) vs TargetDiff 0.860 — a gap of -0.723 reflecting pool-diversity ceiling (the underlying fragment pool is saturated). The structurally meaningful diversity metric is n_distinct (count of unique generated molecules per cell), which we lift from 1 → 20 via the 4-fix bundle (rule-symmetry + chem-aware decoder + auto-pt-strict gate + partner tiles). Paper §4 main diversity claim cites n_distinct, not diversity_tanimoto. R20-P8 plans apples-to-apples retrain with R18 Multi-PH feature attached."

### §6.7 m15 RMSD (旧定义)

**§6 writing**:
> "The rigid_fragment_rmsd metric (1.241 ± 0.626 Å, per-pocket uniform across 5 pockets) measures self-RMSD between two ETKDGv3 + UFF-optimized conformers of the **same** molecule (pocket does not enter the calculation). Identical 1.241 across pockets is the **expected** behavior, not a staging artifact. To measure true docking quality, we introduce the docked_pose_rmsd metric (Phase D) — RMSD between Vina-docked pose and crystal ligand pose after Kabsch alignment, which actually reflects docking accuracy."

### §6.8 Hilbert 4-cell R² (Wf-83 original)

**§6 writing**:
> "Original Wf-83 baseline reports Hilbert+ASTSR combined R² = -6.71 on PlatinAI 4-cell across 40 seeds. Wf-122 Phase 1 (Hilbert L2 regularization) and Wf-127 Phase 4 (L2 + ASTSR scale guard) ship as defensive measures preventing divergence but do not lift R² from baseline. Architectural signal floor remains the binding constraint (see §6.1)."

## §3 §6 与表 1-5 lead claim 的对应

| §6 honest-negative | §4 lead claim 替代 |
|---|---|
| §6.1 PlatinAI 4-cell R² | **不引**到 §4 main; 8 lead claim 用 DrugOOD + NCI-60 + n_distinct 替代 |
| §6.2 L5 Hilbert R² | 不引到 §4 main; framework description 在 §3 |
| §6.3 L7 multi-PH + PLL R² | 不引到 §4 main; paper §3 framework 提到 |
| §6.4 L10 3-way R² | 不引到 §4 main; paper §3 framework 提到 |
| §6.5 m3 Vina cisplatin | **§4.5.6 protein-binder pool 替代** — 用 5-pocket protein-binder pool 数据 |
| §6.6 diversity_tanimoto | **§4.5.3 n_distinct 替代** — 引 n_distinct 1→20 |
| §6.7 m15 RMSD 旧定义 | **§4.5.7 docked_pose_rmsd 替代** — Phase D 新 metric |
| §6.8 Hilbert 4-cell R² | 不引到 §4 main; paper §3 framework |

## §4 §6 写作原则(per party-twenty-spirit §3.5)

**承认**:
1. 不假装 13 math theorems 全 lift 到 R² ≥ 0
2. 不假装 m1 Ru AUC 0.7123 是 paper lead(单 seed)
3. 不假装 cisplatin Vina 代表真 binding
4. 不假装 diversity_tanimoto vs TargetDiff 0.860 是 framework lead
5. 不假装 5-seed dmpnn 5-metal R² ≥ 0(架构性 ceiling)
6. 不假装 m15 RMSD 1.241 是 docking quality(self-RMSD)
7. 不假装 Hilbert+ASTSR 4-cell 是 universal 提升
8. 不假装 NCI-60 9-cell 是真正 wet-lab(纯 in-silico)

**不写**:
1. 不写 "we achieved" 当没真 achieve
2. 不写 "state-of-the-art" 当 metric 是 honest-negative
3. 不写 "improved over baseline" 当 lift 是 artifact
4. 不写 "novel method" 当方法已被文献充分覆盖

**paper §6 必须含**:
- 8 个 honest-negative 显式编号 + 数值 + reason
- 架构性 ceiling 解释(12-d RDKit signal floor)
- Vina 物理模型局限解释(cisplatin covalent)
- Wet-lab 验证缺口(0 IC50 / 0 LOI / 0 MoU)
- Future work(R20-P6 wet-lab + R22 tier-1 metric + R23 Phase I clinical)

## §5 Cross-reference

- **TODO-131** weak metric lift master plan §4 honest-negative list
- **TODO-132** paper §4 lead rewrite §6 honest-negative ship
- **party-twenty-spirit §3.5** — 0-tolerance honest-negative framing
- **Wf-127** Hilbert L2 + ASTSR guard ship
- **Wf-128** 3-way real pIC50 ship
- **Wf-129** L7 multi-PH + PLL ship
- **Wf-121** cisplatin real Vina ship
- **Wf-126** m15 RMSD honest finding
- **Wf-83** original Hilbert 4-cell R² baseline
- **Wf-12** R12 diversity_tanimoto baseline

## §6 Final verdict

**paper §6 honest-negative ship**:✅ **ready**
- 8 个 honest-negative 显式编号 + 数值 + reason
- 架构性 ceiling 解释
- Vina 物理局限解释
- Wet-lab 验证缺口 + Future work

**ETA**:1 天 ship paper §6 v2

**paper §4 + §6 一起 ship**:**3-4 天**(paper §4 1-2 天 + paper §6 1 天 + paper §3 method 1 天 + paper §7 conclusion 0.5 天)

**Target journal**:**JCIM (IF 5.6)** — Q2 paper 立即可投