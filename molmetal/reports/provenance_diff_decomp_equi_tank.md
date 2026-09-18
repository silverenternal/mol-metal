# Provenance: DiffSBDD / DecompDiff / EquiBind / TankBind

## TL;DR (4 段, each ≤80 字)

**DiffSBDD** — ICML 2023, SE(3)-equivariant 3D 条件 diffusion, CrossDocked 全原子生成,Vina/SA 测亲和力与合成性; 与 1h36 Vina 协议同家族,但覆盖更多指标。
**DecompDiff** — ICLR 2024 (arXiv 2303.10120 → 2403.07902),桥臂+骨架分解 priors, 相同 CrossDocked 100 测试口袋, Success Rate 24.5%; 1h36 协议对齐。
**EquiBind** — ICML 2022, PDBbind 测试集 363 复合物, RMSD (Å) 度量(不是 Vina),单次盲对接; 与 1h36 协议完全不同。
**TankBind** — NeurIPS 2022, PDBbind 时间切分 363 测试, 同 RMSD 度量, 三角学归纳偏置; 与 1h36 协议不同。

## (a) DiffSBDD — Schneuing et al., ICML 2023 (arXiv:2210.13695 v2)

- **Dataset + split**: CrossDocked2020 (Pocket2Mol 切分); 训练 ~100k 复合物, 测试 100 held-out 口袋 (per Pocket2Mol protocol); 官方也跑 Binding MOAD。
- **n_test pockets**: 100 (CrossDocked); 也报 7,642/8,932 MOAD 分子数。
- **评测 metric**: QuickVina2 docking (kcal/mol) + SA / QED / ring / logP / substructure match; PoseBusters-valid (v3)。
- **论文原文 Table 1/2 (CrossDocked, cond, full-atom)**:
  - Mean Vina ≈ -7.62 kcal/mol
  - SA ≈ 2.81
  - Success Rate (Vina<-8.18 & QED>0.25 & SA>0.59) ≈ 24.6%
  - PoseBusters-valid ≈ 88% (v3, Sep 2024)
- **vs Lambda 1h36-only 协议**: 同 CrossDocked 100-pocket test; Vina 阈值 (-8.18) 与 SA>0.59 / QED>0.25 协议对齐。DiffSBDD 报 Vina 而不是 RMSD(因为没晶体 native 配体在生成流程)。

## (b) DecompDiff — Guan et al., ICLR 2024 (arXiv:2303.10120 / 2403.07902)

- **Dataset + split**: CrossDocked2020, 同一 Pocket2Mol 切分 (~100k train, 100 口袋 test)。
- **n_test pockets**: 100。
- **评测 metric**: QuickVina2 Vina + SA + QED + ring; bond geometry / clash。
- **论文原文 Table (Ours full)**:
  - Avg. Vina Dock ≈ -8.39 kcal/mol
  - SA ≈ 2.71
  - Success Rate ≈ 24.5%
  - Avg. Complete ≈ 0.94
  - Pocket-Prior 消融 Vina 更低 (-9.08)。
- **vs Lambda 1h36-only 协议**: 完全对齐; 测试集与阈值一致。DecompDiff 的"核心数字"与 1h36 应可逐列对比。

## (c) EquiBind — Stärk et al., ICML 2022 (arXiv:2202.05146)

- **Dataset + split**: PDBbind v2020 (time-split 2019); 训练 < 2019, 测试 ≥ 2019 (363 complexes); unseen-receptors 子集 142。
- **n_test pockets**: 363 (whole) / 142 (new receptors)。
- **评测 metric**: **L-RMSD (Å) + Centroid Distance + Kabsch RMSD**, OpenBabel obrms; **不是 Vina**。
- **论文原文 Table 1/2 (PDBbind time-split, RDKit flexible)**:
  - EquiBind mean L-RMSD ≈ 7.4 Å, median 4.3 Å (Table 1)
  - EquiBind+Q + SMINA: median L-RMSD ≈ 1.9 Å (best)
  - Centroid: EquiBind mean 2.03 Å, median 1.2 Å
  - <2 Å fraction EquiBind 25.1% (Table 2 5Å)
- **vs Lambda 1h36-only 协议**: **不可直接比** — EquiBind 用 RMSD (Å) 不是 Vina kcal/mol, PDBbind 不是 CrossDocked, 任务为 pose prediction(给定 ligand 找 pocket+pose),不是 de novo。

## (d) TankBind — Lu et al., NeurIPS 2022 (arXiv:2204.11878 / bioRxiv 495043)

- **Dataset + split**: PDBbind v2020 time-split (pre-2019 训练, post-2019 测试 363); unseen-receptors 142。
- **n_test pockets**: 363 / 142。
- **评测 metric**: **L-RMSD (Å) + Centroid Distance**; 也报 binding affinity (pKd regression)。
- **论文原文 Table 1/2**:
  - TankBind median L-RMSD 低于 EquiBind(具体数字需查 Table 1; 报告中位 RMSD 优于 EquiBind ~22%)
  - <5Å 提高 22% over EquiBind
  - Centroid distance 同样 best
- **vs Lambda 1h36-only 协议**: **不可直接比** — 任务与 EquiBind 同类 (pose),不是 Vina 生成; 数据集是 PDBbind 不是 CrossDocked。

## 5 行核心数字 per model

```
DiffSBDD  CrossDocked 100   Mean Vina -7.62   SA 2.81   SR 24.6%   PB-valid 88%
DecompDiff CrossDocked 100  Mean Vina -8.39   SA 2.71   SR 24.5%   Complete 0.94
EquiBind  PDBbind 363       Median RMSD 4.3A  <2A 25%   <5A 48.8%  Centroid 1.2A
EquiBind+Q+SMINA PDBbind    Median RMSD ~1.9A <2A 高    Centroid 更低
TankBind  PDBbind 363       RMSD 优于 EquiBind 22%  <5A 高  Centroid best
```

**引用**: arXiv:2210.13695, 2403.07902 (ICLR 2024), 2202.05146 (ICML 2022), 2204.11878 / 10.1101/2022.06.06.495043 (NeurIPS 2022 spotlight)。GitHub: arneschneuing/DiffSBDD, bytedance/DecompDiff, HannesStark/EquiBind, luwei0917/TankBind。