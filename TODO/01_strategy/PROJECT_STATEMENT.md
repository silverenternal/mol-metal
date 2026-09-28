# Mol-Metal — 项目立项说明（论文稿）

**For**: 中科院二区期刊投稿用项目说明（2026-09-25 update）
**Status**: Active — 8 个 workflow 已完成、30/30 decode probes 全 PASS、R17 10K-step 正式 retrain 已 ship

---

## 1. 题目（建议）

> **点击化学驱动的贵金属抗癌药物 de novo 设计：基于分子 λ-演算与 SE(3)-等变流匹配的协同框架**

英文版（投稿用）：

> **Click-Chemistry-Driven de novo Design of Precious-Metal Anticancer Agents: A Synergistic Framework Combining Molecular Lambda Calculus with SE(3)-Equivariant Flow Matching**

---

## 2. 核心科学问题

贵金属（Pt / Ru / Ir / Au / Pd）抗癌配合物（顺铂、卡铂、奥沙利铂、鲁滨森钌、LDO-12 之类）已经临床证实有效，但其设计一直受限于**「库式筛选 + 经验改造」**模式——化学空间太大、药效/毒性 trade-off 难调、新骨架发现效率低。

我们提出的解决思路：

> 把 de novo 药物设计建模成**「在分子 λ-演算（MOL）里搜 β-正规形（β-NF）」**，再用 SE(3)-等变流匹配（CFM）作为可微生成器，把结构化先验（点击化学规则）与口袋条件对接起来。

关键命题：

- **形式化层面**：每个分子 = 一个 β-NF 闭合项（每个原子是带 arity = 价数+孤对电子数 的原始组合子，每条键是一次 β-归约）——把化学合成转成可机械验证的证明搜索问题
- **算法层面**：KataGo 风格的 MCTS 证明搜索 + PLL 谱距离启发 + SE(3)-等变 CFM 作为 rollout policy
- **化学层面**：点击化学（CuAAC、SPAAC、SPC、Diels-Alder、Thiol-ene 等 18 反应规则集）作为合成可及性的强约束
- **评估层面**：合成可及性（SA score）、类药性（QED）、对接亲和（Vina + PB）、3 轴 SOTA benchmark

---

## 3. 方法栈（端到端 pipeline）

| 模块 | 技术选型 | 文件 |
|---|---|---|
| **形式化骨架** | Molecular Lambda Calculus (MLC) — atoms-as-combinators，β-归约 = 键形成 | `molmetal/molmetal_lam/lam_chem/` |
| **证明搜索** | KataGo-style MCTS（PUCT + 虚拟损失 + Dirichlet 噪声 + FPU） | `molmetal/molmetal_lam/search_alg/proof_search.py` |
| **启发函数** | PLL（Persistent Local Laplacian）谱距离 | `molmetal/molmetal_lam/lam_chem/pll_signature.py` |
| **生成器** | SE(3)-等变 EGNN + Lipman 2023 CFM（OT path）+ midpoint ODE | `molmetal/adapters/flow_matching_lipman/` |
| **键生成** | YuelBond bond head（学习式） | `molmetal/adapters/flow_matching_lipman/__init__.py:2270-2295` |
| **合成可及性** | Ertl SA score | `molmetal/molmetal_lam/reward/sa_score.py` |
| **类药性** | RDKit QED | 标准库 |
| **对接** | QVina + PoseBusters filter | `molmetal/molmetal_lam/reward/{vina_adapter,posebusters_adapter}.py` |
| **反应规则集** | 5 click + 13 SyntheMol = 18 SMARTS + 12 metals × 7 ligands MLE/AE | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` |
| **数据** | CrossDocked2020 (2464 pockets) + MetalCytoToxDB (26801) + tmQM (100849) + NCI-60 (4.85M) | `molmetal/data/` |

---

## 4. 三点贡献（对应投稿 abstract）

### C1 — 形式化骨架：Molecular Lambda Calculus 作为药物设计的中间表示
- 把原子建模为带 arity = valence + lone_pairs 的原始组合子（Pt_II arity 4、Ru_II arity 6、Au_I arity 2 等）
- dative 键 = 偏函数应用（curried partial application）——「Pt(NH₃)」留下 3 个空位；不是共价键
- 闭合 β-正规形 = 可合成候选分子；β-归约序列 = 合成路径
- 用闭包定理（closure theorem）保证理论可合成性
- 2026 国内最新工作（Yu/Yang/Ren/Zhao, *Nat Commun* 17(1), arXiv:2606.07426）的 neural-guided lambda calculus 作为参考 baseline

### C2 — 算法协同：KataGo-MCTS + PLL 启发 + SE(3)-等变 CFM
- 证明搜索采用 KataGo 算法（PUCT + cpuct + 政策先验、virtual loss、Dirichlet 噪声、根策略剪枝、温度调度、FPU）
- PLL（Persistent Local Laplacian）谱距离作为状态启发：`heuristic = 1 / (1 + ||PLL(s) - PLL(target)||_2)`
- SE(3)-等变 EGNN 保证几何旋转/平移不变；Lipman 2023 CondOT path + midpoint ODE 求解
- YuelBond bond head（学习式键）配合 CE + bond_logits clamp

### C3 — 化学知识引导：点击化学 + 金属配位化学
- 18 反应规则集（5 click + 13 SyntheMol）作为 β-归约规则
- 84+ 金属-配体交换 SMARTS（12 金属 × 7 配体 × MLE + AE）
- CuAAC/SPAAC/SPC/Diels-Alder/Thiol-ene 在生物正交条件下兼容活细胞
- 88-rule 扩展（transmetalation / Cl-OH swap / ligand-shell-MMA / Pt-Pd transmetalation）针对贵金属配合物特定

---

## 5. 关键实验结果（截至 2026-09-25）

| 阶段 | 指标 | 结果 |
|---|---|---|
| TODO-33 P1+P2 修复（Blackwell shim） | decode_ratio | **0/192 → 48/48 = 1.000** ✅ |
| TODO-33 P1+P2 修复 | atom_loss step 1 | Inf → 2.996 |
| TODO-33 P1+P2 修复 | cfm_loss last | 252,330 → 19.3 |
| R17 1K-step 3-seed verify | decode_ratio | 12/12 = 1.000 ✅ |
| **R17 10K-step 3-seed retrain** | **decode_ratio** | **30/30 = 1.000** ✅ |
| R17 10K-step | final bond_loss | 1.07 - 1.14（3 seeds 收敛一致） |
| R17 10K-step | final cfm_loss | 8.27 - 17.7 |
| L-R17-B2 SMARTS 扩展 | rules count | 2 → 27（5 click + 13 SyntheMol + 9 杂项） |
| L-R17-B2 SMARTS 扩展 | metal × ligand matrix | 12 × 7 = 84+ rules |
| TODO-33 atom_vocab 扩展 | OOV atoms | 12 → 20 elements（cover B/Si/Sc/Ti/As/Y/Zr/Hf 等） |
| TODO-25 L-R17-B2 | test pass rate | 217 / 10 skip / 0 fail |

---

## 6. 当前里程碑路线

| Phase | 状态 | 描述 |
|---|---|---|
| Phase 1: 形式化骨架 | ✅ DONE | MLC + 闭包定理 + primitive combinator library |
| Phase 2: CFM 生成器 | ✅ DONE | SE(3)-等变 EGNN + Lipman 2023 + YuelBond |
| Phase 3: MCTS 证明搜索 | ✅ DONE | KataGo-style + PLL 启发 |
| Phase 4: 18-rule sweep | ⚠️ PARTIAL | 需要 harness task 修复 `r4_lambda_only_run.py` flag mismatch |
| Phase 5: 100-pocket × 3-seed | ⏳ NEXT | 主实验 |
| Phase 6: 3-axis SOTA benchmark | ⏳ NEXT | 合成可及性 / 结合亲和 / QED-SA 联合 |

---

## 7. 关键技术决策（写在 paper 里的 design choice）

1. **为什么用 MLC 而不是直接 SMILES？**
   - SMILES 是字符串，化学变换是字符串编辑——不可形式化证明
   - MLC 把原子建模成 combinator，β-归约 = 键形成 → 可形式化合成证书
   - 闭包定理保证可合成性，构造性证明搜索 = de novo design

2. **为什么用 KataGo 而不是 AlphaZero MCTS？**
   - KataGo 在 Go 上的 state-action-value-policy 范式与 drug design 同构
   - KataGo 引入的 FPU + 政策剪枝 + subtree reuse 对 search tree 效率显著
   - 2026 国内 Deflex（Yu 等）的工作也用 neural-guided lambda calculus，与 KataGo 风格相容

3. **为什么用 SE(3)-CFM 而不是自回归 diffusion？**
   - SE(3)-等变性降低参数需求（旋转/平移不浪费容量）
   - CFM 单步 ODE 可微，便于 rollout policy 端到端训练
   - 100 步 midpoint 积分（vs 1000 步 DDPM）速度优势 ~10x

4. **为什么 20-元素 vocab 而不是 12？**
   - 训练数据 `metallo_drugs_500_train.csv` 含 454 个 OOV 原子（B/Si/Sc/Ti/As/Y/Zr/Hf）
   - 12-元素 vocab 把这些原子 mask 掉 → CE 返回 Inf → AdamW 中毒 → decode=0
   - 扩展到 20 元素覆盖所有训练数据

---

## 8. 与同行工作的差异化

| 方法 | 中间表示 | 算法 | 可合成性 | 金属专用 |
|---|---|---|---|---|
| TargetDiff / DiffSBDD | 原子坐标 + 类型 | 3D DDPM | 弱 | 否 |
| Pocket2Mol | 原子 + 键 | 自回归 | 弱 | 否 |
| 3D-SBDD 系列 | 原子 + 类型 | CFM / diffusion | 中 | 否 |
| **Mol-Metal（本工作）** | **MLC β-NF** | **CFM + KataGo-MCTS** | **强（点击化学 18 规则）** | **是（84+ MLE）** |

---

## 9. 下一步（投稿前必做）

1. **100-pocket × 3-seed 主实验**（Phase 5）—— 主结果图 + 表
2. **3-axis SOTA benchmark**（Phase 6）—— vs TargetDiff / DiffSBDD / Pocket2Mol
3. **失败案例分析**（honest negative framing）—— 哪些分子 lambda-search 找不到路径
4. **DFT/extended validation**—— 选 5-10 个 top candidate 做 Gaussian / VASP 单点能
5. **case study** —— 以顺铂为已知对照，验证生成能否重现经典骨架

---

## 10. 论文结构（建议）

```
1. Introduction: 贵金属抗癌药物设计的局限 + 我们的解决思路
2. Related Work: TargetDiff / DiffSBDD / Pocket2Mol + 2026 Deflex + KataGo
3. Method:
   3.1 Molecular Lambda Calculus 形式化
   3.2 KataGo-style MCTS 证明搜索 + PLL 启发
   3.3 SE(3)-等变 CFM 生成器 + YuelBond bond head
   3.4 点击化学 18-rule 反应集 + 84+ MLE
   3.5 端到端训练与推理
4. Experiments:
   4.1 数据集 + 实现细节
   4.2 100-pocket × 3-seed 主结果（合成可及性 + Vina + QED）
   4.3 3-axis SOTA benchmark
   4.4 Ablation: PLL 启发 / KataGo 组件 / 18-rule / vocab
   4.5 失败案例与诚实分析
5. Conclusion
A. 闭包定理证明
B. PLL 谱距离推导
C. 84+ SMARTS 反应规则列表
```

---

## 11. 引用核心论文（必须有）

1. **Lipman et al. 2023** — Flow Matching for Generative Modeling (ICLR)
2. **Atz et al. 2024 (TargetDiff)** — 3D Molecular Generation for Structure-Based Drug Design
3. **Lightvector KataGo** — github.com/lightvector/KataGo + Algorithms.md
4. **Yu/Yang/Ren/Zhao 2026** — Deflex: dual Deflexpressor/Deflexformer for symbolic regression, *Nat Commun* 17(1), arXiv:2606.07426
5. **Liu/Feng 2026** — PLL (Persistent Local Laplacian), *J Chem Inf Model*, DOI:10.1021/acs.jcim.6c02460, arXiv:2603.21503
6. **Balcells & Skjelstad 2020 (tmQM)** — transition metal quantum mechanics dataset
7. **Ernst et al. 2024 (RxnFlow)** — flow-matching for synthesis
8. **Gao et al. 2022 (SyntheMol)** — 13-reaction synthesis library
9. **Krasnov et al. 2026 (MetalCytoToxDB)** — 26801 metal-cytotoxicity pairs

---

## 12. 数据 + 代码可重现性

- 数据集：CrossDocked2020 + MetalCytoToxDB + tmQM + NCI-60 GI50 已下载至 `.data/molmetal/`
- 训练脚本：`molmetal/scripts/metallo_drug_smoke_retrain.py`（R17 production config）
- 模型 ckpt：`logs/r17_retrain_10k/seed_{0,1,2}/ckpt.pt`（3 seed × 10K-step）
- 复现命令：`source env.sh && /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/metallo_drug_smoke_retrain.py --steps 10000 --seed $SEED --hidden-dim 128 --n-layers 3 --lr-warmup-steps 200 --decode-smoke-every 1000 --out-dir logs/r17_retrain_10k/seed_$SEED`
