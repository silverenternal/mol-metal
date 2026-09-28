> **STATUS: SUPERSEDED 2026-09-26 by R17 close-out (commit 4561625) + Track-B Lambda + PLL ship.**
>
> Track B 5 sub-workflows 全部 SHIP:
> - B0 PLL engine (Liu & Feng 2026, DOI:10.1021/acs.jcim.6c02460) ✅ (`TODO/completed/32_trackB_mlc_pll_wireup_done.md`)
> - B1 atoms-table 30-metal extension ✅ (`TODO/completed/32_trackB_uc_b1_metal_atoms_30.md`)
> - B2 β-reduction reducer (PLL 谱 cost) ✅ (`TODO/completed/32_trackB_mlc_pll_wireup_done.md`)
> - B3 click chemistry rule library (5+ rules) ✅ (R17 ship, see 18-rule β-reductions in `TODO/completed/41_r17_final_closeout.md` §R17 final state)
> - B4 PLL→LamMCTS heuristic + FPU reduction ✅ (`TODO/completed/32_late_phase.md`)
> - B5 end-to-end pipeline smoke (100 候选分子 + λ 推导) ✅ (`TODO/completed/32_late_phase.md`)
>
> 当前: `TODO/completed/41_r17_final_closeout.md` (R17 close-out) + `TODO/completed/32_trackB_mlc_pll_wireup_done.md` + `TODO/completed/32_late_phase.md` + `TODO/completed/32_trackB_uc_b1_metal_atoms_30.md`.
> M-R17 GPU 路径 仍 BLOCKED on dGPU cold-power-cycle gate.
> 此文件保留作为 historical record, 不再决策。
>
> 归档: `TODO/04_history/32_ultracode_trackB_lambda_calculus_superseded_2026-09-26.md`

# Ultracode Track B: Lambda 算法轨 (Molecular Lambda Calculus + PLL 2026)

> **LEGACY** — pre-R17; see `TODO/INDEX.md` and `TODO/EXECUTION_ORDER.md` for current planning.
> Superseded by: `TODO/completed/32_trackB_mlc_pll_wireup_done.md` + `TODO/completed/32_late_phase.md` + `TODO/completed/32_trackB_uc_b1_metal_atoms_30.md` (R17 close-out).

> 启动日期: 2026-09-25
> 状态: 待派发
> 派发目标: ultracode session B
> 关联目录: `TODO/13_lambda_clickchem/`
> **核心数学参考 (2026 国内数学新工作)**: Persistent Local Laplacian (PLL) by Jian Liu, 重庆理工大学 数理科学研究中心

## 目标 (1 周冲刺)

按 `TODO/13_lambda_clickchem/plan.md` 建立 Molecular Lambda Calculus (MLC) 的 5 层基础,并把 **PLL 2026 作为类型指纹 + β-NF 判定器** 集成进搜索算法。

## Week-1 任务清单

### B0 — PLL 2026 数学引擎落地 (Day 1-2,优先级最高)
- [ ] 读 Liu & Feng 2026 (*J. Chem. Inf. Model.*, DOI: 10.1021/acs.jcim.6c02460)
- [ ] 实现 `molmetal/molmetal_lam/lam_chem/pll_signature.py`: 局部拉普拉斯矩阵 + 谱分解
  - 输入: RDKit mol / 点云
  - 输出: PLL eigenvalue sequence (top-k)
- [ ] Unit test: 在 5 个 PDBbind 分子上验证 PLL 谱稳定性 (旋转/平移不变)
- [ ] Benchmark: PLL 计算 vs 分子原子数 — 确认切除定理带来的加速比

### B1 — 原子-as-组合子 (Day 2-3)
- [ ] `molmetal/molmetal_lam/atoms/combinators.py`: `Atom` dataclass + `PRIMITIVE_ATOMS` 表
- [ ] 验证 cisplatin = `Pt_II NH3 NH3 Cl Cl` 在 β-NF
- [ ] Unit test: 30 个有机金属分子的 arity 计算与 RDKit valence 一致

### B2 — 键-as-Application + β-reduction (Day 3-4)
- [ ] `molmetal/molmetal_lam/lam_chem/ast.py`: `LamVar` / `LamAbs` / `LamApp` 三类节点
- [ ] `molmetal/molmetal_lam/lam_chem/reducer.py`: β-reduction 单步 + β-NF 判定
- [ ] **关键**: 用 B0 的 PLL 谱距离作为 reduction cost — β-reduction 一步的 PLL 谱变化量

### B3 — Click chemistry Lambda 规则库 (Day 4-5)
- [ ] `molmetal/molmetal_lam/lam_chem/click_rules.py`: CuAAC / SPAAC / Diels-Alder 作为 rewrite rules
  - 例如: `CuAAC(azide, alkyne) → triazole` 形式化为 `(λx.λy.triazole(x,y)) azide alkyne`
- [ ] 复用 RDKit reaction SMARTS (`[CX2:1]#[CX2:2].[N:3]=[N+:4]=[N-:5]>>[CX2:1]1[C:2][N:3]=[N+:4]=[N-5]1`)
- [ ] Unit test: 10 个已知 click 反应的 SMILES 输出验证

### B4 — MCTS + PLL 启发式 (Day 5-6)
- [ ] `molmetal/molmetal_lam/search_alg/mcts.py`: `LamMCTS` class
- [ ] **关键创新**: 启发式函数 = `1 / (1 + ||PLL(state) - PLL(target)||_2)`
- [ ] 在 CrossDocked pocket-1 上跑 1000 iterations,记录 best score / 路径多样性 / 合成可行性

### B5 — Pipeline 串联 + smoke test (Day 6-7)
- [ ] `molmetal/molmetal_lam/pipeline/closed_loop.py`: 端到端 click chem pipeline
- [ ] 输出 100 候选分子,每个附 Lambda 推导表达式
- [ ] 与 Track A 对接: 把 EGNN IC50 预测作为 reward signal 喂给 MCTS

## 完成判据 (Definition of Done)

1. **PLL 引擎**单测全绿 + 在 PDBbind 上与论文报告数对得上
2. **β-reduction** 在 100 个随机分子上都能终止 (cofinite 校验)
3. **CuAAC 规则** 对 10/10 已知 click 反应正确输出 SMILES
4. **MCTS + PLL 启发式** 在 CrossDocked pocket-1 上 best score > 0.7
5. 在 `TODO/completed/` 下新建 `32_trackB_mlc_pll_wireup_done.md` 总结

## 与 Track A 同步点

- **周三 (Day 3) 下午**: Track B 暴露 `PLLSignature(mol) → ndarray` 接口给 Track A 作为可选分子描述符
- **周五 (Day 5) 下午**: Track B 接收 Track A 的 EGNN IC50 预测作为 MCTS reward signal (取代 placeholder 随机 reward)

## 数学参考 (2026 国内数学新工作)

**主参考**: 刘健 (Jian Liu, 重庆理工大学 数理科学研究中心) & 冯洪松 (Hongsong Feng, UNC Charlotte).
**论文**: *Persistent Local Laplacian Prediction of Protein–Ligand Binding Affinities*.
**期刊**: *J. Chem. Inf. Model.*, 2026.
**DOI**: 10.1021/acs.jcim.6c02460
**arXiv**: 2603.21503 (2026-03-23)
**资助**: NSFC 12401080.

**为什么是 PLL 而非 PySR/Cranmer 2023**:
- PLL 是**真正的中国数学工作** (重庆理工 + NSFC 资助),符合"2026 国内数学新工作"的定位
- PLL 谱的局部化 + 稳定性天然契合 λ-演算的"局部归约 + α-conversion 不变"
- PLL + Transformer 可作 β-NF oracle,这是 PySR 完全不能胜任的

## 风险与回退

- 若 PLL 实现性能不够,先调 `scipy.sparse.linalg.eigsh` 取 top-k 特征值
- 若 β-reduction 不终止,引入 size limit (e.g. 分子量 < 500 Da) 作为外部 termination
- 若 Track A reward 接口未就绪,先用 docking score (QuickVina) 占位

## 资源

- GPU: 不强需,PLL 可在 CPU 上跑;MCTS + EGNN 需要 GPU
- 数据: PDBbind v2007/v2013/v2016 (PLL benchmark); CrossDocked pocket-1 (MCTS smoke)
- 文档: `TODO/13_lambda_clickchem/{plan,molecular_lambda_calculus}.md`, `references/SynFlowNet/`
