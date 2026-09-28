> **STATUS: SUPERSEDED 2026-09-26 by R17 close-out (commit 4561625) + Track-A Flow Matching wire-up.**
>
> Track A 4 sub-workflows (UC-A1..A4) 全部 SHIP:
> - A1: flow_matching + FLOWR clone + Lipman adapter 落点确认 ✅
> - A2: CFM loss + OT path scheduler wire-up ✅
> - A3: EGNN velocity field wrapper + SE(3) 等变性 test ✅
> - A4: MoleculeGenerator Protocol 实现 + CrossDocked pocket-1 smoke test ✅
>
> 当前: `TODO/completed/37_model_track_mr17.md` (R17 M-track synthesis) + `molmetal/reports/wf_r17_gpu_cli/final.md` + `molmetal/reports/wf_paper_sweep_spec/final.md` + `molmetal/reports/wf_24paper_refs/final.md`.
> paper 76 → 79 页;bib 90 → 114;0 unresolved refs.
> 此文件保留作为 historical record, 不再决策。
>
> 归档: `TODO/04_history/31_ultracode_trackA_flow_matching_superseded_2026-09-26.md`

# Ultracode Track A: 模型轨 (Flow Matching + EGNN)

> **LEGACY** — pre-R17; see `TODO/INDEX.md` and `TODO/EXECUTION_ORDER.md` for current planning.
> Superseded by: `TODO/completed/37_model_track_mr17.md` (R17 close-out).

> 启动日期: 2026-09-25
> 状态: 待派发
> 派发目标: ultracode session A
> 关联目录: `TODO/12_flow_matching/`

## 目标 (1 周冲刺)

按 `TODO/12_flow_matching/plan.md` 把 Lipman 2023 Flow Matching 与 Mol-Metal 的 EGNN 完成 wire-up,产出可在 CrossDocked 上跑通 1-pocket smoke test 的 end-to-end pipeline。

## Week-1 任务清单

### A1 — Clone 母论文代码 (Day 1)
- [ ] `git clone https://github.com/facebookresearch/flow_matching.git molmetal/references/flow_matching`
- [ ] `git clone https://github.com/jule-c/flowr.git molmetal/references/FLOWR` (Cremer 2025/2026 强基线)
- [ ] 核对 Lipman adapter 落点: `molmetal/adapters/flow_matching_lipman/`

### A2 — CFM loss wire-up (Day 2-3)
- [ ] 包装 `flow_matching.loss.ConditionalFlowMatchingLoss` 为我们的 `MolCFMConfig`
- [ ] OT path scheduler: `ConditionalOTScheduler` → `molmetal/adapters/flow_matching_lipman/path_schedulers.py`
- [ ] Unit test: 在 CrossDocked pocket-1 上 loss 收敛 100 step,打印 KL/OT distance

### A3 — EGNN velocity field (Day 4-5)
- [ ] 把现有 EGNN (`molmetal/models/egnn.py`) 包装为 velocity field function
- [ ] 输入: `(coords, atom_features, edge_index)`;输出: `(v_coords, v_features)`
- [ ] SE(3) 等变性 unit test: 随机旋转/平移 100 次,验证输出变换一致

### A4 — Pocket-conditioned generation smoke test (Day 6-7)
- [ ] `MoleculeGenerator` Protocol 实现: `molmetal/ports/generator.py` → `FlowMatchingAdapter`
- [ ] 在 CrossDocked pocket-1 上生成 100 分子,对接 QuickVina,记录 best IC50 / SA / QED
- [ ] 与现有 `LipmanFM` (completed todo) 对比: 产率 / 3D RMSD / SE(3) 不变性

## 完成判据 (Definition of Done)

1. `python -m molmetal.pipelines.fm_smoke --pocket 1 --n-samples 100` 全程跑通,耗时 < 30 min
2. 输出 `metrics/fm_smoke_pocket1.json`: best IC50 / SA / QED / RMSD / 端到端时长
3. Wire-up 通过 3 个 unit test: `tests/test_fm_lipman_adapter.py`, `tests/test_egnn_velocity_se3.py`, `tests/test_molecule_generator.py`
4. 在 `TODO/completed/` 下新建 `31_trackA_fm_egnn_wireup_done.md` 总结

## 风险与回退

- 若 EGNN 与 FM 维度不匹配,回退到 `references/flow_matching/examples/2d_pointcloud` 的简单 MLP velocity field 做最小可跑通版本
- 若 QuickVina 安装问题,回退到 `molmetal/dockers/qvina_cpu` docker 镜像

## 与 Track B 同步点

- 周三 (Day 3) 下午对齐: Track A 暴露 `EGNNVelocityField.forward(coords, h) → (v_x, v_h)` 接口给 Track B 作为"reward signal"输入
- 周五 (Day 5) 下午对齐: Track B 把 PLL-based β-NF oracle 接入 Track A 的 generation pipeline 作为可选 posterior

## 资源

- GPU: 1×H100 或 A100 (40GB),如无可用 CPU smoke test
- 数据: CrossDocked pocket-1 (已有)
- 文档: `TODO/12_flow_matching/plan.md`, `references/flow_matching/flow_matching/`
