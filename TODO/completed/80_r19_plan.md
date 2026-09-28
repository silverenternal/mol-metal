# TODO 80 — R19 PLAN 战略规划 (Wf-27 synthesis)

**Date**: 2026-09-26 (Wf-27 ship)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIP (R19_PLAN.md 28 sections, 1187 lines, 66.5 KB)
**Parent**: `TODO/03_plans/R18_PLAN.md` §15 close-out
**Sibling inputs**:
- `TODO/03_plans/R18_PLAN.md` §15 (R18 close-out, 23 mechanism ship)
- `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` (Blackwell 95.6 GB / RTX 5090 31.8 GB confirmed available)
- `TODO/INDEX.md` §11 (24 wf ledger)

---

## §1 TL;DR

R19 PLAN 已 ship 到 `TODO/03_plans/R19_PLAN.md` (1187 lines, 28 sections):

| 段 | 内容 | 来源 |
|---|---|---|
| §1-5 | R19 总体 vision + vs R18 区别 + 边界 + continuity + R20+ 关系 | Wf-27 Phase 1 |
| §6-12 | Hardware (Blackwell + RTX 5090) + Software + 6 phases × 12 weeks GPU 训练 + Multi-GPU + wall-time | Wf-27 Phase 2 |
| §13-18 | 投稿路线 (Digital Discovery / JCIM) + paper v2 timeline + Cover letter + Reviewer rebuttal | Wf-27 Phase 3 |
| §19-23 | Wet-lab 合作 (ORNL / PKU) + 3 个 milestones (m18 / cisplatin IC50 / lead compound) | Wf-27 Phase 4 |
| §24-27 | Skill ecosystem 8 → 12 (新增 4: mol-metal-blackwell-training / paper-v2 / wet-lab / r19-math) | Wf-27 Phase 5 |
| §28 | R19 PLAN 总结 + 关键决策 | Wf-27 Phase 6 (synthesis) |

---

## §2 R19 vs R18 区别

| 维度 | R18 (已 ship) | R19 (规划) |
|---|---|---|
| **状态** | ✅ 21 wf SHIP, 32 R18 真活 modules + 1235 tests + 23 mechanism | 🔄 R19 = 实际 GPU retrain + paper v2 + wet-lab |
| **计算资源** | CPU-only (R18 ship modules 都是 skeleton) | **GPU Blackwell + RTX 5090 实际可用** (Wf-22 confirmed 116.5 GB free) |
| **结果类型** | MEASURED 全部 = CPU pass / PROJECTED GPU lift | R19 = 实际 GPU retrain MEASURED |
| **文档** | paper §3.4-3.9 + 23 mechanism mapping + 21 wf reports | R19 = paper v2 (160+ pages) + R19_PLAN |
| **合作** | 0 wet-lab outreach | R19 = ORNL / PKU wet-lab 合作 |

---

## §3 R19 6 phases GPU 训练 schedule (摘要)

| Phase | Week | 内容 | 期望 metric lift |
|---|---|---|---|
| 1 | 1-2 | M22 SE(3) SB retrain (GPU 1 避开 kanzi) | m3 Vina Dock -8.5 → -10.5 |
| 2 | 3-4 | M23 Transport Relax + M24 Multi-PH retrain | m14 Decode ratio ≥ 1.000, m7 Diversity ≥ 0.75 |
| 3 | 5-6 | Hilbert-ASTSR + Joint Multi-PH | ASTSR R² ≥ 0.62, -50% wall-time |
| 4 | 7-8 | M25-M30 mechanisms GPU retrain | 综合 R18 lift |
| 5 | 9 | Phase A re-attempt (multi-GPU + scaffold split) | m1 Ru AUC 0.712 → 0.85+ (retest) |
| 6 | 10-12 | Pitfall P2.5 Phases C-E (MD relaxation + AMBER) | MD relaxed coords |

---

## §4 R19 投稿路线

| 目标 | 期刊 | IF | 优势 | 风险 |
|---|---|---|---|---|
| 首选 | **Digital Discovery (RSC)** | Q1 IF 8.5 | Triton + lipman + ML × chemistry 故事 | Round-13 paper grade 仍需补 GPU 数据 |
| 次选 | **J. Chem. Inf. Model. (ACS)** | Q1 IF 5.6 | methods + pharma audience | method-heavy, 需 proofs |
| 备选 | Patterns (Cell) | Q1 IF 6.0 | broader scope | 故事需要 MCTS 主导 |

---

## §5 R19 wet-lab 合作 3 milestones

| Milestone | 内容 | 期望数据 |
|---|---|---|
| **M1** | 单个 m18 protein-metal binding 实验验证 (ITC) | Kd 数量级匹配 R18 PROJECTED |
| **M2** | cisplatin 衍生物 cytotoxicity IC50 测定 (NCI-60 9-cell) | IC50 ≤ 5 µM (Phase A re-test) |
| **M3** | lead compound 体外活性验证 (cell uptake via ICP-MS) | 验证 in-cell Pt/Ru 浓度 |

---

## §6 Cross-reference

- `TODO/03_plans/R19_PLAN.md` (1187 lines, 28 sections)
- `TODO/03_plans/R18_PLAN.md` §15 close-out
- `TODO/INDEX.md` §11 24 wf ledger + §12 deferred blocks
- `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json` (Blackwell + RTX 5090 可用)
- `TODO/completed/75_blackwell_retrain.md` (Wf-22 honest findings)
- `TODO/completed/77_todo_cleanup.md` (Wf-24 TODO cleanup)

---

## §7 honest framing

R19 PLAN **不是承诺**所有 6 phases 100% ship。
- Phase 1-2 (M22/M23/M24 retrain): 高概率成功 (GPU 实际可用)
- Phase 3-4 (Hilbert + mechanisms): 中概率,依赖 5-seed × 50-epoch wall-time 实际值
- Phase 5 (Phase A re-attempt): 低概率, R17 close-out 时 5-seed × 100-epoch 跑了 30 min budget 被 block; R19 应给 ≥ 4 hours budget
- Phase 6 (MD relaxation + AMBER): 高概率,但需要 AMBER license 验证
- Wet-lab milestones: 100% 依赖 ORNL/PKU 等 facility 实际 reply outreach (Wf-28 已准备 data + template)

---

## §8 最终 verdict

**R19 PLAN ship status**: 28 sections / 1187 lines / 66.5 KB all on disk
**Workflow return status**: R19_PLAN.md 实质 ship,但 workflow 框架因为 return statement ReferenceError 标记为 partial (x1_ok=true, x2-x5_ok=false). 这不影响 ship 的 file,因为 file 在 phase 1 (x1) 就已经全部写入。

如果 R19 启动,主要工作按 R19_PLAN.md §6-12 GPU training schedule + §13-18 投稿路线 + §19-23 wet-lab milestones 进行。

**Date**: 2026-09-26 (Wf-27 ship)
