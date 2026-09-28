# EXECUTION_ORDER — What to do next + Why + How

**Date**: 2026-09-26 (R18 close-out — `R18_PLAN.md` §15 + INDEX §10 + `completed/73_r18_closeout_final.md`)
**Audience**: 任何接手这个项目的人；包括 Hugo Lee 自己的下一步行动
**Purpose**: 把整个 TODO 目录下的内容按"执行优先级"组织起来，从"现在做什么"出发反向导航
**Status**: Active
**Latest update**: 2026-09-26 (TODO-82 metric gap filling plan ship)

---

## 0. 当前位置 (where we are — 2026-09-26)

### 当前状态: 全部 R18 PLAN 真活 100% ship

**R18 PLAN 47-item completion matrix**: P0 4/4 ✅ + P1 7/7 ✅ + P2 5/5 ✅ + Pitfall-T1 5/5 ✅ + Pitfall-T2 13/16 ✅ + Integration 5/5 ✅ + Bench 5/5 ✅ = **44/47 ship, 3 DEFER** (Pitfall-Tier-2 P2.5 Phases C-E = MMseqs2 + AMBER + YASARA 第三方闭源)。

**R18 close-out MEASURED** (CPU-only, this build):
- 19 wf reports ship in `TODO/completed/{53..74}_*.md`
- 52+ R18 modules import OK (9 math_skills + 9 adapters + 14 lam_chem/search_alg/reward + 5 scripts + 15+ verify/bench/test)
- **≈1 235 tests pass** (R18 ≈1 018 + R17 baseline ≈217 + 30/30 decode probes)
- **M25 Hilbert PD 452.16× speedup @ n=30** vs evolutionary (CPU smoke)
- 32 modules × PlatinAI 5K (160K primitives in 3.30s); MetalCytoToxDB 5-metal (800K primitives in 15.05s)
- 5-fold CV 32 modules × 5 folds × 2 runs mean_acc = 0.9375
- paper build healthy (132+ 页 v1.7, 0 unresolved refs, 6 new bib keys)
- 23 个新机制 M22-M44 + 8 skills (5 R17 + 3 NEW R18) + §6 honest limitations

**R17 + 之前 round 全部 ship** (R12 / R13 / R14 / R15 / R16 / R17 Pilot + R17 close-out):
- 30/30 decode probes = 1.000 / 14/14 evaluators / 6 shim fixes / 7/7 pitfall guards
- m14 Decode ratio 1.000 / m17 DrugOOD OOD 0.891/0.697/0.748 / Deflexpressor R²=1.0000 / m11 n_distinct 4-7

**R17 已知失败** (永久 BLOCK, paper §6 走 honest framing):
- m1 Ru AUC 0.712 (gate ≥ 0.86) / m2 Pt A2780 0.350 / MCF-7 0.584 / A549 0.630 / A2780cis 0.648

---

## 1. GPU 训练 + 现在该做什么 (priority order)

### GPU 训练: Blackwell sm_120 可用 (Wf-23 in flight), RTX 5090 备用

**Why**: 全部 R18 真活 CPU-only ship，但 PROJECTED lift（m3 Vina −8.5 → −10.5+ / m7 Diversity 0.137 → 0.75 / m4 SR 0.00 → 0.65 / m16 Pearson r 0.207 → 0.75 / ODE 6 ms → 0.06 ms / ASTSR R² ≥ 0.62）都需要 Blackwell GPU retrain 才能 MEASURED。
**How**: Wf-23 GPU ultracode queue (in flight, 4-周排期) — Blackwell sm_120 single-card + torch 2.6+CUDA 12.6 baseline
**Tasks** (Phase 1-4 ultracode): Phase 1 Transport Relax / Phase 2 SE(3) SB / Phase 3 Multi-PH+Hilbert / Phase 4 Joint retrain (M22-M30 10 个机制 + Bass martingale)
**Fallback**: RTX 5090 (32 GB) Blackwell 单卡备份；超 30-min CI budget 时 fallback
**Status**: ⏳ Wf-23 in flight (Wf-21 lambda 5 remaining ship today, Wf-22 mid-flight, Wf-23 GPU queue next)

### R19 优先级: GPU 训练 + paper v2 + wet-lab 合作

1. **Block GPU training** —— M22-M30 10 个机制 Blackwell retrain (Wf-23, 4 周)
2. **Phase A re-attempt** —— multi-GPU + scaffold split + 100-pocket × 10-seed 重测 m1/m2
3. **Wet-lab 第一个 milestone** —— ORNL/PKU/协和 m18 protein-metal binding 实验
4. **paper v2** —— 替换 MEASURED → ACTUAL GPU retrain numbers；保留 §6 honest limitations

### Phase A — R17 Tier-1 AUC re-attempt (R19 second)

**Why**: Blackwell + ultracode blacklist 解除后，m1 Ru AUC + m2 Pt A2780 可重测
**How**: `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` → R19 升级版 (multi-GPU + scaffold split + 100-pocket × 10-seed)
**Tasks**: m1 Ru 5-metal multi-task → ≥ 0.86 / m2 A2780cis resistance features → ≥ 0.85 / m2 A2780 数据诊断 / paper §4 abstract 重写
**Status**: ⚠️ 待 GPU 训练就绪后启动

### Phase B — Wet-lab 合作 (R19 third)

**How**: ORNL / PKU / 协和 collaboration — m18 protein-metal binding 实验 first
**Status**: 📋 R19 排期

### Phase C — 长期愿景

- R20+: Quotient-Space Diffusion full-deploy + Bass martingale diffusion full-deploy
- 投稿至 CAS Zone 2（中科院二区）journal

---

## 2. 按场景导航

### 场景 1: "我刚接手这个项目，先看什么？"

**阅读顺序** (建议 2 小时):
1. `README.md` (200 lines) — 项目概述 + R17/R18 终态
2. `EXECUTION_ORDER.md` (本文) — 当前位置 + 下一步
3. `01_strategy/PROJECT_STATEMENT.md` (200 lines) — paper 立项 + 题目 + 3 贡献
4. `01_strategy/INSIGHTS_FROM_WORKFLOWS.md` (180 lines) — 7 个关键启发
5. `02_metrics/METRICS_TARGETS.md` (517 lines) — 20 指标 × Tier 1/2/3 对照
6. `03_plans/R18_PLAN.md` §15 — R18 close-out synthesis

### 场景 2: "我需要修一个 metric regress"

**步骤**: `MECHANISM_METRIC_MAPPING.md` §4 (failure mode → mechanism diagnostic) → §5 (ablation 设计) → 跑 `tests/test_pitfall_guards.py` → 跑 `scripts/eval_all_evaluators.py`

### 场景 3: "我要添加新机制或新 metric"

`MECHANISM_METRIC_MAPPING.md` §1 (21 机制 + 20 指标含 R18 M22-M44) → §2 (新指标) → §3 (contribution matrix) → §4 (failure mode) → §5 (ablation) → 更新 `mol-metal-mechanism-mapping/SKILL.md`

### 场景 4: "我要写 paper section"

| Section | 主要来源 |
|---|---|
| §1 Introduction | `PROJECT_STATEMENT.md` §3 (3 个核心贡献) |
| §2 Related Work | `01_research/sota_papers_2025_2026.md` |
| §3 Method | `PROJECT_STATEMENT.md` §3-5 + `R18_PLAN.md` (§3.4 SE(3) SB / §3.5 Transport Relax / §3.6 Multi-PH+Hilbert / §3.7 Monge-Ampère+WGF+SB-Bass+Bergman) |
| §4 Results | `METRICS_TARGETS.md` Table 2 |
| §4.5 Honest Negative | `INSIGHTS_FROM_WORKFLOWS.md` §3 + `completed/58_weak_to_strong.md` |
| §5 Discussion | `PROJECT_STATEMENT.md` §7 |
| §6 Limitations | `MERGE_GATE.md` §6 + `R18_PLAN.md` §15.4 (5 段 honest gap) |
| §7 Future Work | `R18_PLAN.md` §15.5 (R19+ roadmap) |
| Appendix A: Closure | `13_lambda_clickchem/molecular_lambda_calculus.md` |
| Appendix B: 84-rule matrix | `completed/35_lambda_b2_b3_deflexpressor.md` §1 |

### 场景 5: "我要加新 skill"

参考 `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` 结构模板。8 个现有 skill:
- `mol-metal-lambda-calculus` / `mol-metal-cfm-training` / `mol-metal-click-chemistry` / `mol-metal-anticancer` / `mol-metal-mechanism-mapping` (5 R17)
- `mol-metal-r18-math` / `mol-metal-paper-writing` / `party-twenty-spirit` (3 NEW R18)

---

## 3. TODO 目录结构 (R18 close-out 后)

```
TODO/
├── README.md / EXECUTION_ORDER.md / INDEX.md
├── 01_strategy/  (PROJECT_STATEMENT / INSIGHTS_FROM_WORKFLOWS)
├── 02_metrics/   (METRICS_TARGETS / MECHANISM_METRIC_MAPPING / EVALUATORS_MIGRATED)
├── 03_plans/     (R17_AUC_RECOVERY / R18_MATH_MIGRATION / R18_PLAN / WEAK_TO_STRONG / MERGE_GATE)
├── 04_history/   (9 legacy + 6 R17 SUPERSEDED banner)
├── 01_research/ ... 13_lambda_clickchem/   ← topic dirs
├── pending/  (6 active)
├── completed/ (74 files, NN ≤ 74)   ← R18 + R17 + earlier ship
└── archive/  (4 files)
```

---

## 4. 单文件导航 (按文件大小排序)

| File | Lines | 用途 | 何时读 |
|---|---|---|---|
| `README.md` | 200 | 项目概述 | 接手项目时 |
| `EXECUTION_ORDER.md` | 本文件 | 导航 + 下一步 | 每次 session 开头 |
| `INDEX.md` | 150 | 状态跟踪 (R18 §10 ship) | 找特定 doc 位置 |
| `01_strategy/PROJECT_STATEMENT.md` | 195 | paper 立项 | 写 paper §1 |
| `01_strategy/INSIGHTS_FROM_WORKFLOWS.md` | 180 | 7 个启发 | 写 paper §4.5 / rebuttal |
| `02_metrics/METRICS_TARGETS.md` | 517 | 20 指标 + SOTA | 写 paper §4 表格 |
| `02_metrics/MECHANISM_METRIC_MAPPING.md` | 228 | 21 机制 × 20 指标 | metric regress / ablation |
| `02_metrics/EVALUATORS_MIGRATED.md` | 306 | 14 evaluators | 配 evaluator 时 |
| `03_plans/R18_PLAN.md` | 524 | R18 + §15 close-out | 写 paper / GPU 训练时 |
| `03_plans/R18_MATH_MIGRATION_PLAN.md` | 291 | R18 战略 | 启动 R19 时 |
| `03_plans/R17_AUC_RECOVERY_PLAN.md` | 345 | Phase A 重启模板 | Phase A re-attempt 时 |
| `03_plans/MERGE_GATE.md` | 324 | merge 5 gate | 合并两条线时 |

---

## 5. 现在立刻执行的 3 件事 (this week's checklist — updated 2026-09-26)

**TODO-82 Metric Gap Filling Plan** (NEW!): `pending/82_metric_gap_filling_plan.md` — 7 high-priority gap + 3 medium gap + 6 phases × 10 days + Wf-29~Wf-34 plan

- [ ] **Task 1**: 启动 Wf-29 (Phase A 修 bug + smoke benchmark) — `train_se3sb.py` AttributeError fix + GPU 1 跑 5-seed × 10-epoch
- [ ] **Task 2**: 启动 Wf-30 (Phase B Vina Dock GPU retrain) — 10 pockets × 5 candidates × GPU 1 + `train_se3sb.py` 5-seed × 50-epoch on GPU 0
- [ ] **Task 3**: 启动 Wf-31 (Phase C m1/m2/m7) — `train_ru_5metal_se3sb.py` + `train_cisplatin_resistance_v2.py` + `bench_multi_ph_diversity.py`

**Cross-skill 配套**: 启动 Wf-34 (Phase F NEW skill `mol-metal-gpu-execution` ship) + 用 8 现有 skills 配合 (cfm-training / r18-math / anticancer / mechanism-mapping 等)。

## 6. 等通知 (R18 ultracode 状态)

- ✅ 全部 R18 wf ship (Wf-1 → Wf-19 + Wf-21 + Wf-22 mid-flight)
- ⏳ Wf-23 GPU ultracode queue (in flight)

## 7. 用户 (Hugo) 决策点

| Decision | 选项 |
|---|---|
| **D1**: R17 Tier-1 AUC fail 怎么办? | A) drop Tier-1 claim 走 honest negative (R18 已选) / B) Phase A re-attempt (R19 GPU 优先) |
| **D2**: A2780 0.350 怎么办? | A) drop A2780 报 3-cell / B) 调查数据 bug / C) GPU re-attempt |
| **D3**: R18 P0-1 SE(3) SB GPU 失败时? | A) 退到 midpoint ODE / B) 标 incremental / C) skip |
| **D4**: Wet-lab validation? | A) R19 wet-lab / B) skip / C) external collaboration |
| **D5**: Blackwell sm_120 30-min CI budget 超时? | A) RTX 5090 fallback / B) multi-GPU split / C) DEFER |