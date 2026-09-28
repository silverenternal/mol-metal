# TODO-82 — Metric Gap Filling Plan (post-R18 close-out)

**Date**: 2026-09-26
**Status**: 🚀 **Active — immediate next work**
**Priority**: **HIGHEST** (R18 PLAN 数学骨架 ship, 但实际 MEASURED lift 未达成)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `WEAK_TO_STRONG_PLAN.md` §1.2 (8 WEAK metrics)
**Sibling inputs**:
- `WEAK_TO_STRONG_PLAN.md` §1.1 8 STRONG + §1.2 8 WEAK
- `METRICS_TARGETS.md` 20 metrics × Tier 1/2/3
- `MECHANISM_METRIC_MAPPING.md` 23 R18 mechanisms M22-M44
- `MOL-METAL-ANTICANCER` skill (m1/m2/m21)
- `MOL-METAL-CFM-TRAINING` skill (m14 + CFM retrain)
- `MOL-METAL-CLICK-CHEMISTRY` skill (m12 synthesis)
- `MOL-METAL-MECHANISM-MAPPING` skill (cross-skill diagnostic)
- `MOL-METAL-PAPER-WRITING` skill (paper §4 honest framing)
- `MOL-METAL-R18-MATH` skill (9 R18 theorems)
- `PARTY-TWENTY-SPIRIT` skill (0 容忍 不掩盖)

---

## §0. Why this plan exists

Wf-1~28 全部 ship,但**真正"算法/数学/模型"ship** vs **"实际 MEASURED metric lift"** 之间有 7 个 gaps:

| Gap | Metric | MEASURED | SOTA target | Δ | Status |
|---|---|---|---|---|---|
| **G1** | m3 Vina Dock median | **-1.2885 kcal/mol** | -9.37 (TransDiffSBDD) | +8.08 ❌ | R18 SE(3) SB ship 但未 GPU 跑 |
| **G2** | m1 Ru AUC | **0.7123** | 0.81 (Krasnov) | -0.098 ❌ | Wf-22 retrain honest-neg 0.630 |
| **G3** | m2 Pt A2780cis | **0.648** | 0.85 (Rusanov) | -0.202 ❌ | Wf-22 retrain honest-neg 0.630 |
| **G4** | m7 Diversity Tanimoto | **0.1065** | 0.65 | -0.543 ❌ | Wf-25 Hilbert R² -0.443 |
| **G5** | m16 Affinity Pearson r | **0.207** | 0.86 (Boltz-2) | -0.65 ❌ | seed-fragile |
| **G6** | m19 Multi-metal LOMO | **0.535** | 0.70 | -0.165 ❌ | R18 Hilbert 没跑 |
| **G7** | Vina Dock on 10 CrossDocked pockets | **5/10 pockets MEASURED** (1h36/3eml/3ny8/4aaw/4rlu) | 100 pockets × 3 seeds | **95% pockets missing** ❌ | receptor PDB not staged |

**总 missing ship**: 7 个 high-priority gap + 3 个 medium gap (m4 Success Rate, m8 PB-Valid prod, m10 Vina Min) = **10 个真实缺口**

---

## §1. Cross-skill strategy

利用 8 个现有 skills + 1 个 NEW skill (planned in §3) 来快速、聪明地解决:

| Gap | 主要 skill | 辅助 skill | 输出 |
|---|---|---|---|
| **G1 Vina Dock** | mol-metal-cfm-training | mol-metal-r18-math (T1 SE3 SB) | `molmetal/scripts/train_vina_se3sb.py` |
| **G2 m1 Ru AUC** | mol-metal-anticancer | mol-metal-mechanism-mapping | `molmetal/scripts/train_ru_5metal_se3sb.py` |
| **G3 m2 Pt A2780cis** | mol-metal-anticancer | mol-metal-click-chemistry | `molmetal/scripts/train_cisplatin_resistance_v2.py` |
| **G4 m7 Diversity** | mol-metal-r18-math (T5 Multi-PH) | mol-metal-click-chemistry | `molmetal/scripts/bench_multi_ph_diversity.py` |
| **G5 m16 Affinity** | mol-metal-anticancer | mol-metal-cfm-training | `molmetal/scripts/train_affinity_regression.py` |
| **G6 m19 LOMO** | mol-metal-mechanism-mapping | mol-metal-anticancer | `molmetal/scripts/train_multimetal_lomo.py` |
| **G7 10-pockets Vina** | mol-metal-cfm-training | mol-metal-click-chemistry | `molmetal/scripts/dock_10pockets.py` |
| **NEW: GPU execution skill** | — | — | `.claude/skills/mol-metal-gpu-execution/` |

---

## §2. 6 phases (Week 1-4, GPU Blackwell + RTX 5090 dual-GPU)

### Phase A — 修 train_se3sb.py bug + 立即 5-seed benchmark (Wf-29 / Day 1-3)

| Task | 时间 | Skill |
|---|---|---|
| Fix `train_se3sb.py:613` AttributeError | 30 min | (1-line fix) |
| 跑 `train_se3sb.py` on GPU 1 5-seed × 10-epoch × MetalCytoToxDB (smoke 测可用) | 1 hour | mol-metal-cfm-training |
| 跑 `train_multiph.py` on GPU 1 5-seed × PlatinAI 5K (smoke 测可用) | 30 min | mol-metal-r18-math (T5) |
| 跑 `train_transport.py` on GPU 1 5-seed × MetalCytoToxDB | 30 min | mol-metal-r18-math (T2) |
| **Output**: 4 个 benchmark JSON + 实际 MEASURED wall-time + memory | 1 day | — |

### Phase B — m3 Vina Dock GPU retrain (Wf-30 / Day 3-5)

| Task | 时间 | Skill |
|---|---|---|
| 准备 10 CrossDocked pocket receptor PDB staging (1h36/3eml/3ny8/4aaw/4rlu + 5 more) | 2 hours | mol-metal-cfm-training |
| `molmetal/scripts/dock_10pockets.py` (新) — 跑 100 candidates × 10 pockets × GPU 1 | 4 hours | mol-metal-cfm-training |
| 5-seed × 50-epoch × `train_se3sb.py` on GPU 0 (Blackwell) | ~12 hours | mol-metal-cfm-training |
| **Output**: m3 Vina Dock median + std across 10 pockets × 5 seeds | 2 days | — |

### Phase C — m1 + m2 + m7 GPU retrain (Wf-31 / Day 5-7)

| Task | 时间 | Skill |
|---|---|---|
| `train_ru_5metal_se3sb.py` (新) on GPU 0, 5-seed × 50-epoch × MetalCytoToxDB 5-metal | ~8 hours | mol-metal-anticancer |
| `train_cisplatin_resistance_v2.py` (新) on GPU 0, 5-seed × 50-epoch × PlatinAI 4-cell | ~8 hours | mol-metal-anticancer |
| `bench_multi_ph_diversity.py` (新) on GPU 1, 5-seed × PlatinAI 5K diversity | ~2 hours | mol-metal-r18-math (T5) |
| **Output**: 实际 m1/m2/m7 MEASURED AUC + lift vs R17 baseline | 2 days | — |

### Phase D — m16 + m19 + m8 GPU bench (Wf-32 / Day 7-9)

| Task | 时间 | Skill |
|---|---|---|
| `train_affinity_regression.py` (新) on GPU 1, PlatinAI 4-cell affinity model | ~4 hours | mol-metal-anticancer |
| `train_multimetal_lomo.py` (新) on GPU 0, 5-seed × 50-epoch × leave-one-metal-out | ~6 hours | mol-metal-mechanism-mapping |
| `bench_pb_valid.py` (新) on GPU 1, 10 pockets × 5 candidates PB-Valid | ~2 hours | mol-metal-cfm-training |
| **Output**: 实际 m16/m19/m8 MEASURED values | 2 days | — |

### Phase E — paper v2 actual data replace PROJECTED (Wf-33 / Day 9-10)

| Task | 时间 | Skill |
|---|---|---|
| 重新生成 paper §4 Table 2 with MEASURED values | 2 hours | mol-metal-paper-writing |
| 重新生成 paper §6 Limitations with MEASURED honest-neg | 1 hour | mol-metal-paper-writing |
| 重新生成 paper §3.4-3.7 with MEASURED SE(3) SB lift | 2 hours | mol-metal-r18-math |
| `paper/main.pdf` rebuild to 200+ pages | 30 min | — |
| **Output**: paper v3 with 全部 MEASURED | 1 day | — |

### Phase F — 新 skill 注册 + final synthesis (Wf-34 / Day 10)

| Task | 时间 | Skill |
|---|---|---|
| 新建 `.claude/skills/mol-metal-gpu-execution/` (NEW skill) | 2 hours | party-twenty-spirit |
| 更新 8 个现有 skills 加 R19 GPU retrain reference | 1 hour | (cross-skill) |
| 写 `TODO/completed/82_metric_gap_filling.md` final synthesis | 30 min | — |
| **Output**: NEW skill + 9 skills updated + final report | 1 day | — |

**Total**: ~10 days wall-time (mostly GPU training), **1 human week**

---

## §3. NEW skill: `mol-metal-gpu-execution`

**Rationale**: 现有 8 skills 没有专门"如何用 Blackwell + RTX 5090 跑 R18 真活"的执行 skill。新 skill 整合:
- GPU env check (`check_blackwell_env.py`)
- venv setup (`setup_molmetal_env.sh`)
- Multi-GPU 调度 (Blackwell 主训练 + RTX 5090 对照/并行 seed)
- 5-seed × 50-epoch 标准 bench protocol
- MEASURED vs PROJECTED 标签 strict
- Honest-negative reporting (per party-twenty-spirit §3.5)

**Skill structure** (~150 lines):
```
mol-metal-gpu-execution/SKILL.md
  - frontmatter (name + description + tools)
  - §1 When to use (任何 Blackwell RTX 训练)
  - §2 Hardware (Blackwell sm_120 + RTX 5090 + kanzi contention policy)
  - §3 venv setup (molmetal/.venvs/mol-metal-venv symlink)
  - §4 Standard train script template
  - §5 Multi-GPU 调度 (CUDA_VISIBLE_DEVICES, 避开 kanzi)
  - §6 5-seed × 50-epoch standard protocol
  - §7 MEASURED vs PROJECTED labels
  - §8 honest-negative reporting
  - §9 Worked example (run train_se3sb.py on GPU 1)
  - §10 Cross-reference (其他 8 skills)
```

---

## §4. Execution timeline + Workflow 计划

| Workflow | Phase | 期望 output | Wall-time |
|---|---|---|---|
| **Wf-29** | A (修 bug + smoke) | 4 benchmark JSON | 1 day |
| **Wf-30** | B (Vina Dock GPU retrain) | m3 -1.2885 → PROJECTED -8.5 | 2 days |
| **Wf-31** | C (m1/m2/m7 GPU retrain) | m1/m2/m7 MEASURED + lift | 2 days |
| **Wf-32** | D (m16/m19/m8 bench) | m16/m19/m8 MEASURED | 2 days |
| **Wf-33** | E (paper v3) | main.pdf 200+ pages | 1 day |
| **Wf-34** | F (NEW skill + final) | mol-metal-gpu-execution + 9 skills updated + synthesis | 1 day |
| **TOTAL** | | | **~9 days** |

---

## §5. Honest framing

按 party-twenty-spirit §3.5 (0 容忍 + 不掩盖):

| Risk | Mitigation |
|---|---|
| GPU 训练失败 (torch / venv / OOM) | 立即报告 honest-negative,不假装 ship |
| Wf-22 历史 honest-neg (m1 0.712 → 0.630) | 这次用 R18 modules (se3sb/trans/multiph) 重跑,期望 lift 但不假装 guaranteed |
| Phase D m16 affinity seed-fragile | 5-seed mean + std 报告,不取 max |
| Phase B Vina 100×3 slow (Wf-16 mock 估算 100 hours) | 缩减到 10×5 = 50 dockings,wall-time 6-8 hours |
| kanzi GPU 0 contention | Wf-22 显示 kanzi 18-69% util,Phase B 用 GPU 1,Phase C/D 也用 GPU 1 |

**核心原则**: MEASURED 数字 > PROJECTED 数字。**wf failure ≠ 真活 failure**, 真实磁盘文件 + 测试通过 才是 MEASURED。

---

## §6. 验收 criteria

### Phase A 完成标准:
- 4 个 train scripts (se3sb / transport / multiph / r17 baseline) 实际跑通 GPU 1
- 4 个 benchmark JSON 输出到 `molmetal/reports/wf29_*`
- 实际 wall-time 报告 (per 5-seed, per pocket)

### Phase B 完成标准:
- m3 Vina Dock median: -1.2885 → **-8.5 以上** (PROJECTED) 或 honest-neg
- 10 pockets × 5 candidates × 5 seeds = **250 docking evaluations**
- 报告 delta vs R17 baseline (-1.2885)

### Phase C 完成标准:
- m1 Ru AUC: 0.7123 → **0.81 以上** (PROJECTED, Krasnov tier) 或 honest-neg
- m2 Pt A2780cis: 0.648 → **0.85 以上** (PROJECTED) 或 honest-neg
- m7 Tanimoto: 0.1065 → **0.65 以上** (PROJECTED) 或 honest-neg

### Phase D 完成标准:
- m16 Affinity r: 0.207 → **0.50 以上** (更 realistic target, Boltz-2 0.86 太高) 或 honest-neg
- m19 LOMO: 0.535 → **0.70 以上** 或 honest-neg
- m8 PB-Valid: 1.000 (smoke) → **0.94 across 10 pockets** 或 honest-neg

### Phase E 完成标准:
- paper v3 main.pdf 200+ pages, 0 unresolved
- §4 Table 2 全部 MEASURED (no PROJECTED)

### Phase F 完成标准:
- NEW skill `mol-metal-gpu-execution` ship + 9 skills R19 updated
- `TODO/completed/82_*.md` final synthesis

---

## §7. 不在本 plan 中的事

1. **Wet-lab 实际 IC50** (R19 deferred to ORNL/PKU outreach)
2. **Pitfall P2.5 Phases C-E** (MD relaxation / AMBER / YASARA — 3rd-party binary deps)
3. **MMseqs2 binary** (CPU-only fallback shipped in Wf-7)
4. **Q-Bench / iMoLD update** (deferred to R19 §13-18 paper v3 timeline)
5. **Data generation beyond PlatinAI 5K + MetalCytoToxDB** (deferred to wet-lab outreach)

---

## §8. Cross-reference

- `WEAK_TO_STRONG_PLAN.md` — 20 metric ledger (this plan addresses §1.2 8 WEAK metrics)
- `METRICS_TARGETS.md` — Tier 1/2/3 SOTA baselines
- `MECHANISM_METRIC_MAPPING.md` §1/§5/§6 — 23 R18 mechanisms (M22-M44)
- `R18_PLAN.md` §15 — R18 close-out (32 modules ship)
- `R19_PLAN.md` §6-12 — GPU training schedule (this plan is the execution of §6-12)
- `.claude/skills/*/SKILL.md` — 8 existing skills + 1 NEW (mol-metal-gpu-execution)
- `TODO/completed/75_blackwell_retrain.md` — Wf-22 honest findings
- `TODO/completed/76_r18_blackwell_retrain_v2.md` — Wf-23 partial ship
- `TODO/completed/78_hilbert_multiph_real.md` — Wf-25 honest findings (R² instability)

---

## §9. Final verdict

**This plan is the execution layer for R19_PLAN.md §6-12 GPU training schedule.**

If Wf-29 ~ Wf-34 ALL ship:
- 7/8 WEAK metrics 真实 MEASURED + R18 lift
- paper v3 with 全部 MEASURED
- NEW skill `mol-metal-gpu-execution` for future R19+ cycles
- **R18 close-out → R19 close-out** 完整过渡

If some metrics stay WEAK after Wf-29 ~ Wf-34, those become **R20 plan items** (not R19).

**Date**: 2026-09-26 (created)
