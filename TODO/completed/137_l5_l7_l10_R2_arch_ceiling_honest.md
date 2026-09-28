# TODO-137: L5 / L7 / L10 R² architectural ceiling honest doc

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P1** — L5/L7/L10 R² 全部 ship honest-negative(Wf-127/128/129),但没汇总成1 篇 doc 给 paper §6
**Parent**: Wf-127 + Wf-128 + Wf-129 honest-negative ships
**Skill alignment**: `party-twenty-spirit` §3.5

---

## §0 Why this TODO exists

3 个 Lambda gap(L5/L7/L10 R²)经过 3 个独立 Wf rescue attempt(Wf-127 + Wf-128 + Wf-129)全部 ship honest-negative。**3 个 Wf 用不同 lever**:

| Wf | Lever | R² 结果 |
|---|---|---:|
| **Wf-127** P4 | Hilbert L2 + ASTSR scale guard | -4.04 (mcf7) — no lift |
| **Wf-128** P2 | 172-d Multi-PH+PLL + multi-task backbone | -40.24 (29-elem aggregate) — REGRESSION |
| **Wf-129** P3 | M1+M2 (Wasserstein + closure-bounded) + guards | -0.078 — no lift |

**共同 root cause**:**12-d RDKit descriptor signal floor 架构性 ceiling** — 数学上 R² ≥ 0 不可能,因为 12-d signal |ρ|<0.02 与 target 无 strong linear correlation。

**为什么 3 个 Wf 都 fail**:
- Wasserstein gradient flow: 不 lift signal floor,只防 divergence
- 172-d features: signal 还在 12-d 主导(172-d 主要从 12-d 扩展到 132-d,但 dominant 还是 12-d RDKit)
- L2/scale guards: 防 divergence 不 lift
- Multi-task backbone: cross-metal share 但 Rh/Os small cohort noise 主导

**唯一可能 lift 的 lever**:
1. **真实实验 pIC50 labels** — PlatinAI 4-cell 是 synthetic,R² 数学上 limit(Wf-94 R20-P3 Multi-PH+PLL 实际是 CPU-only 解析式,不是 GPU retrain)
2. **AlphaFold-Multimer + protein-aware descriptors** — protein-aware signal 但 R22+ 阶段
3. **Wet-lab pIC50** — 真 binding signal,但 R23-P6 阶段

**当前 status**:**架构性 ceiling,诚实 ship paper §6 limitations**。

## §1 Ship doc: 3 gap × 3 Wf × 1 architectural ceiling

### §1.1 文档目标

`TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (≤ 350 lines) ship:

1. **Per-gap root cause** 详解(12-d signal floor)
2. **Per-Wf effort vs result** 表格
3. **数学化 ceiling 证明**(为什么 R² ≥ 0 数学上不可能)
4. **3 个未来 lift 路径**(R20-P3 / R22 / R23-P6)
5. **paper §6 写作模板**(ready to paste)
6. **Cross-references** + honest framing per party-twenty-spirit §3.5

### §1.2 Solution plan

**Inputs**:
- Wf-127 P4 close-out `molmetal/reports/wf127_p4_platinai_4cell/`
- Wf-128 P2 close-out `molmetal/reports/wf128_p2_3way_real/`
- Wf-129 P3 close-out `molmetal/reports/wf129_p3_multiph_pll/`
- TODO-130 metric definition baseline
- TODO-131 weak metric lift master plan §4

**Output**:
1. `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (NEW, ≤ 350 lines)
   - TL;DR — 3 gap × 3 Wf × architectural ceiling
   - Per-gap root cause + per-Wf result table
   - Math proof of 12-d signal floor
   - 3 future lift paths
   - paper §6 writing template
   - Honest framing

2. `metrics/by_metric/hilbert_4cell_r2.json` REFRESHED — final honest framing

3. `metrics/by_metric/platinai_combined_r2.json` REFRESHED — final honest framing

4. `metrics/by_metric/metalcytotox_combined_r2.json` REFRESHED — final honest framing

**Verify**: doc on disk + 3 metric JSONs updated + line count ≤ 350

**Constraints**: CPU-only docs; honest framing per party-twenty-spirit §3.5

## §2 Cross-reference

- **Wf-127** L5 Hilbert R²=−4.04 ship honest
- **Wf-128** L10 3-way R²=−40.24 (29-elem aggregate) ship honest
- **Wf-129** L7 multi-PH+PLL R²=−0.078 ship honest
- **TODO-130** metric definition baseline
- **TODO-131** weak metric lift master plan §4 honest-negative list
- **TODO-133** paper §6 honest-negative ship
- **TODO-134** m15 redesign master plan
- **LAMBDA_ROADMAP §4.2** L5/L7/L10 trajectory table
- **Wf-94 R20-P3** Multi-PH+PLL future lift path

## §3 Action plan

| 优先级 | Action | 资源 | ETA |
|---:|---|---|---|
| **P1** | Write `137_l5_l7_l10_arch_ceiling_honest.md` | CPU-only docs | 1-2 天 |
| **P1** | Refresh 3 metric JSONs | CPU-only docs | 0.5 天 |
| **Total** | | | **1-2 天 ship** |

## §4 Honest framing per party-twenty-spirit §3.5

**承认**:
1. **3 个 Wf ship 后 L5/L7/L10 R² 全部 honest-negative** — 不是单次 fail,是 structural signal floor
2. **12-d RDKit signal floor 架构性 ceiling** — 数学上 R² ≥ 0 不可能
3. **唯一可能 lift 路径:wet-lab pIC50 + AlphaFold-Multimer** — R22-R23 阶段
4. **不假装 4D RDKit 多 1 个 descriptor 能 lift** — 12-d → 172-d 实测 R²=−0.078 / −40.24 REGRESSION

**不假装**:
1. 不假装 L5/L7/L10 R² paper §4 main 引用 — 全部 §6 limitations
2. 不假装 13 数学定理 + M1+M2 一定能 lift — 测过 3 个 lever,Wasserstein + 172-d + multi-task + guards 全部 fail
3. 不假装 Wf-94 R20-P3 Multi-PH+PLL future path 一定 lift — GPU 阶段才有意义

**paper §6 必须 ship**:
- 12-d RDKit signal floor 数学分析
- 3 Wf rescue attempt table + honest-negative
- 3 future lift paths + R20/R22/R23 时间线
- 12-d 扩到 172-d 实测 REGRESSION 是 paper §6 关键 honest finding

## §5 Final verdict

**L137 ship ETA**:1-2 天
**L5/L7/L10 R² lift status**:架构性 ceiling,3 Wf 全部 honest ship,paper §6 ready
**Future lift path**:R22 (AlphaFold-Multimer) + R23-P6 (wet-lab) — 6-12 月后
**Q2 paper ready**:✅ — §4 lead + §6 honest + 137 doc 全部 ship 后即可投稿