# Plan A · Step 3 — Closed-loop 全跑 + Tier-2 main figure

**Date:** 2026-10-07 · **GPU:** ~30 hr · **LOC:** ~300 (benchmark pipeline + paper figure scripts) · **Wall:** 2-4 wk · **Status:** DESIGN, gated on Step 1+2 completion

## §0 这是 Tier-2 (二区) paper 的 main figure 阶段

Plan A overview §2.2 把 Step 3 排最后,因为:
- 依赖 Step 1 (DMPNN value) + Step 2 (CFM prior) 都完工
- 依赖 dmpnn_rh widen + rh_lomo 数据 fix 完工 (让 pIC50 在 Rh 上可用)
- 工作量主要在 **benchmark pipeline + 评估协议 + paper figure 生成**,不是算法

**关键产物**:
1. 完整 closed-loop benchmark 在 9 cell-line (NCI-60 子集) × 5 metal (Pt/Ru/Ir/Au/Rh) × 10 seed = **450 records**
2. 4 个 metric table:pIC50 vs target / SA / QED / synthesis path length
3. Tier-2 paper Section 4 (main figure) + Section 5 (ablation)

## §1 现状 (MEASURED, 2026-10-07)

Step 1 + Step 2 完工后:
- Lambda MCTS reward = `0.6·DMPNN_pIC50 - 0.2·(SA-3) + 0.2·QED + PLL_bonus`
- Lambda MCTS prior = CFM prior head (learned)
- 路径里每一步 reaction 都"知情"pIC50 目标

**Step 3 验证**:closed-loop 是否真的产出**高 pIC50 candidate**。

## §2 Step 3 目标 — 跑 main benchmark + 出 paper figure

### §2.1 Benchmark 协议

```
Cell-line set (NCI-60 subset): A2780, MCF7, A549, NCI-H460, OVCAR-3, 
                               PC-3, MDA-MB-231, HT-29, HCT-116  (9)
Metal set:                   Pt, Ru, Ir, Au, Rh                   (5)
Seed per (cell-line, metal): 10
Total records:               9 × 5 × 10 = 450
Per record:                  n_simulations=100, max_depth=6, top_k=20
Total GPU-CPU time:          ~30 hr (parallel 50 hr Blackwell + 5090)
```

**Per-record metrics**:
1. `top_1_pIC50` (predicted by DMPNN) — main metric
2. `top_1_SA`, `top_1_QED`, `top_1_Validity` — molecular quality
3. `synthesis_path_length` (median across top-10)
4. `n_distinct_path` (diversity of distinct reaction paths reaching high pIC50)
5. `S2T` (simulations to threshold) — efficiency
6. `ER@N` (exploit ratio) — convergence

### §2.2 Ablation table (Section 5)

| Row | Prior | Value | Lipinski gate | Expected |
|---|---|---|---|---|
| 1 | Symbolic | hand-crafted | ✓ | baseline |
| 2 | Symbolic | DMPNN pIC50 | ✓ | +pIC50 lift (Step 1) |
| 3 | CFM learned | hand-crafted | ✓ | +prior lift (Step 2 alone) |
| 4 | CFM learned | DMPNN pIC50 | ✓ | **+combined lift** (Step 1+2) |
| 5 | CFM learned | DMPNN pIC50 | ✗ (open) | +open-search lift |
| 6 | random | DMPNN pIC50 | ✓ | sanity check |

**每行 450 records × ~2 min/record = ~15 hr wall / 行**,**总 ~90 hr CPU**。并行 Blackwell + 5090 = ~45 hr wall。

### §2.3 文件清单

| 文件 | 操作 | LOC |
|---|---|---|
| `molmetal/scripts/plan_a_step3_bench.py` | NEW closed-loop benchmark | ~250 |
| `molmetal/scripts/plan_a_step3_aggregate.py` | NEW ablation table aggregation | ~150 |
| `molmetal/scripts/plan_a_step3_figure.py` | NEW paper figure generation (matplotlib) | ~200 |
| `configs/plan_a_step3.yaml` | NEW full benchmark config | ~50 |
| `molmetal/tests/test_plan_a_step3.py` | NEW smoke + aggregation tests | ~80 |

**总计 ~730 LOC, 5 文件**

### §2.4 Paper figure (4 figures)

**Figure 1** (main): 4-panel
- (a) pIC50 distribution per (cell-line, metal) — heatmap
- (b) SA vs QED scatter, colored by pIC50 — quality landscape
- (c) Synthesis path length distribution per (cell-line, metal) — complexity
- (d) S2T curve (cumulative fraction vs sims) — efficiency

**Figure 2** (ablation): 6 rows × 4 metric table — per §2.2 row definitions

**Figure 3** (case study): top-3 candidates per cell-line — structures + paths + metrics

**Figure 4** (analysis): prior entropy vs pIC50 correlation, value network calibration curve

## §3 实施顺序 (gated on Step 1+2 + dmpnn_rh fix)

1. **Step 1 完工 + MEASURED delta 出表**
2. **Step 2 完工 + MEASURED delta 出表**
3. **dmpnn_r10 Rh widen 完工** (commit per dmpnn_r10_fix_plan.md)
4. **rh_lomo 完工** (commit per rh_lomo_fix_plan.md)
5. **(2 day)** 写 closed-loop benchmark pipeline + 配置文件
6. **(45 hr GPU)** 跑 6-row ablation × 450 records each
7. **(1 day)** 聚合 ablation table,生成 Figure 1-4
8. **(1 day)** write Tier-2 paper Section 4 + Section 5 (per §5.3 NO paper work without authorization — this is **planning only**,actual paper 启动 等 user OK)

**Total: ~2-4 wk wall + ~45 hr GPU**

## §4 Step 3 完成判据

✅ All of:
1. 450-record closed-loop benchmark JSON on disk (`molmetal/reports/plan_a_step3/main.json`)
2. 6-row ablation table on disk (`molmetal/reports/plan_a_step3/ablation.json`)
3. Figure 1-4 PNG/PDF on disk (`molmetal/reports/plan_a_step3/figures/`)
4. **main result 行 (Row 4) top_1_pIC50 mean > Row 1 baseline mean** (统计显著,paired t-test)
5. Tier-2 paper Section 4+5 draft 准备好 (但不写 paper/* 等 user OK)
6. Commit landed

## §5 MEASURED delta 表 (待 Step 3 跑完填)

| Row | Prior | Value | top_1_pIC50 (mean ± std) | SA | QED | Validity | S2T |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | Symbolic | hand-crafted | TBD | TBD | TBD | TBD | TBD |
| 2 | Symbolic | DMPNN pIC50 | TBD | TBD | TBD | TBD | TBD |
| 3 | CFM learned | hand-crafted | TBD | TBD | TBD | TBD | TBD |
| 4 | CFM learned | DMPNN pIC50 | TBD | TBD | TBD | TBD | TBD |
| 5 | CFM learned | DMPNN pIC50 | TBD | TBD | TBD | TBD | TBD |
| 6 | random | DMPNN pIC50 | TBD | TBD | TBD | TBD | TBD |

## §6 风险 + 缓解

⚠️ **GPU 时间 ~45 hr** 比较大 — 但 Blackwell 90GB + 5090 32GB 都空,两卡并行 ~25 hr
   - 缓解:Step 3 启动前 verify 两卡空闲 + §5.1 standing GPU auth

⚠️ **450 records × 6 rows = 2700 records**,数据量大
   - 缓解:output 用 parquet 不是 JSON (size ~10× smaller)

⚠️ **paper figure 生成可能 issue** (matplotlib backend / 中文字体)
   - 缓解:用 Agg backend + DejaVu Sans 字体 (per mol-metal-paper-writing skill §字体)

⚠️ **paper 启动是 user-OK gated per §5.3**
   - 缓解:Step 3 完成后,我只交付 paper Section 4+5 **plan draft**,actual paper work 等 user 拍板

⚠️ **baseline (Row 1) 可能也强** — hand-crafted SymbolicPrior + Lipinski 已经能 find 合理路径
   - 缓解:Row 6 (random prior) 作 sanity check,如果 Row 1 vs Row 6 lift 不显著,ablation 解释力不够

## §7 跨引用

- `plan_a_overview.md` §2.2 + §3 (Tier-2 vs 一区 framing)
- `plan_a_step1_dmpnn_value.md` (Step 1 必须先)
- `plan_a_step2_cfm_prior.md` (Step 2 必须先)
- `plan_a_data_pipeline.md` (Step 3 用 9 cell-line 数据来源)
- TODO/pending/dmpnn_r10_fix_plan.md → completed/ (PRECURSOR)
- TODO/pending/rh_lomo_fix_plan.md → completed/ (PRECURSOR)
- AGENTS.md §3.2 `mol-metal-anticancer` skill (pIC50 / MetalCytoToxDB / A2780 / MCF7)
- AGENTS.md §5.1 (standing GPU auth — 45 hr 在范围内)
- AGENTS.md §5.3 (paper work gated)
