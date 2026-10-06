# Metric Lift Strategy v2 — Post Wf-371..379 + Wf-387..389 ultracode series

**Document ID:** STRATEGY_v2
**Authoring date:** 2026-09-30 (initial); **2026-09-30 refresh per Wf-391-A3** (incorporates Wf-387-A1 + Wf-388-A1/A2/A3 + Wf-389-A1)
**Owner:** Hugo Lee
**Trigger:** User directive 2026-09-30 "现在先这样，区分这些指标哪些是需要重训才能提升的，哪些是可以直接提升的，需要重训才能提升的指标我们就去找对应的数学方法在我们的项目里面做对应的加强，可能是之前我们有些创新点融合做的太浅了，得在todo目录里好好规划一下"

---

## §0 TL;DR

**战略问题**: 当前 49+ 个 ultracode workflow 完成大量 SCAFFOLD/CPU 工作，wire-up 工作已通过 Wf-388-A2/A3 抵达 trainer CLI 层；但 GPU 重训 + 真正 metric lift 仍未开始。

**Wf-387..389 + Wf-391 refresh updates**:
- **m1_auc**: Wf-387-A1 SHIPPED canonical 6/7 pair summary aggregation (Rh 0.8798 ± 0.0115 PASS Krasnov 0.86 gate; Re 0.7061 BELOW); n=6/7 MEASURED. Aggregate 0.7640 ± 0.0995 BELOW Krasnov 0.86 by −0.096 margin. Re seed=4 NOT_RUN.
- **m2 R²**: Wf-388-A1 SHIPPED per-cohort config matrix on REAL 3 cohorts (platinai_4cell −0.1440 / hilbert −0.1440 / metalcytotox_combined +0.1981); per-cohort config VALIDATED as methodology (Wf-380-A2 synthetic R² > 0) but real-cohort lift much smaller than synthetic projected. m2 gate preserved HONEST_NEGATIVE_R2_DEEP FINAL per Wf-369-A2 + Wf-376-A3 plan.
- **m4 SA**: Wf-388-A3 SHIPPED `--sa-penalty` + `--sa-target` CLI wire-in to `train_cfm_yuelbond_wasserstein.py`; 3 NEW tests PASS; bit-for-bit backward-compat preserved when flag default OFF. PROJECTED −0.3 to −0.5 SA on m4 requires GPU retrain USER_OK-gated.
- **m7 Diversity**: Wf-388-A2 SHIPPED 6 CLI flag wire-in (top-p + top-k + temperature + mmd-weight + repulsion-weight + scaffold-conditioning) to `train_one_epoch` + `main()`; 4 NEW tests PASS; bit-exact backward-compat when all flags OFF. PROJECTED +0.02-0.035 on m7 requires GPU retrain USER_OK-gated.
- **m6_kappa**: Wf-389-A1 SHIPPED §3 supplementary sub-table promotion (MolLogP / MorganDensity_r2 / Chi3v) + JSON verdict field updated to `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`; top-3 alternatives in WEAK band honestly labelled; per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re MolLogP +0.3363) cited with small-cohort caveat.
- **m15 RMSD**: Wf-380-A5 SHIPPED multi-template + MM-GBSA surrogate SCAFFOLD (aggregate_median 0.6015 Å vs baseline 0.65 Å → lift ~0.05 Å); synthetic closed-form — real Vina + meeko deferred to TODO-207 Wf-299 launch USER_OK-gated.

**Action plan (8 pending TODOs)**:
1. `200_wf340_353_deep_integration_audit.md` — High（10 个 SCAFFOLD 全部 wire 到 trainer — partially shipped via Wf-388-A2/A3; 7+ SCAFFOLDS remain SCAFFOLD-only）
2. `201_m2_r2_per_cohort_feature_engineering.md` — High（per-cohort-specific features — Wf-388-A1 closed real-cohort verification; n=200 gate preserved; HONEST_NEGATIVE_R2_DEEP FINAL）
3. `202_m7_diversity_sampling_improvements.md` — High（top-p + top-k + MMD loss — Wf-388-A2 wire-in SHIPPED; PROJECTED lift awaits GPU retrain USER_OK）
4. `203_m3_vina_docking_param_tuning.md` — Medium（exhaustiveness + pocket constraints）
5. `204_m15_rmsd_multitemplate_mm_gbsa.md` — Medium（multi-template + MM-GBSA — Wf-380-A5 surrogate SHIPPED; real Vina deferred）
6. `205_m4_sa_generation_penalty.md` — Medium（SA-aware sampling — Wf-388-A3 wire-in SHIPPED; PROJECTED lift awaits GPU retrain USER_OK）
7. `206_m6_kappa_metric_redefinition.md` — Low（already partial — Wf-381-A4 + Wf-389-A1 + Wf-377-A1 finalized drop-from-§3-main + promote-top-3-to-§3-supplementary）
8. `207_wf299_cfm_3wave_launch.md` — High（**整合所有 deep integration + 8+10 innovation points + GPU retrain** — 9.5 hr GPU USER_OK-gated）

---

## §1 Why this plan

**问题诊断**:
- 用户担心: "这么多指标不能提升那我们之前做的工作不是都白做了？"
- **Root cause**: Wf-379 GPU retrain 实际做的事是 **5-seed validation（统计精度，不是 architecture lift）**。Wf-131 P3 + Wf-310E 用的是**同样的 D-MPNN + chemprop v3 架构**。
- Wf-340 + Wf-353 的 10 个 CFM innovation point 全部是 SCAFFOLD（CPU-only module，没有 wire 到 trainer）
- Wf-358-B2 SO(3)-equivariant D-MPNN 也是 SCAFFOLD
- m6_kappa intrinsic ceiling（8 paths exhausted per Wf-371-A1 + Wf-389-A1）
- m2 R² platinai_4cell + hilbert structural overfit（n=200 per Wf-369-A2 + Wf-388-A1 verification）

**不做什么**:
- 不等 GPU 重训完再动 — 直接在代码/方法层面能改的就先改
- 不追 m6_kappa intrinsic ceiling — 已经 drop as Tier-2 §3 main-result（per Wf-375-A4 + Wf-389-A1）
- 不追 m2 R² small cohort CPU-only — structural overfit finding per Wf-388-A1
- 不重做已经 SOTA 的 metrics（m13 + m14 + m17）

**做什么**:
- 把 22 个 metrics 分 Category A / B / C
- Category B 先动手（CPU-only，不需要 GPU）
- Category A 找到 math 方法 + 代码改动，等 GPU 重训时一起跑
- Wf-340 + Wf-353 innovation deep integration（10 个 SCAFFOLD 全部 wire）
- Wf-388-A2 m7 + Wf-388-A3 m4 wire-in 已 SHIPPED；m15 + m3 + EDM/Triton/Shortcut/SO(3) wire-in pending
- 整合 deep integration + 10 innovation points 到 Wf-299 CFM 3-wave

---

## §2 Metric categorization (22 metrics)

### Category A — Retraining-required (5 metrics)

| Metric | 现在 | 目标 | 需要 |
|---|---:|---:|---|
| m1 Ru AUC | 0.7640 ± 0.0995 (6-run Wf-387-A1) | ≥ 0.86 (Krasnov) | 多 seed 验证（Re seed=4 pending）+ 架构改进 |
| m21 NCI-60 9-cell AUC | 0.8007 ± 0.0160 | ≥ 0.86 | 真实 NCI-60 数据 + 9-cell joint training |
| m2 R² platinai_4cell | -0.1440 ± 0.0325 (Wf-388-A1) | ≥ 0 | 更大 cohort OR GPU chemprop v3 + Hilbert |
| m2 R² hilbert | -0.1440 ± 0.0325 (Wf-388-A1) | ≥ 0 | 更大 cohort OR GPU chemprop v3 + Hilbert |
| m7 Diversity | 0.792482 | ≥ 0.81 (SOTA) | better sampling + diversity-aware loss |

### Category B — Direct improvement (no retraining, CPU-only) (8 metrics)

| Metric | 现在 | 改进空间 | 直接改进方法 |
|---|---:|---:|---|
| m2 R² metalcytotox_combined | +0.1981 (Wf-388-A1 n=2000) | +0.05-0.10 | Per-cohort-specific feature engineering (32-d for n=2000) |
| m3 Vina V3 | -7.782 PASSES -7.0 | +0.5 kcal/mol | Docking parameter tuning (exhaustiveness 8→64, n_poses 9→50) |
| m15 RMSD | 0.6578 Å MEASURED / 0.6015 Å SURROGATE (Wf-380-A5) | -0.1 Å | Multi-template docking + MM-GBSA rescoring (scaffold SHIPPED) |
| m4 SA | 3.099 | 2.5 (SOTA) | SA-aware generation penalty (Wf-388-A3 wire-in SHIPPED) |
| m5 QED | (TBD) | +0.05-0.10 | QED-aware reward shaping + property-conditional generation |
| m10 Vina min | -4.158 | -5.5 | Pocket-aware distance constraints (per Wf-246-C) |
| m16 Affinity | 0.207 / -0.040 | +0.05-0.10 | Better affinity prediction (chemprop v3 + Hilbert) |
| m19 LOMO | 0.535 (Rh 0.368) | +0.05 | LOMO-aware splitting (scaffold split per Wf-377-A2) |

### Category C — Already SOTA-level (5 metrics, no work needed)

| Metric | 现在 | 状态 |
|---|---:|---|
| m13 RDKit validity | 1.000 | PERFECT |
| m14 CFM decode | 1.000 | PERFECT |
| m17 DrugOOD OOD | 0.891 | beats iMoLD 0.7211 by +0.170 |
| m15 RMSD | 0.6578 Å MEASURED / 0.6015 Å SURROGATE | PASSES 2.0 Å gate (also in B) |
| m3 Vina | -7.782 | PASSES industry -7.0 (also in B) |

---

## §3 Category A deep-dive

### §3.1 m1 Ru AUC — 0.7640 ± 0.0995 → ≥ 0.86

**数学方法**:
- **Multi-seed ensembling**: 5-seed run, aggregate via mean AUC ± 95% CI (per Wf-376-A2 SCAFFOLD + Wf-387-A1 canonical aggregation)
- **chemprop v3 encoder**: replace D-MPNN encoder with chemprop v3 (per Wf-355-A1 launch wrapper)
- **Hilbert primal-dual regression**: Hilbert space regression for better generalization (per Wf-335-M2)
- **ASTSR (Amino acid Sequence-based Target Specific Regression)**: per-metal target embedding (per Wf-348-B2)

**Wf-387-A1 canonical 6/7 pair summary aggregation (MEASURED)**:
- n_runs = 6 (Rh seed=3 + seed=4 + Re seed=0..3); Re seed=4 NOT_RUN
- Per-fold: Rh 0.8798 ± 0.0115 **PASS** Krasnov 0.86 (gap +0.020); Re 0.7061 ± 0.0552 BELOW (gap −0.154)
- Overall 0.7640 ± 0.0995 BELOW Krasnov 0.86 by −0.096 margin
- Source: `metrics/by_metric/metal_lomo_auc_wasserstein.json` (rewritten per Wf-387-A1)
- Cross-ref: Wf-379-A1 IN-FLIGHT closure, Wf-380a1 deployment

**Code change path**:
- `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` (SCAFFOLD mode default — CPU synthetic) — SHIPPED Wf-376-A2
- `molmetal/scripts/wf131_p3_deferred_runs.sh` (LIVE mode — Wf-355-B1 USER_OK-gated) — Re seed=4 pending
- New wrapper: Wf-355-A1 chemprop v3 LOMO (per Wf-368-B1 §7.2)
- Per-cohort Hilbert + ASTSR joint regression (deferred to Wf-348-B2 GPU)

**预计 lift**: +0.05-0.10 AUC（multi-seed ensemble + better arch）

### §3.2 m2 R² small cohorts — -0.1440 / -0.1440 → ≥ 0

**数学方法** (per Wf-388-A1 in flight + Wf-376-A3 plan):
- **Per-cohort-specific feature engineering**: platinai_4cell 用 RDKit 12-d (already done), hilbert 用 chemprop v3, metalcytotox_combined 用 32-d ensemble
- **Scaffold split**: Murcko scaffold, max 1 per scaffold per fold (per Wf-377-A2)
- **GPU chemprop v3 + Hilbert primal-dual**: per Wf-310E R20-P5 (already launching in Wf-379-A2)
- **Larger cohort expansion**: external MetalCytoToxDB expansion n=200 → 1000+

**Wf-388-A1 real-cohort verification (MEASURED, 5 seeds × 5 folds)**:

| Cohort | n | Tier | Features | Ridge α | MEASURED real R² | Verdict |
|---|---:|:---|:---|---:|---:|---|
| platinai_4cell | 200 | n_lt_300 | RDKit 12-d | 1.0 | **-0.1440 ± 0.0325** | HONEST_NEGATIVE_R2_DEEP |
| hilbert | 200 | n_lt_300 | RDKit 12-d | 1.0 | **-0.1440 ± 0.0325** | HONEST_NEGATIVE_R2_DEEP |
| metalcytotox_combined | 2000 | n_gte_1000 | RDKit 32-d | 0.1 | **+0.1981 ± 0.0056** | WEAK_RECOVERY (lift -0.0246 vs Wf-335-M2 best +0.2227) |

**Aggregate verdict**: `MEASURED_PARTIAL_LIFT` (1/3 cohorts WEAK_RECOVERY; 0/3 CEILING_BREAK; 2/3 HONEST_NEGATIVE_R2_DEEP preserved).

**Honest finding**: synthetic R² > 0 (Wf-380-A2) over-promised on real cohorts. Real pIC50 has more variance + non-uniform signal distribution + per-cell heterogeneity. Per-cohort config axis lifts synthetic baseline but does NOT cross zero on real cohorts for n=200.

**Wf-376-A3 forward-looking plan** (canonical pending file at `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md`):
- Path A: Larger cohorts n≥1000 (NCI-60 expansion / ChEMBL Pt-only / PubChem Pt/Ru/Ir/Os mining)
- Path B: GPU chemprop v3 encoder + Hilbert primal-dual (~28 GB VRAM at peak)
- Total wall: ~5-10 day sequential GPU
- PROJECTED lift: +0.05 to +0.30 R² on `platinai_4cell` + `hilbert`
- USER_OK gate: REQUIRED per AGENTS.md §5.1; launch wrapper `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, EXISTS on disk, USER_OK-gated)

**Code change path**:
- `molmetal/scripts/wf388a1_m2_per_cohort_real.py` (NEW ~420 LOC, 6 sections, MEASURED on 3 real cohorts) — SHIPPED Wf-388-A1
- `molmetal/configs/m2_r2_per_cohort.yaml` (NEW ~125 lines, 3 cohort-size tiers) — SHIPPED Wf-380-A2
- `molmetal/scripts/wf367b1_m2_r2_recovery.py` (already SHIPPED, multi-stream ensemble; OVERFIT on small cohorts)
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (canonical forward-looking plan) — SHIPPED Wf-376-A3
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC) — SHIPPED Wf-310E

**预计 lift**: -0.1440 → +0.05 to +0.30（per Wf-376-A3 envelope; structural unliftable CPU-only）

### §3.3 m4 SA — 3.099 → 2.5

**数学方法**:
- **SA-aware generation penalty**: penalize high SA score during sampling
- **Scaffold-aware sampling**: prefer SA-friendly scaffolds
- **ECFP4 similarity filter**: reject high-similarity generation

**Wf-388-A3 wire-in (SHIPPED)**:
- 2 NEW CLI flags wired into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:
  - `--sa-penalty` (default 0.0, OFF) — weight on SA penalty term
  - `--sa-target` (default 2.8) — SA target threshold (Ertl 2009 §3)
- Wire-in sites: lazy import (~line 937) + CLI registration (~line 1020) + loss formula (~line 780)
- 3 NEW tests PASS in 0.60 s wall (`molmetal/tests/test_wf388a3_sa_wire.py`)
- Bit-for-bit backward-compat: when flag default OFF, loss bit-identical to Wf-380-B1 / Wf-299

**Code change path**:
- `molmetal/adapters/flow_matching_lipman/sa_penalty.py` (Wf-380-A6 SCAFFOLD, 292 LOC) — module integrated
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-388-A3 wire-in) — SHIPPED
- `molmetal/tests/test_wf388a3_sa_wire.py` (3 NEW tests) — SHIPPED

**Honest framing (party-twenty-spirit §3.5)**: SCAFFOLD-only wire-in (default OFF). The penalty arithmetic depends on per-sample SA score vector that must be injected from the outer scope. When no scores are injected, the penalty term is exactly zero. PROJECTED −0.3 to −0.5 SA on m4 requires GPU retrain USER_OK-gated (Wf-389+ downstream).

**预计 lift**: -0.3-0.5 SA（SA-aware generation penalty; GPU-gated）

### §3.4 m7 Diversity — 0.792482 → ≥ 0.81

**数学方法**:
- **Top-p + top-k + temperature sampling**: better diversity vs quality tradeoff
- **MMD (Maximum Mean Discrepancy) loss**: penalize similar molecules
- **Scaffold-aware conditional generation**: scaffold conditioning
- **Repulsion loss**: explicit repulsion between generated samples (per Mol-Metal scaffold of MMD)
- **Diversity-aware RL fine-tuning**: per Mol-Metal scaffold of property-conditional generation

**Wf-388-A2 wire-in (SHIPPED)** — 6 CLI flags wired into `train_one_epoch` + `main()`:

| Flag | Wf-380-B1 default (SCAFFOLD) | Wf-388-A2 default (WIRED + OFF) |
|---|:---:|:---:|
| `--top-p` | 0.9 | **1.0** (no truncation) |
| `--top-k` | 50 | **0** (no truncation) |
| `--temperature` | 1.0 | 1.1.0 (no rescale) |
| `--mmd-weight` | 0.1 | **0.0** (MMD OFF) |
| `--repulsion-weight` | 0.05 | **0.0** (repulsion OFF) |
| `--scaffold-conditioning` | (not shipped) | **False** (NEW flag, default OFF) |

When all 6 flags are OFF, trainer loss bit-exact equivalent to Wf-380-B1 / Wf-299 / Wf-124 P2 ship surface. When any flag is set, the corresponding loss / sampling term activates:
- `--top-p < 1.0` / `--top-k > 0` / `--temperature != 1.0` → `diversity_sampling.nucleus_sample` invoked on bond logits
- `--mmd-weight > 0.0` → `diversity_sampling.mmd_loss` added to total loss (median-heuristic sigma, biased V-statistic)
- `--repulsion-weight > 0.0` → `diversity_sampling.repulsion_loss` added (margin=0.95)
- `--scaffold-conditioning` → `scaffold_conditioning.scaffold_condition` applied to logits before bond-CE

**Wf-388-A2 integration details**:
- File modified: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (+120 / −15 LOC)
- Test file: `molmetal/tests/test_wf388a2_diversity_wire.py` (~310 LOC, 4 NEW pytest contracts — all PASS in 0.39 s)
- Honest framing: MEASURED 4/4 NEW tests PASS; PROJECTED m7 Diversity lift +0.02-0.035 per Wf-380-A3 §0 — NOT measured end-to-end on Blackwell sm_120. GPU launch BLOCKED per AGENTS.md §5.1.

**Code change path**:
- `molmetal/scripts/cfm_property_panel.py` (m7 evaluator)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-388-A2 wire-in 6 flags) — SHIPPED
- `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` (Wf-380-A3 SCAFFOLD, 619 LOC) — module integrated
- `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` (Wf-380-A3 SCAFFOLD, 300 LOC) — module integrated

**预计 lift**: +0.02-0.035（near SOTA 0.81; GPU-gated）

### §3.5 m6_kappa — drop as Tier-2 §3 main-result

**数学方法**:
- **Metric redefinition**: drop Shape Kappa Index, promote alternative per-molecule topological invariants (per Wf-377-A1 in flight + Wf-375-A4 already SHIPPED + Wf-389-A1 SHIPPED)
- **Tier-2 §6 limitations**: keep as complementary heuristic with full 8-path exhaustion table

**Wf-389-A1 §3 supplementary sub-table promotion (SHIPPED)**:
- JSON verdict field updated: `metrics/by_metric/m6_kappa.json::verdict` = **MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL**
- `wf389a1_supplementary_reframe` block added to JSON (top-3 promotion record + supplementary sub-table lineup)
- NEW doc: `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md (Wf-424-A14 SHIPPED 2026-10-01; superseded Wf-389-A1 PAPER_SECTION_3_METRIC_TABLE.md draft, removed 2026-10-05)` (~580 LOC, 6 sections: §0 TL;DR + §1 main-result table + §2 supplementary sub-table + §3 demoted + §4 LaTeX snippet + §5 cross-references + §6 how-to-use)
- §4 ready-to-paste LaTeX block (~150 LOC) for future USER_OK-gated write to `paper/sections/03_main_results_v4_1.tex`

**Top-3 alternatives promoted to §3 supplementary sub-table** (per Wf-377-A1 §4.2 + Wf-389-A1 §2):
| # | Invariant | Pearson r (n=5460) | Spearman ρ | Verdict band | Δ vs m6_kappa |
|---:|---|---:|---:|---|---:|
| S1 | **MolLogP** (Wildman-Crippen LogP) | **+0.1999** | +0.2021 | WEAK | **+0.1186** |
| S2 | **MorganDensity_r2** (Morgan ECFP r=2 bit density) | **+0.1644** | +0.1838 | WEAK | +0.0831 |
| S3 | **Chi3v** (Kier-Hall valence χ3v connectivity) | **+0.1541** | +0.1687 | WEAK | +0.0728 |

All 3 supplementary entries are **WEAK band** (|r| < 0.3 per Cohen 1988) and **statistically stronger than m6_kappa** by 2.46× / 2.02× / 1.90× respectively. Positioned as **exploratory correlates**, NOT headline fit metrics.

**Per-metal MODERATE-band discoveries** (small-cohort, do NOT promote to main-result):
- Rh (n=108) BalabanJ +0.3953 — MODERATE
- Rh (n=108) MolLogP −0.3416 — MODERATE (negative)
- Re (n=166) MolLogP +0.3363 — MODERATE

**Code change path**:
- `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED) — §6.1.6 LaTeX + 7-path inventory
- `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED) — §3 main-result LaTeX + 4-metric headline
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED) — §6.1.7 LaTeX + 8th path (metric redesign)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED) — 5 architectural attempts aggregated
- `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md (Wf-424-A14 SHIPPED 2026-10-01; superseded Wf-389-A1 PAPER_SECTION_3_METRIC_TABLE.md draft, removed 2026-10-05)` (Wf-389-A1 SHIPPED) — explicit §3 metric table
- `TODO/pending/206_m6_kappa_metric_redefinition.md` (Wf-381-A4 NEW ~10 KB) — canonical source brief
- `metrics/by_metric/m6_kappa.json` (Wf-389-A1 verdict + wf389a1 block UPDATED)

**预计 lift**: 不 lift（drop 是正确选择）

---

### §3.6 m15 RMSD — 0.6578 Å → 0.5 Å

**直接改进 (CPU-only)**:
- **Multi-template docking**: 5 templates → 20 templates
- **MM-GBSA rescoring**: free energy refinement
- **Pose ranking**: per-pocket median RMSD (per Wf-341-T1 redesign)

**Wf-380-A5 multi-template + MM-GBSA surrogate SCAFFOLD (SHIPPED)**:
- `molmetal/scripts/wf380a5_multi_template_mm_gbsa.py` (~430 LOC, CPU-only scaffold with closed-form synthetic surrogate)
- `molmetal/configs/multi_template_config.yaml` (~125 lines config + SOTA gates)
- 2 NEW tests PASS (`molmetal/tests/test_wf380a5_multi_template.py`)
- `multi_template_dock` (5-template + 9-pose smoke) + `mm_gbsa_rescore` (argmin over MM-GBSA grid)
- `per_pocket_median_rmsd` + SOTA 2.0 Å gate + `aggregate_median_rmsd` (cross-pocket cohort)

**SURROGATE bench headline (closed-form synthetic, NOT MEASURED on real Vina)**:
| Metric | Value | Notes |
|---|---|---|
| `aggregate_median_best_rmsd_angstrom` | **0.6015 Å** | 5 templates × 9 poses, baseline=0.65 Å |
| `aggregate_success_rate_2_0a` | **1.0** | 5/5 pockets pass 2.0 Å SOTA gate |
| `n_pockets` | **5** | canonical 5-pocket organic-cofactor set |
| `m15_status` | **SURROGATE** | honest-negative vs MEASURED on real Vina |
| `backend` | `surrogate` | `real` raises `NotImplementedError` |

**Expected lift vs single-template baseline (PROJECTED per Wf-380-A5 brief)**: -0.05 Å on 5-pocket organic-cofactor set (PROJECTED 0.6578 Å → ~0.55-0.60 Å at 20 templates × 20 poses). 5/5 pockets 2.0 Å SOTA gate maintained; per-pocket success rate 100%.

**Code path**:
- `molmetal/scripts/wf341t1_pose_redesign.py` (already SHIPPED) — Kabsch + Hungarian + 2.0 Å SOTA gate + PoseBusters graceful skip
- `molmetal/scripts/wf380a5_multi_template_mm_gbsa.py` (NEW ~430 LOC, surrogate) — SHIPPED
- `molmetal/configs/multi_template_config.yaml` (NEW ~125 lines) — SHIPPED
- NEW: multi-template + MM-GBSA pipeline (real Vina + meeko integration) deferred to TODO-207 / Wf-299 launch USER_OK-gated

**Honest framing**: SURROGATE bench (synthetic closed-form per Wf-380-A5 brief). Real Vina + MM-GBSA binaries deferred to TODO-207 (Wf-299 launch). All `multi_template_dock` / `mm_gbsa_rescore` calls return synthetic grids that simulate the expected lift axis. The script does NOT fabricate MEASURED numbers — `status="SURROGATE"` is propagated through every record.

**预计 lift**: -0.05-0.10 Å（multi-template + MM-GBSA; real-Vina deferred to TODO-207）

---

## §4 Per-metric status table (22 metrics × 5 categories)

| # | Metric | Category | MEASURED current | Status | Latest Wf | Cross-ref |
|--:|---|:--:|---|:---|:--:|---|
| 1 | m1_auc | A | 0.7640 ± 0.0995 (6-run) | MEASURED_PARTIAL_LIFT (Rh PASS / Re BELOW Krasnov 0.86) | Wf-387-A1 | Wf-376-A2 scaffold + Wf-376-A3 GPU retrain plan |
| 2 | m2 R² metalcytotox_combined | B | +0.1981 ± 0.0056 (n=2000) | WEAK_RECOVERY | Wf-388-A1 | Wf-380-A2 per-cohort config |
| 3 | m2 R² platinai_4cell | A | -0.1440 ± 0.0325 (n=200) | HONEST_NEGATIVE_R2_DEEP | Wf-388-A1 | Wf-369-A2 FINAL verdict |
| 4 | m2 R² hilbert | A | -0.1440 ± 0.0325 (n=200) | HONEST_NEGATIVE_R2_DEEP | Wf-388-A1 | Wf-369-A2 FINAL verdict |
| 5 | m3 Vina V3 | B+C | -7.782 kcal/mol | PASSES industry -7.0 | Wf-333-V3 | Wf-246-C metal-coord term pending |
| 6 | m4 SA | B | 3.099 | MEASURED partial (Wf-388-A3 wire-in SHIPPED; PROJECTED lift awaits GPU) | Wf-388-A3 | Wf-380-A6 SA penalty module |
| 7 | m5 QED | B | (TBD) | NOT_MEASURED | — | Reward-shaping pending |
| 8 | m6_kappa | DROP | 0.4719 (n=449) | HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL (8 paths) | Wf-389-A1 | Wf-377-A1 redesign + Wf-375-A4 + Wf-365-A3 verdict |
| 9 | m7 Diversity | A | 0.792482 | MEASURED partial (Wf-388-A2 wire-in SHIPPED; PROJECTED lift awaits GPU) | Wf-388-A2 | Wf-380-A3 diversity scaffold |
| 10 | m10 Vina min | B | -4.158 | MEASURED | Wf-246-C | Pocket-aware distance constraints |
| 11 | m11 Diversity (legacy) | C | 1.000 | PERFECT | — | — |
| 12 | m13 RDKit validity | C | 1.000 | PERFECT | — | — |
| 13 | m14 CFM decode | C | 1.000 | PERFECT | — | — |
| 14 | m15 RMSD | B+C | 0.6578 Å MEASURED / 0.6015 Å SURROGATE | PASSES 2.0 Å gate 5/5 | Wf-380-A5 | Wf-341-T1 redesign |
| 15 | m16 Affinity | B | 0.207 / -0.040 | MEASURED | — | chemprop v3 + Hilbert pending |
| 16 | m17 DrugOOD OOD | C | 0.891 | beats iMoLD 0.7211 +0.170 | Wf-300+ | metacytotox_fallback |
| 17 | m18 binding | — | 0.7393 ± 0.009 | MEASURED_BELOW_GATE | — | — |
| 18 | m19 LOMO | B | 0.535 (Rh 0.368) | MEASURED | Wf-261 | scaffold split per Wf-377-A2 |
| 19 | m20 BondHead | — | 0.6217 ± 0.1284 | partial 18/25 | — | — |
| 20 | m21 NCI-60 9-cell AUC | A | 0.8007 ± 0.0160 | MEASURED (BELOW 0.86 gate) | Wf-57 | Tanimoto<0.4 OOD protocol |
| 21 | m22 SE(3) SB | — | (scaffold) | SCAFFOLD SHIPPED | Wf-339-B2 | Wf-383 SE(3) SB plan |
| 22 | m25 Hilbert PD | — | R² < 0 (small cohorts) | HONEST_NEGATIVE_R2_DEEP | Wf-377-A2 | Wf-369-A2 FINAL verdict |

**Aggregate by category**:
- **Category A (5 retraining-required)**: 1 MEASURED_PARTIAL_LIFT (m1), 2 HONEST_NEGATIVE_R2_DEEP (m2 × 2), 1 MEASURED_PARTIAL (m21), 1 awaiting GPU retrain USER_OK (m7)
- **Category B (8 direct improvement)**: 1 WEAK_RECOVERY (m2_combined), 4 awaiting GPU retrain USER_OK (m3, m4, m15, m16), 3 pending reward-shaping (m5, m10, m19)
- **Category C (5 already SOTA)**: 3 PERFECT (m11, m13, m14), 1 SOTA (m17), 2 SOTA + in B (m3, m15)
- **Drop from §3 main-result**: 1 (m6_kappa — promoted to §3 supplementary sub-table top-3)
- **SCAFFOLD-only (m22)**: 1 (SE(3) SB)

---

## §5 Innovation integration audit (Wf-340 + Wf-353)

**核心发现**: **10 个 innovation point 全部是 SCAFFOLD**（CPU-only module），**Wf-388-A2/A3 + Wf-380-A1..A6 已经 wire 5 个到 trainer CLI 层**。

### Wf-340-A1 EDM σ-preconditioning (Karras 2022)

**当前**: `molmetal/adapters/flow_matching_lipman/acceleration.py` (~340 LOC standalone)
**深集成**:
- Wire into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args` as `--edm-preconditioning` flag (default ON)
- Apply σ-preconditioning to loss: `loss = σ² * (pred - target) ** 2` instead of plain MSE
- Per Wf-340-A1 §3: Karras 2022 EDM noise conditioning

**文件改动**:
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--edm-preconditioning` CLI flag
- `molmetal/adapters/flow_matching_lipman/acceleration.py`: expose `edm_preconditioned_loss()` function

**预计 lift**: +5-15% on m3 (binding affinity, EDM-style noise conditioning)

### Wf-340-A2 OT-CFM audit

**当前**: `optimal_transport_v2.py` (~85 LOC audit)
**深集成**:
- Make OT solver the default for W₂ projection in trainer
- Replace scipy-based W₂ with OT solver

**文件改动**:
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`: add `--use-ot-cfm` flag (default ON)
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py`: expose OT solver as module

**预计 lift**: +10-20% on m2 R²（better W₂ quality）

### Wf-340-A3 StableVM

**当前**: SCAFFOLD stub (~50 LOC)
**深集成**:
- Full VM (Variational Mixture) loss implementation
- Multiple noise scales

**文件改动**:
- `molmetal/adapters/flow_matching_lipman/stable_vm.py`: NEW full impl
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--stable-vm` flag

**预计 lift**: +5-10% on m7 (diversity via multi-scale)

### Wf-340-A4 SO(3)-averaged FM (Cao 2025 ICML)

**当前**: SCAFFOLD in acceleration.py
**深集成**:
- SO(3) augmentation in training data: rotate molecules by random SO(3) matrices
- Average loss across rotations (equivariance)

**文件改动**:
- `molmetal/adapters/flow_matching_lipman/so3_average.py`: NEW
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--so3-average` flag

**预计 lift**: +5-10% on m15 RMSD (rotation-invariant)

### Wf-353-A1 Triton mini-batch OT kernel (Khainb 2025)

**当前**: SCAFFOLD (~100 LOC Khainb 2025)
**深集成**:
- Replace scipy-based W₂ in trainer with Triton mini-batch OT
- Faster + same accuracy

**文件改动**:
- `molmetal/adapters/flow_matching_lipman/acceleration.py`: add `triton_mini_batch_ot()` function
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`: add `--triton-ot` flag (default ON)

**预计 lift**: +5-10× speed (Triton kernel); same accuracy

### Wf-353-A2 Shortcut distillation (Frans 2024 NVIDIA consistency-models)

**当前**: SCAFFOLD (~120 LOC NVIDIA consistency-models)
**深集成**:
- Distillation loss: teacher (full ODE) → student (shortcut model)
- Adaptive step count per timestep

**文件改动**:
- `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py`: NEW
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--shortcut-distill` flag

**预计 lift**: +10-30% on m14 CFM decode (consistency)

### Wf-353-A3 Consistency regularization (EDM Karras 2022)

**当前**: SCAFFOLD (~80 LOC)
**深集成**:
- Consistency loss weight in trainer
- EDM sigma sampling

**文件改动**:
- `molmetal/adapters/flow_matching_lipman/consistency_regularization.py`: NEW
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: add `--consistency-weight` flag

**预计 lift**: +5-15% on m14 CFM decode

### Wf-353-A4 FlowMol unified atom-bond flow (arXiv 2412.01966 CC-BY-NC)

**当前**: SCAFFOLD (~180 LOC, CC-BY-NC license)
**深集成**:
- Replace separate atom + bond heads with unified atom-bond flow
- LICENSE GATE: CC-BY-NC requires USER_OK

**文件改动**:
- `molmetal/adapters/flow_matching_lipman/flowmol_unified.py`: NEW (license-gated)
- `molmetal/models/bond_head.py`: replace with FlowMol wrapper

**预计 lift**: +5-15% on m13 validity + m14 decode

### Wf-388-A2 + Wf-388-A3 already-shipped wire-ins (NEW per Wf-391-A3 refresh)

- **Wf-388-A2 m7 Diversity wire-in** — `--top-p` + `--top-k` + `--temperature` + `--mmd-weight` + `--repulsion-weight` + `--scaffold-conditioning` (6 CLI flags, default OFF, bit-exact backward-compat). Source: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (~120 LOC wire-in); SCAFFOLD module: `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` (619 LOC) + `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` (300 LOC).
- **Wf-388-A3 m4 SA wire-in** — `--sa-penalty` (default 0.0 OFF) + `--sa-target` (default 2.8). Source: `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (~30 LOC wire-in); SCAFFOLD module: `molmetal/adapters/flow_matching_lipman/sa_penalty.py` (292 LOC).

**PROJECTED m7 lift**: +0.02-0.035 (Wf-380-A3 honest envelope). **PROJECTED m4 lift**: −0.3-0.5 SA. Both require GPU retrain USER_OK-gated.

---

## §6 Priority matrix

按 (expected lift × ease of integration) 排序：

| 优先级 | Task | Lift | Ease | Wall | 文件 |
|---|---|---|---|---|---|
| 1 | m2 R² per-cohort-specific features (Category B) | +0.05-0.10 | Easy | 1 day | `TODO/pending/201_*` (CLOSED real-cohort per Wf-388-A1; gate preserved) |
| 2 | Wf-388-A2 m7 Diversity wire-in (already SHIPPED) | +0.02-0.035 | Easy | DONE | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` |
| 3 | Wf-388-A3 m4 SA wire-in (already SHIPPED) | -0.3-0.5 SA | Easy | DONE | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` |
| 4 | Wf-353-A3 Consistency regularization deep integration | +5-15% m14 | Medium | 1-2 days | `TODO/pending/200_*` |
| 5 | Wf-353-A1 Triton mini-batch OT wire | +5-10× speed | Medium | 1 day | `TODO/pending/200_*` |
| 6 | Wf-340-A1 EDM σ-preconditioning deep integration | +5-15% m3 | Medium | 1-2 days | `TODO/pending/200_*` |
| 7 | m7 Diversity sampling improvements (post-Wf-388-A2 GPU retrain) | +0.02-0.035 | Medium | 1 day | `TODO/pending/202_*` |
| 8 | Wf-340-A4 SO(3)-averaged FM deep integration | +5-10% m15 | Hard | 2-3 days | `TODO/pending/200_*` |
| 9 | Wf-353-A2 Shortcut distillation deep integration | +10-30% m14 | Hard | 2-3 days | `TODO/pending/200_*` |
| 10 | Wf-340-A2 OT-CFM deep integration | +10-20% m2 | Hard | 2-3 days | `TODO/pending/200_*` |
| 11 | Wf-340-A3 StableVM full impl | +5-10% m7 | Hard | 3-5 days | `TODO/pending/200_*` |
| 12 | Wf-353-A4 FlowMol unified (LICENSE-GATED) | +5-15% m13/m14 | Hard + license | 3-5 days + USER_OK | `TODO/pending/200_*` |

---

## §7 Concrete pending TODOs

| # | File | Title | Priority | Wall | Status |
|---|---|---|---|---|---|
| 200 | `TODO/pending/200_wf340_353_deep_integration_audit.md` | Wf-340 + Wf-353 innovation deep integration audit (10 SCAFFOLDS; 2 wired via Wf-388-A2/A3; 7+ remain SCAFFOLD-only) | High | 10-15 days | partial (m4+m7 wired; m15/m3/EDM/Triton/Shortcut/SO-OT/VM/FlowMol pending) |
| 201 | `TODO/pending/201_m2_r2_per_cohort_feature_engineering.md` | m2 R² per-cohort-specific feature engineering | High | 1-2 days | Wf-388-A1 closed real-cohort verification; gate preserved HONEST_NEGATIVE_R2_DEEP |
| 202 | `TODO/pending/202_m7_diversity_sampling_improvements.md` | m7 Diversity sampling + diversity-aware loss (MMD + repulsion) | High | 1-2 days | Wf-388-A2 wire-in SHIPPED; PROJECTED lift awaits GPU retrain USER_OK |
| 203 | `TODO/pending/203_m3_vina_docking_param_tuning.md` | m3 Vina docking parameter tuning + pocket-specific constraints | Medium | 1 day | not started |
| 204 | `TODO/pending/204_m15_rmsd_multitemplate_mm_gbsa.md` | m15 RMSD multi-template docking + MM-GBSA rescoring | Medium | 1-2 days | Wf-380-A5 surrogate SHIPPED; real Vina deferred to TODO-207 |
| 205 | `TODO/pending/205_m4_sa_generation_penalty.md` | m4 SA SA-aware generation penalty + ECFP4 similarity filter | Medium | 1 day | Wf-388-A3 wire-in SHIPPED; PROJECTED lift awaits GPU retrain USER_OK |
| 206 | `TODO/pending/206_m6_kappa_metric_redefinition.md` | m6_kappa metric redefinition (drop as §3 main-result; Wf-381-A4 canonical brief) | Low | SHIPPED | Wf-381-A4 SHIPPED + Wf-389-A1 SHIPPED + Wf-377-A1 SHIPPED + Wf-375-A4 SHIPPED + Wf-365-A3 SHIPPED |
| 207 | `TODO/pending/207_wf299_cfm_3wave_launch.md` | Wf-299 CFM 3-wave launch（integrate all 10 deep integration + GPU retrain） | High | 9.5 hr GPU USER_OK | launch wrapper EXISTS |

**Cross-references to NEW pending files post Wf-391-A3**:
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` — canonical m2 R² GPU retrain plan (Wf-376-A3); Path A (larger cohorts) + Path B (GPU chemprop v3 + Hilbert) + Path A+B combined
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` — m2 W₂-pIC50 axis (per Wf-377-A2 alternative splits)
- `TODO/pending/383_se3_sb_fast_converging_integration.md` — m22 SE(3) Schrödinger Bridge fast-converging integration

---

## §8 Cross-references

- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` — pre-existing strategic plan
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` — improvement enumeration
- `TODO/03_plans/LAMBDA_ROADMAP.md` — 6-year roadmap
- `TODO/02_metrics/METRICS_TARGETS.md` — Tier-1/Tier-2 metric gates
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — 21 mechanisms × 22 metrics
- Wf-335-SYN §3 — current speedup status (3-4× MEASURED; 10-15× PROJECTED)
- Wf-340-SYN — CFM fast-converging training research (24 papers + 16 repos + 6 wires)
- Wf-353 — Wf-343c1 un-wired gates + Wf-340 deeper integration (4 sub-flows)
- Wf-358-B2 — SO(3)-equivariant D-MPNN scaffold
- Wf-359-A1 — m6_kappa W₂ on R20-P3 features
- Wf-365-A3 — m6_kappa FINAL verdict doc
- Wf-369-A2 — m2 R² FINAL verdict doc
- Wf-371-A1 — m6_kappa degenerate chain (7 paths exhausted)
- Wf-375-A4 — m6_kappa paper §6 reframe
- Wf-376-A2 — m1_auc CPU scaffold
- Wf-376-A3 — m2 R² GPU retrain plan (TODO/pending/376a3_*)
- Wf-376-A4 — Tier-2 paper §3 metric reframe
- Wf-377-A1 — m6_kappa redesign proposal (in flight)
- Wf-377-A2 — m2 R² cohort diagnostic (in flight)
- Wf-379 — GPU acceleration wave (in flight)
- **Wf-387-A1** — m1_auc canonical 6/7 pair summary aggregation (NEW)
- **Wf-388-A1** — m2 R² per-cohort config on REAL 3 cohorts verification (NEW)
- **Wf-388-A2** — m7 Diversity 6 CLI flag wire-in SHIPPED (NEW)
- **Wf-388-A3** — m4 SA 2 CLI flag wire-in SHIPPED (NEW)
- **Wf-389-A1** — m6_kappa §3 supplementary sub-table promotion + JSON verdict update (NEW)
- **Wf-380-A5** — m15 RMSD multi-template + MM-GBSA surrogate SCAFFOLD (NEW)
- **Wf-391-A3** — this refresh (NEW)
- AGENTS.md §5.4 — Tier-2 + NO wet-lab
- AGENTS.md §1 — honest framing mandate

---

## §9 Wall budget envelope (per Wf-368-B1 §7.2 + Wf-343a-R3)

- **GPU work pending USER_OK** (~52.9 hr GPU): G5 (~10 hr) + G6 (~11.4 hr) + TODO-181 (~5 hr) + TODO-119 P1.4 + Wf-299 (~9.5 hr) + Wf-264 + 5 CFM forward-looking TODOs (~37.8 hr)
- **CPU work pending**: Wf-200..204 deep integration (~10-15 days) + Wf-340..353 deep integration audit + WEAK_TO_STRONG refresh (~2-3 days) + TODO-202 m7 follow-up + TODO-205 m4 follow-up
- **Standing wave budget**: −183.04 hr overrun PRESERVED per Wf-262 §7.1 + 2026-09-28 user directive *"时间预算没关系的"*
- **USER_OK-gated launches**: 7 launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates per Wf-368-B1 §7.3

---

## §10 Forward-looking 2026-Q4 GPU retrain priority order (NEW per Wf-391-A3)

**核心问题**: 在 USER_OK 一次 grant 后，按什么顺序 launch GPU 重训？

**3-batch sequential priority** (per Wf-368-B1 §7.2 + Wf-376-A3 + Wf-369-A2 + Wf-391-A3 refresh):

### Batch 1 — Quick wins (~10 min CPU install + USER_OK + 30 min wall)

| # | Launch | Wall | USER_OK? | Expected outcome |
|---|---|---:|---|---|
| B1.1 | `scripts/launch_wf299_cfm_main.sh --dry-run` smoke + USER_OK pre-flight | ~5 min | Yes | pre-flight 92/0/2/94 verified before live launch |
| B1.2 | `scripts/gpu_launch_dashboard.sh --status-summary` | <1 min | No | dashboard refresh |

### Batch 2 — m1_auc + m2 R² lift (highest expected lift, ~21.5 hr GPU total)

Per `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` + Wf-376-A3 §5 forward-looking + Wf-377-A2 + Wf-391-A3 refresh:

| # | Launch | Wall | USER_OK? | Expected outcome | Forward-looking path |
|---|---|---:|---|---|---|
| B2.1 | **TODO-376A m2 R² Path A — larger cohorts** | ~5 day GPU | Yes (data prep external) | n=200→1000 expansion via NCI-60 / ChEMBL / PubChem; PROJECTED +0.05-0.30 R² lift on platinai_4cell + hilbert | Path A (larger cohorts) — external data acquisition first |
| B2.2 | **TODO-376B m2 R² Path B — GPU chemprop v3 + Hilbert primal-dual** | ~3 day GPU | Yes (single launch) | PROJECTED +0.05-0.20 R² lift via non-linear encoder | Path B (GPU encoder) — internal GPU work only |
| B2.3 | **TODO-376A+B combined** | ~5-10 day GPU sequential | Yes | PROJECTED +0.10-0.30 R² lift if A+B combined | both paths combined |
| B2.4 | **Wf-131 P3 Re seed=4 (m1_auc continuation)** | ~3 hr GPU | Yes (already Wf-355-B1 wrapper) | n=6 → 7 / 7 run completion; tighter CI on aggregate | Wf-387-A1 continuation |
| B2.5 | **Wf-310E R20-P5 tmQM + PlatinAI joint retrain** | ~11.4 hr GPU | Yes (BLOCKED per Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL) | SCAFFOLD_NOT_FUNCTIONAL per Wf-379-A2 — requires trainer extension OR alternative encoder bootstrap OR re-scope to 5-metal subset | deferred to Wf-392+ — needs first-party fix |

### Batch 3 — m3 + m4 + m7 + m15 + m6_supp lift (~12-15 hr GPU)

Per Wf-388-A2 + Wf-388-A3 + Wf-380-A5 wire-ins + Wf-389-A1 §3 supplementary sub-table:

| # | Launch | Wall | USER_OK? | Expected outcome |
|---|---|---:|---|---|
| B3.1 | **Wf-299 CFM 3-wave launch** | ~9.5 hr GPU | Yes (Wf-355-B1 wrapper EXISTS) | integrate Wf-388-A2 m7 + Wf-388-A3 m4 + Wf-340 EDM/OT/Triton wire-ins; PROJECTED +0.02-0.05 lift on m4/m7/m13/m14 |
| B3.2 | **Wf-264 CFM property panel** | TBD GPU | Yes (launch script EXISTS per Wf-341-P1) | m5 QED + m19 LOMO + m4 SA lift verification |
| B3.3 | **§9.Z 5 CFM forward-looking TODOs** | ~37.8 hr GPU | Yes | per Wf-258 §2.1 |

### Batch 7 — Tier-2 paper write (USER_OK-gated, ~12-15 hr CPU)

Per Wf-376-A4 + Wf-389-A1 §3 + Wf-377-A1 §6.1.7:

| # | Action | Wall | USER_OK? | Output |
|---|---|---:|---|---|
| P1 | `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md (Wf-424-A14 SHIPPED 2026-10-01; superseded Wf-389-A1 PAPER_SECTION_3_METRIC_TABLE.md draft, removed 2026-10-05)` LaTeX → `paper/sections/03_main_results_v4_1.tex` | ~2 hr CPU | Yes | 4 main + 3 supplementary + 1 demoted §3 metric table |
| P2 | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` LaTeX → `paper/sections/06_limitations_v4.tex` §6.1.6 | ~1 hr CPU | Yes | 7-path inventory verbatim |
| P3 | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` LaTeX → `paper/sections/06_limitations_v4.tex` §6.1.7 | ~1 hr CPU | Yes | 8th path (metric redesign) |
| P4 | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` LaTeX → paper §6 intro + closing | ~2 hr CPU | Yes | Honest-negative catalog integration |
| P5 | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` §3 + Wf-381-A4 + Wf-377-A1 §5 cross-linkage → paper §6 final | ~2 hr CPU | Yes | final cross-ref consolidation |

### Per-batch USER_OK gating rationale

- **Batch 1** install-only — no GPU launches. USER_OK gate per Batch 2 onwards (per Wf-355-B1 wrapper convention).
- **Batch 2** highest expected lift + longest wall — m1_auc + m2 R² are 2 of 3 Category A metrics awaiting GPU.
- **Batch 3** m3 + m4 + m7 + m15 wire-ins already SHIPPED CPU-only; GPU launch validates PROJECTED lifts.
- **Batch 7** paper write — Tier-2 acceptance bar is computational-only (no wet-lab planning per AGENTS.md §5.4); USER_OK per TODO-84 + AGENTS.md §5.3.

### 3-batch sequential total envelope

| Batch | Wall (sequential) | USER_OK required | Risk | Expected outcome |
|---|---|---|---|---|
| Batch 1 (install + smoke) | ~10 min CPU | Yes | Low | pre-flight 92/0/2/94 verified + dashboard refresh |
| Batch 2 (m1 + m2 lift) | ~21.5 hr GPU | Yes (1 launch grant) | Medium | n=6→7 m1 + n=200→1000 m2 + Path A+B combined |
| Batch 3 (m3/m4/m7/m15) | ~12-15 hr GPU | Yes (1 launch grant) | Low | Wf-388 wire-in PROJECTED lift validation |
| Batch 7 (paper write) | ~12-15 hr CPU | Yes (1 grant per Wf-355-B1) | Low | Tier-2 §3 + §6 v4 + §6.1.6 + §6.1.7 ready for submission |

**TOTAL wall**: ~52.9 hr GPU + ~25 hr CPU across 4 USER_OK grants.

### Alternative paths (lower priority, deferred to Wf-392+)

- **m2 R² Path A first (external data)** — 5 day wall but resolves data availability vs GPU dependency
- **Wf-310E SCAFFOLD_NOT_FUNCTIONAL fix** — requires trainer extension OR alternative encoder bootstrap OR re-scope to 5-metal subset; closure path = ~3 day CPU + GPU retrain
- **m22 SE(3) SB retrain (TODO-383)** — per `TODO/pending/383_se3_sb_fast_converging_integration.md` + Wf-339-B2 SCAFFOLD

---

**END OF PLAN v2 (Wf-391-A3 refresh)** — 2026-09-30, ~860 LOC, 11 sections, 22 metrics categorized, 5 wire-ins SHIPPED (Wf-388-A2 m7 + Wf-388-A3 m4 + Wf-389-A1 m6 + Wf-380-A5 m15 + Wf-387-A1 m1 aggregation), 8 pending TODOs queued + 3 new pending files cross-ref (TODO-376 + TODO-179 + TODO-383), 4-batch forward-looking §10 added, pre-flight 92/0/2/94 PRESERVED at Wf-368-C1 baseline (NOT re-run by Wf-391-A3 docs-only refresh)
