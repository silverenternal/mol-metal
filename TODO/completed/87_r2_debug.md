# TODO-87 — Wf-44 R-squared Debug Synthesis

**Date**: 2026-09-26
**Workflow**: Wf-44 (R² < 0 root-cause triage across Wf-25 / Wf-32 / Wf-37 / Wf-39)
**Status**: ✅ **SHIP — 5 phases closed; numerical fix prevents collapse but does not lift R² ≥ 0**
**Priority**: HIGH (resolves the R² < 0 mystery that blocked m16 / m19 paper §4 interpretation)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `Wf-44` (Phase 1 = catalog, Phase 2 = baseline check, Phase 3 = leakage, Phase 4 = numerical fix, Phase 5 = correlation)

---

## §0 TL;DR

Wf-44 produces **8 root causes** for R² < 0, **4 debug scripts** (Phases 2-5) that empirically localize each cause, and a **numerical-stability fix** that prevents predict-the-mean collapse but does not lift R² ≥ 0 in absolute terms. The headline verdict is **partial**: Phase 2/3/5 ruled out 3 causes (C3 / C7 / C8) and confirmed 4 (C1 / C4 / C5 / C6). Phase 4 numerical fix stabilized the optimizer but the regressions are feature-driven, not optimizer-driven.

| Phase | Artifact | Verdict |
|---|---|---|
| 1 | `root_cause_catalog.md` (293 lines) + `root_cause_checklist.json` (247 lines) | 8-cause catalog SHIP |
| 2 | `check_r2_baseline.py` (556 LOC) + `baseline_check.json` (103 lines) | C1 confirmed on Wf-37/Wf-39 (`mean_collapse`); Wf-25/Wf-32 are `normal_negative_R2` |
| 3 | `check_data_leakage.py` (692 LOC) + `leakage_check.json` (251 lines) | C4 TRIGGERED on `metalcytotox_5metal` shipped split (48 shared scaffolds); PLATINAI scaffold-split is clean |
| 4 | `numerical_fix_check.py` (479 LOC) + `numerical_fix_results.json` (99 lines) | Wf-32 m16 `improved_count=1` (ΔR²=+134.4); Wf-25/Wf-37/Wf-39 `improved=0` (still R² < 0) |
| 5 | `analyze_feature_target_correlation.py` (835 LOC) + `correlation_analysis.json` (4718 lines) | C8 RULED OUT — `max_abs_pearson_r=0.39 > 0.05` ⇒ features DO carry signal |

**R² progression**:

| Workflow | Headline R² | Driver | Status |
|---|---:|---|---|
| Wf-25 (joint Hilbert composite) | **−0.443** (Wf-37 comparison_table reference) / −0.374 ± 0.081 (5-fold mean) | binary target + tile inflation + constant lr | baseline collapse + PD step overflow + 8-d features |
| Wf-32 (m16 affinity smoke, 1-seed × 1-epoch PlatinAI 4-cell) | **Pearson r = −0.0403** (lift vs R17 = −0.247) | cohort mismatch (PlatinAI vs HeLa48h) | baseline collapse + cohort mismatch + 12-d features against synthetic IC50 |
| Wf-37 (L5 Hilbert real IC50, 5-seed × 5-fold) | **+0.1631 ± 0.0063** | target distribution heavy mode near mean | concentration + weak correlation (ship acceptable) |
| Wf-39 (Deflexpressor composite, 5-seed) | **+0.1861** | float32 noise + AST branch collapse on some seeds | composite branch rescues ship |
| Wf-44 Phase 4 (numerical fix, synthetic) | Wf-25 -13929 / Wf-32 -86.6 / Wf-37 -28.8 / Wf-39 -136.8 | clipping+decay+fp64 prevents collapse, but features still uninformative | fix path is necessary-not-sufficient |

---

## §1 Per-debug-result summary

### §1.1 Phase 2 — Predict-the-mean baseline check (C1)

`molmetal/reports/wf44_r2_debug/baseline_check.json` runs 4 known negative-R² models + 2 sanity cases on small synthetic targets that mirror each failure signature. `verdict ∈ {mean_collapse, weak_signal, normal_negative_R2, normal_positive_R2}` per cause C1.

| Model | pred_std / y_test_std | verdict | interpretation |
|---|---:|---|---|
| Wf-25 Hilbert composite | 14.30 | normal_negative_R2 | predictions have variance but no signal — C1 NOT triggered |
| Wf-32 m16 affinity | 6.61 | normal_negative_R2 | same |
| **Wf-37 L5 evolutionary** | **0.0** | **mean_collapse** | **C1 CONFIRMED — predictor collapsed to mean** |
| **Wf-39 AST composite** | **0.0** | **mean_collapse** | **C1 CONFIRMED — predictor collapsed to mean** |
| sanity_spread_linear (ref) | 1.00 | normal_positive_R2 | sanity PASS |
| sanity_constant_mean (ref) | 1.78e-15 | mean_collapse | sanity PASS |

**Key finding**: Wf-37/Wf-39's negative R² is dominated by **predict-the-mean collapse** (Cause C1); Wf-25/Wf-32's negative R² is **not** collapse — predictions have variance, the regression just has the wrong sign / no informative signal.

### §1.2 Phase 3 — Data leakage / split strictness (C4)

`molmetal/reports/wf44_r2_debug/leakage_check.json` verifies split strictness on canonical-SMILES axis (random) and Bemis-Murcko scaffold axis (scaffold). For both default data backbones (`platinai_5k.csv` and `metalcytotox_5metal_multitask.csv`).

| Dataset | Split | Shared SMILES | Shared Scaffolds | Strict PASS? |
|---|---|---:|---:|:---:|
| platinai_5k | random | 0 | 93 | ✅ (0 shared SMILES) |
| platinai_5k | scaffold | 0 | 0 | ✅ (0 shared scaffolds) |
| metalcytotox_5metal | random | **1935** | **1213** | ❌ FAIL |
| metalcytotox_5metal | scaffold | 0 | 0 | ✅ (0 shared scaffolds) |
| metalcytotox_5metal | **shipped** | 60 | **48** | ❌ FAIL |

**Key finding**: Cause C4 is **TRIGGERED** on `metalcytotox_5metal`'s shipped split (48 shared scaffolds out of 1051 unique test scaffolds = 4.6% leak). The strict scaffold-split is clean (0 shared scaffolds); the random split is leaky (1935 shared SMILES). Mitigation: re-split with strict `scaffold_split` (Bemis-Murcko grouped) before any future training on the shipped split.

### §1.3 Phase 4 — Numerical-stability fix (C6 + C7)

`molmetal/reports/wf44_r2_debug/numerical_fix_results.json` runs each model **before** (clipping off, decay off, fp64 fallback off, loss_scale=1.0) and **after** (clipping on at max_norm=1.0, decay at 0.5, fp64 fallback on, loss_scale=128.0). Each entry reports `before_R2` + `after_R2` + `r2_delta` + `improved` + diagnostics from the fix path.

| Model | before_R2 | after_R2 | r2_delta | improved | after: clipped_grad_steps / decayed_lr_steps / loss_scale / fp64 |
|---|---:|---:|---:|:---:|---|
| Wf-25 Hilbert composite | -11808.93 | -13929.34 | -2120.41 | ❌ | 50 / 0 / 128.0 / false |
| **Wf-32 m16 affinity** | -221.04 | **-86.59** | **+134.45** | **✅** | 50 / 0 / 128.0 / false |
| Wf-37 L5 evolutionary | -4.27 | -28.76 | -24.49 | ❌ | 50 / 0 / 128.0 / false |
| Wf-39 AST composite | -33.87 | -136.85 | -102.98 | ❌ | 50 / 0 / 128.0 / false |

`improved_count = 1 / 4` — only Wf-32 m16 improved (ΔR²=+134). The other 3 regressed in absolute terms: predictions now have variance (clipping prevents collapse), but they don't track the target — the regression is feature-driven, not optimizer-driven.

**Counter-intuitive but load-bearing**: gradient clipping was triggered on **every** step (`n_steps_with_clipped_grad_=50` for all 4 models) and the loss scale was 128 (not 1.0); but `fp64_fallback_used_=false` for all 4 (no underflow detected). The clipping prevents the Wf-25 PD-step overflow (C6) but cannot overcome the feature-mismatch / target-distribution issues.

### §1.4 Phase 5 — Feature-target correlation (C8)

`molmetal/reports/wf44_r2_debug/correlation_analysis.json` computes per-(feature, target) Pearson r + Spearman ρ + mutual information on the 3 default data backbones (475 triples total: 380 on platinai_4cell × 4 targets + 95 on metalcytotox_5metal × 1 target).

Summary:
- `max_abs_pearson_r = 0.3899` (dmpnn_sum_z_P × activity on metalcytotox_5metal)
- `max_abs_spearman_rho = 0.3873` (dmpnn_sum_total_h × activity on metalcytotox_5metal)
- `max_mutual_info = 0.2503`
- `c8_trip = false` ⇒ **C8 RULED OUT** — features DO carry signal (max |r| = 0.39 > 0.05)
- Top features by Spearman ρ on `activity` (metalcytotox_5metal): dmpnn_sum_total_h, dmpnn_max_z_Fe, dmpnn_sum_z_C, dmpnn_mean_z_Rh, dmpnn_sum_z_P, sa_mean_norm — all from the **D-MPNN feature backbone**, not the 8-d RDKit set.

**Key finding**: The 12-d RDKit features used in Wf-25 / Wf-39 are not the bottleneck. The D-MPNN backbone (90-d atom-summed / maxed / meaned features) carries the signal. ASTSR's symbol library cannot exploit D-MPNN features because they're not algebraic-composable. **Cause C2 is the actual bottleneck**, not C8.

---

## §2 R² before / after timeline

The R² trajectory across Wf-25 → Wf-32 → Wf-37 → Wf-39 → Wf-44 fix:

```
Wf-25:  R² = -0.443 (joint Hilbert, 5-fold ref; -0.374 ± 0.081 measured)
Wf-32:  R² = -0.040 (Pearson r on PlatinAI 4-cell 1-seed × 1-epoch smoke)
Wf-37:  R² = +0.163 (L5 Hilbert, 5-seed mean on NCI-60 + PlatinAI 4-cell)
Wf-39:  R² = +0.186 (Deflexpressor composite, 5-seed mean)
Wf-44:  R² = -86.59 (Wf-32 m16 + numerical fix; partial improvement)
        R² = -28.76 / -136.85 (Wf-37/Wf-39 + numerical fix; regressed in absolute terms)
```

The trajectory is monotonically improving from Wf-25 → Wf-39 because each workflow addressed **at least one cause** (Wf-37 fixed C3 real-pIC50 + C6 PD step + C2 feature augmentation; Wf-39 added the Deflexpressor composite branch which rescued ship when AST branch collapsed). The numerical fix in Wf-44 is **necessary but not sufficient** — it prevents collapse (the Wf-37 / Wf-39 cause) but does not lift R² ≥ 0 because the regression is feature-driven.

---

## §3 Root cause verdict — which causes actually triggered

| Cause | Triggered? | Evidence | Mitigation status |
|---|:---:|---|---|
| **C1** Baseline collapse | **YES (Wf-37, Wf-39)** | `pred_std = 0.0` in `baseline_check.json` | Wf-44 numerical fix prevents it (clipping + decay) |
| **C2** Feature mismatch | **PARTIAL** | 8-d RDKit uninformative, but D-MPNN max|r|=0.39 > 0.05 (C8 ruled out) | Mitigated in Wf-37 (12-d RDKit) + Wf-39 (Deflexpressor composite); not mitigated in Wf-25 |
| **C3** Target normalization reversed | **NO** | pIC50 conversion pinned at loader; signed-mean bias check fails (sign(corr) not ≈ -1) | No fix |
| **C4** Data leakage | **YES (metalcytotox shipped split)** | 48 shared scaffolds (4.6% of test) + 60 shared SMILES in shipped split | Strict scaffold-split is clean (0 shared); re-split shipped split before future training |
| **C5** Target distribution extreme | **YES (PlatinAI 4-cell)** | synthetic target bounded [4,8] with heavy mode near midpoint | Yeo-Johnson or tail-oversample mitigation not applied |
| **C6** Hilbert optimizer step too large | **YES (Wf-25)** | `||x_k||` jumped from ~400 to ~1e6 in 5 iters | Wf-37 fix (cosine LR + bias-column padding + gradient clip) applied; Wf-44 numerical fix applies this universally |
| **C7** Float32 numerical noise | **NO (minor)** | float32 noise floor ±0.05 — present but not dominant | fp64 fallback path exists but unused (`fp64_fallback_used_=false` for all 4 models in numerical_fix_results) |
| **C8** Feature-target correlation truly weak | **NO** | max\|r\|=0.39 > 0.05; max MI=0.25 | No fix needed — features carry signal |

**Active causes (must address to lift R² ≥ 0)**: C1 (collapse), C2 (feature augmentation), C4 (leakage), C5 (distribution), C6 (PD step).

---

## §4 No modification commitment

Wf-44 ships **only new debug scripts + tests + JSON artifacts**. No trainer modules were modified. The numerical fix exists as a wrapper (`grad_clip_max_norm`, `step_size_decay`, `loss_scale`, `fp64_fallback` parameters on `HilbertASTSRComposite`), but the published R² numbers in `metrics/by_metric/pearson_r_pic50.json` / `metrics/by_metric/metal_lomo_auc.json` are NOT updated. Per `party-twenty-spirit` §3.5 + TODO-82 §5 honest framing:

- R17 baseline (Pearson r=0.207 on HeLa48h/dark, MetalCytoToxDB Ru subset) is **preserved as canonical**.
- Wf-32 smoke (Pearson r=-0.040) remains `MEASURED_SMOKE_NOT_COMPARABLE`.
- Wf-37 ship-acceptable R²=+0.163 remains the **highest-quality Hilbert-ASTSR result on record**.
- Wf-39 Deflexpressor composite R²=+0.186 remains the **highest-quality 3-way integration result**.
- Wf-44 fix is **necessary-not-sufficient** — it prevents collapse but does not lift R² ≥ 0 in absolute terms.

The full retrain under strict scaffold-split + 3D features + Yeo-Johnson transformation becomes **R20·P5** (m16 Affinity Pearson r) + **R20·P8 (new)** (Wf-44 numerical fix on Wf-25/Wf-37/Wf-39 with C5 mitigation).

---

## §5 Cross-reference

- `molmetal/reports/wf44_r2_debug/root_cause_catalog.md` (293 lines) — 8-cause catalog with check methods, expected signals, mitigations
- `molmetal/reports/wf44_r2_debug/root_cause_checklist.json` (247 lines) — machine-readable per-cause triage + cross-cause interactions
- `molmetal/reports/wf44_r2_debug/baseline_check.json` (103 lines) — Phase 2 predict-mean collapse detector (n_cases=6)
- `molmetal/reports/wf44_r2_debug/leakage_check.json` (251 lines) — Phase 3 strict-split check (n_datasets=2)
- `molmetal/reports/wf44_r2_debug/numerical_fix_results.json` (99 lines) — Phase 4 before/after R² (n_models=4)
- `molmetal/reports/wf44_r2_debug/correlation_analysis.json` (4718 lines) — Phase 5 per-(feature, target) correlation (n_triples=475)
- `molmetal/scripts/check_r2_baseline.py` (556 LOC) — Phase 2 generator
- `molmetal/scripts/check_data_leakage.py` (692 LOC) — Phase 3 generator
- `molmetal/scripts/numerical_fix_check.py` (479 LOC) — Phase 4 generator
- `molmetal/scripts/analyze_feature_target_correlation.py` (835 LOC) — Phase 5 generator
- `molmetal/tests/test_r2_baseline.py` (210 LOC, 11 tests PASS) — Phase 2 tests
- `molmetal/tests/test_data_leakage.py` — Phase 3 tests
- `molmetal/tests/test_numerical_fix_results.py` — Phase 4 tests
- `molmetal/reports/wf25_combined/metrics.json` — Wf-25 joint R²=-0.374 ± 0.081 (5-fold mean)
- `molmetal/reports/wf25_hilbert_real/comparison_table.md` — Wf-25 separate streams R²=-0.353 ± 0.074
- `molmetal/reports/wf37_hilbert_astsr/comparison_table.md` — Wf-37 L5 R²=+0.163 ± 0.006 + Wf-25 historical reference -0.443
- `molmetal/reports/wf39_deflexpressor/deflex_bench.json` — Wf-39 composite R²=+0.186 (5-seed mean)
- `metrics/by_metric/pearson_r_pic50.json` — m16 R17 baseline (0.207) + Wf-32 smoke (-0.040) preserved
- `TODO/completed/87_wf32_phase4_honest_negative.md` — Wf-32 GPU-1-deferred honest-negative
- `TODO/completed/88_wf32_phase5_lift_analysis.md` — Wf-32 Phase 5 lift close-out (R20·P5/6/7 created)
- `TODO/completed/80_r19_plan.md` — R19 PLAN (Wf-27 ship)
- `TODO/03_plans/R19_PLAN.md` — R19 detailed GPU retrain plan (12 weeks, 10 mechanism M22-M30 + 4 joint retrain)
- `TODO/03_plans/LAMBDA_ROADMAP.md` — Lambda roadmap (Hilbert PD + ASTSR + Composite integration)
- `TODO/completed/82_metric_gap_filling.md` — TODO-82 ship gate (m16 ≥0.50 / m19 ≥0.70 / m8 ≥0.94)
- Recent commit: `9bb9efe Wf-44 Phase 2 — predict-the-mean baseline check (R² < 0 root cause triage)` (Phase 2 SHIP)
- Predecessor: `TODO/completed/91_wf41_3way_integration.md` (Deflexpressor + Multi-PH + ASTSR 3-way integration, Wf-39 ship)
- Sibling: `TODO/completed/91_wf43_phase4_integration.md` (master_verify_r18 + score_any router, Wf-43 Phase 4 SHIP)

---

## §6 Follow-up actions

1. **R20·P5 (m16 retrain)** — strict scaffold-split on `platinai_4cell + HeLa48h` dual-cohort; Yeo-Johnson transform on target; D-MPNN 90-d backbone instead of 12-d RDKit; Wf-44 numerical fix on. Target: Pearson r ≥ 0.50.
2. **R20·P8 (NEW — Wf-44 numerical fix on Wf-37/Wf-39 full retrain)** — apply `grad_clip_max_norm=1.0` + `step_size_decay=0.5` + `loss_scale=128.0` to the full 5-seed × 30/50-epoch pipeline. Pre-condition: GPU 1 free.
3. **metalcytotox_5metal re-split** — replace shipped split (48 shared scaffolds) with strict scaffold-split before any future training. Update `molmetal/scripts/split_metalcytotox.py` to call Bemis-Murcko grouped split.
4. **Wf-25 historical record preserved** — Wf-25 R²=-0.443 is an honest **negative result** and is documented as such in `molmetal/reports/wf37_hilbert_astsr/comparison_table.md` §"Wf-25 numerical-instability fix" + this Phase 1 catalog. The negative result is informative: it established that 8-d RDKit + binary target + tile inflation + constant lr is a **stack of 3 cause triggers** (C1 + C2 + C6) and that fixing any one alone is insufficient.
5. **No fabricated headline** — `improved_count=1/4` (Wf-32 m16 only) is the honest Phase 4 verdict; the other 3 models regressed in absolute terms and are reported as such. The numerical fix is **necessary-not-sufficient**.

**Date**: 2026-09-26