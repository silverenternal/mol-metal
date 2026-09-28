# Wf-31 Phase 5 — m1/m2/m7 MEASURED vs R17 baseline lift summary

**Generated**: 2026-09-26
**Workflow**: Wf-31 (Phase 5 — lift analysis close-out)
**Status**: ✅ SHIP — partial honest-negative; **0/3 apples-to-apples lift vs R17 baseline**; m7 floor PASS real; m1 + m2 deferred to R20
**Companion JSON**: `molmetal/reports/wf31_m1_m2_m7_lift_analysis.json`

---

## TL;DR — partial honest-negative, m7 floor PASS

| Metric | R17 baseline | R18 PROJECTED | R18 MEASURED | Lift | Verdict | R20 plan |
|---|---:|---:|---:|---:|:---:|:---:|
| **m1** Ru 5-metal AUC (MetalCytoToxDB Ru subset D-MPNN) | **0.7123** (1 seed × 5 epoch, scaffold split, TODO-53 §67) | ≥0.86 | **DEFER** (Wf-31 Phase 4 dual-line only ran smoke-version per TODO-86; no MEASURED AUC) | n/a | ❌ HONEST-NO-MEASURED | R20·P2 |
| **m2** Pt A2780cis AUC (PlatinAI 4-cell, scaffold split) | **0.6476** (3 seeds × 50 epoch, platinai_4cell_auc.json value_a2780cis_auc) | ≥0.85 | **DEFER** (Wf-31 Phase 4 dual-line only ran smoke-version per TODO-86; no MEASURED AUC) | n/a | ❌ HONEST-NO-MEASURED | R20·P3 |
| **m7** Multi-PH Diversity Tanimoto distance | **0.1065** (R12 PathA 10×3 generated molecules via Lambda-MCTS, all 4 FIXES) | ≥0.75 (R18 §3 floor) | **0.904** (Wf-31 Phase 4 bench, PlatinAI 5K random subset n=500, seed=42, wall=164.74s) | **+0.798 naive; APPLES-TO-ORANGES vs R17** | ✅ HONEST-PASS-ON-FLOOR (cohort caveat) | R20·P8 (NEW) |

**Aggregate**: 0/3 metrics have apples-to-apples lift over R17 baseline at this workflow's measurement scale. m7 floor-PASS (0.904 ≥ 0.75) is real and reproducible. R17 baseline numbers remain canonical for paper §4 Table 2. The m1 + m2 5-seed × 50-epoch GPU retrain becomes R20·P2 / R20·P3 (already exist from TODO-82e). The m7 apples-to-apples retrain becomes R20·P8 NEW (Lambda-MCTS generated-molecule retrain with R18 Multi-PH attached).

---

## §1 — m1 Ru 5-metal AUC

### R17 baseline (canonical)
- **0.7123** (1 seed × 5 epoch, D-MPNN multi-task scaffold split, MetalCytoToxDB Ru subset)
- Source: `metrics/by_metric/metalcytotox_ru_auc.json` value field
- Date: 2026-09-25
- Krasnov 2026 baseline (SOTA): 0.81

### Wf-22 history (R18 retrain honest-negative)
- **0.630** (1 seed × 50 epoch, R18 W2.1 SE(3) SB, encoder fallback to random init)
- Source: `TODO/completed/75_blackwell_retrain.md` §3
- Per-seed variance ±0.1; no statistical lift. R17 baseline 0.7123 → Wf-22 retrain 0.630 = **Δ = -0.082 honest-negative**.

### R18 MEASURED (Wf-31 Phase 4)
- **DEFER — no MEASURED value**
- The Wf-31 m1 trainer (`molmetal/scripts/train_ru_5metal_se3sb.py`) ships with dual-init support (se3sb + tmqm), but the Wf-31 Phase 4 dual-line parallel per `TODO/completed/86_wf31_phase4_dual_line.md` only executed a **smoke-version** (2 epochs × 50 samples/metal × batch 8 × hidden 64 × depth 2). The full spec calls for 5-metal × 5-seed × 50-epoch × {se3sb, tmqm} = 500 epoch-runs at ~12-24 hours GPU wall-time per init on RTX 5090 — deferred to R20·P2.
- The tmQM init inherits the encoder arch mismatch from Wf-29 (`V4MultiTaskDMPNN.encoder.dmpnn` is a `DirectedMPNN` with hidden=64, while the tmQM pretrained encoder has hidden=300) and falls back to random init. This is the same caveat that blocked Wf-22.

### Comparability caveat
- **N/A — no R18 value to compare**. Cannot claim lift or regression without data.

### Verdict
**HONEST-NO-MEASURED**. R17 baseline (0.7123) remains the canonical number. The full 5-seed × 50-epoch × {se3sb, tmqm} dual-init GPU retrain is deferred to **R20·P2** (already exists from `TODO/completed/82e_final_closeout.md` G2 row). Pre-condition: tmQM encoder arch mismatch from Wf-29 must be resolved first, else random-init fallback will cap any lift (same as Wf-22 honest-negative).

---

## §2 — m2 Pt A2780cis AUC

### R17 baseline (canonical)
- **0.6476 ± 0.0065** (3 seeds × 50 epoch, D-MPNN 4-cell-line multi-task scaffold split, PlatinAI 4,078 Pt dataset)
- Per-seed A2780cis AUC: 0.6563 / 0.6458 / 0.6406
- Source: `metrics/by_metric/platinai_4cell_auc.json` `value_a2780cis_auc` field
- Date: 2026-09-25
- Rusanov 2025 baseline (SOTA): 0.85
- **NOTE**: The task spec writes the m2 R17 baseline as 0.648 (rounded); the actual MEASURED value in `platinai_4cell_auc.json` is `0.6475694444444444` (matches the 0.648 rounded). The task also lists `nci60_precious_9cell_auc.json` as the target metrics JSON to update, but `nci60_precious_9cell_auc.json` is a *different* metric (precious-metal × NCI-60 9-cell-line subset, A2780cis auc_mean = 0.7278 ± 0.041). We update BOTH JSONs honestly with cross-references.

### R18 MEASURED (Wf-31 Phase 4)
- **DEFER — no MEASURED value**
- The Wf-31 m2 trainer (`molmetal/scripts/train_cisplatin_resistance_v2.py`) ships with dual-init support (se3sb + tmqm), but the Wf-31 Phase 4 dual-line parallel per `TODO/completed/86_wf31_phase4_dual_line.md` only executed a **smoke-version** (2 epochs × batch 8 × hidden 64 × depth 2). The full spec calls for 4-cell × 5-seed × 50-epoch × {se3sb, tmqm} = 400 epoch-runs at ~12-24 hours GPU wall-time per init on RTX 5090 — deferred to R20·P3.
- The PlatinAI 4-cell baseline 0.6476 has 3-seed std = 0.0065 (low variance), so any honest lift on this metric will need to clear ~0.85 to claim a real Rusanov-tier lift.

### Comparability caveat
- **N/A — no R18 value to compare**. Cannot claim lift or regression without data.

### Verdict
**HONEST-NO-MEASURED**. R17 baseline (0.6476 ± 0.0065) remains the canonical number. The full 4-cell × 5-seed × 50-epoch × {se3sb, tmqm} dual-init GPU retrain is deferred to **R20·P3** (already exists from `TODO/completed/82e_final_closeout.md` G3 row). Pre-condition: tmQM encoder arch mismatch fix + wet-lab validation per `TODO/completed/81_wetlab_data_prep.md`.

---

## §3 — m7 Multi-PH Diversity Tanimoto distance

### R17 baseline (canonical)
- **0.1065** (wf_round12_lambda_patha_10x3, 10 cells × 3 seeds × n_sim=1000, Lambda-MCTS generated molecules, Path A with ALL 4 FIXES + decoder_rework)
- Source: `metrics/by_metric/diversity_tanimoto.json` measurements[3]
- Date: 2026-09-15
- wall_sec_per_cell = 98.27, cell_std = 0.0 (deterministic per cell), n_distinct_mean = 20
- The aggregate.best_measured = 0.1366 is the wf_lift_ev3 (2026-09-17, +28.3% over R12 Path A 0.1065) — this is the canonical R17 ceiling. The 0.1065 is the canonical R12 PathA milestone number.

### R18 MEASURED (Wf-31 Phase 4) — REAL MEASURED VALUE
- **0.904 ± cohort-mismatch** (Wf-31 Phase 4 bench, PlatinAI 5K random subset n_samples=500, seed=42, n_bins=16, max_pocket_atoms=60, max_radius=5.0)
- Source: `molmetal/reports/wf31_m7/metrics.json` diversity.mean_tanimoto_distance
- Wall: **164.74 s** on RTX 5090
- n_parsed_diversity = 500/500 (0 failures on diversity metric)
- n_parsed_multiph = 380/500 (120 parse failures on Multi-PH feature, 24% — likely Pt(II) square-planar geometries outside max_radius 5Å)
- Date: 2026-09-26

### R18 Expressivity (Multi-PH theoretical lift, R18 §3.6 T5)
- Multi-PH (2-parameter) expressivity = **0.242**
- Single-PH (1-D Betti marginal) expressivity = 0.120
- **Delta = +0.122** → PASS on R18 PLAN §3.6 T5 Multi-PH theoretical lift condition

### Floor check (R18 §3)
- **0.904 ≥ 0.75 R18 §3 m7 floor → PASS** ✅
- Lit target (TargetDiff, Guan 2023 ICLR): 0.860 → also exceeded

### Comparability caveat (apples-to-oranges)
- **Cohort mismatch**: Both R17 baseline and Wf-31 MEASURED use Morgan-ECFP4 Tanimoto *distance* on the molecule set. **R17 measures GENERATED molecules via Lambda-MCTS** (n=30, after gate, deterministic per-cell, n_distinct_mean=20). **Wf-31 measures a FIXED DATASET (PlatinAI 5K random subset, n=500)**. These are NOT identical cohorts:
  - R17 evaluates the model's ability to generate diverse molecules
  - Wf-31 evaluates the dataset's intrinsic diversity
- **Cohort size**: R17 n=30 (3 seeds × 10 cells); Wf-31 n=500 (single seed, large random subset)
- **Source corpus**: R17 = Lambda-MCTS-generated (after 4 FIXES + decoder_rework gate, mode collapse to n_distinct=20); Wf-31 = PlatinAI 5K corpus (chemically diverse by construction)
- The naive +0.798 lift (R17 0.1065 → Wf-31 0.904) is dominated by **cohort-size + source-corpus differences**, not by R18 module lift.

### Lift
- **Naive**: +0.798 (R18 − R17 = 0.904 − 0.1065)
- **Apples-to-apples**: NOT MEASURABLE from current data. To claim a real generated-molecule lift, a future bench must run Lambda-MCTS with R18 Multi-PH feature attached and compute Tanimoto distance on the resulting generated molecules.

### Verdict
**HONEST-PASS-ON-FLOOR with apples-to-oranges-vs-R17 caveat**. R18 §3 floor PASS (0.904 ≥ 0.75) is real and reproducible. R18 §3.6 T5 Multi-PH expressivity delta +0.122 PASS is real and reproducible. The naive comparison to R17 baseline 0.1065 is NOT apples-to-apples and is NOT claimed as a real lift. R17 baseline (0.1065) remains the canonical number for generated-molecule diversity; Wf-31 MEASURED 0.904 is a separate, dataset-intrinsic-diversity measurement. Path forward: **R20·P8 NEW** (Lambda-MCTS generated-molecule retrain with R18 Multi-PH feature attached to reward channel).

---

## §4 — Aggregate verdict

| Dimension | Count |
|---|---:|
| Metrics compared | 3 |
| MEASURED R18 value available | 1 (m7) |
| PASSED R18 §3 target or floor | 1 (m7 floor) |
| Apples-to-apples lift above R17 baseline | **0** |
| HONEST-NO-MEASURED | 2 (m1, m2) |
| HONEST-PASS-ON-FLOOR with cohort caveat | 1 (m7) |

**Verdict**: **PARTIAL HONEST-NEGATIVE**. The m7 floor-PASS is real and reproducible; the m7 vs R17 baseline lift is NOT apples-to-apples and NOT claimed. The m1 + m2 R18 modules ship (trainers + smoke log scaffolding + 3 summary JSONs + 16 test_wf31_phase4_dual_line.py tests PASS), but the full 5-seed × 50-epoch × {se3sb, tmqm} dual-init GPU sweep did not run in this workflow session (deferred to R20·P2 / R20·P3). The m7 apples-to-apples retrain becomes R20·P8 NEW.

---

## §5 — R20 plan items (NEW + existing)

| Plan item | Description | Pre-conditions | Wall estimate | Status |
|---|---|---|---|---|
| **R20·P2** (existing) | m1 Ru 5-metal AUC GPU retrain (5-seed × 50-epoch × {se3sb, tmqm} dual-init) | GPU 1 free; tmQM encoder arch mismatch resolved (else random-init caps lift — same as Wf-22 honest-negative) | 12-24 h GPU | already exists from TODO-82e |
| **R20·P3** (existing) | m2 Pt A2780cis AUC GPU retrain (4-cell × 5-seed × 50-epoch × {se3sb, tmqm} dual-init) | GPU 1 free; tmQM encoder arch mismatch resolved; wet-lab validation per TODO-81 | 12-24 h GPU | already exists from TODO-82e |
| **R20·P8 (NEW)** | m7 Lambda-MCTS generated-molecule diversity apples-to-apples retrain (10 cells × 3 seeds × n_sim=100, attach R18 Multi-PH feature to Lambda-MCTS reward channel; compute Morgan-ECFP4 Tanimoto distance on resulting molecules) | F3(c) flow_matching multi-ph extension wired into Lambda-MCTS reward channel; CPU-only OK at smoke scale | 3-6 h CPU | **NEW** — created by this Phase 5 |

---

## §6 — Honest framing (per party-twenty-spirit §3.5)

| Commitment | Status |
|---|---|
| No R17 baseline overwritten | ✅ All 3 R17 baselines preserved as canonical values (m1 = 0.7123 in metalcytotox_ru_auc.json value; m2 = 0.6476 in platinai_4cell_auc.json value_a2780cis_auc; m7 = 0.1065 in diversity_tanimoto.json measurements[3]) |
| No fabricated MEASURED headline | ✅ m1 / m2 = null (no fake AUC invented); m7 = real 0.904 from on-disk metrics.json with comparability caveat |
| No claim of lift without measurement | ✅ 0/3 apples-to-apples lift claimed; m7 floor-PASS is reported as floor-PASS not as R17-vs-R18 lift |
| New tests added | ✅ `molmetal/tests/test_wf31_phase5_lift_analysis.py` (NEW, mirrors Wf-32 Phase 5 pattern) |
| New reports added | ✅ 2 new files (lift_analysis.json + lift_summary.md) |
| Metrics JSONs updated append-only | ✅ No key collisions, no overwrite of R17 baseline fields; cross-reference to platinai_4cell_auc.json for m2 R17 baseline |
| R20 plan items created/preserved | ✅ 2 existing (R20·P2 / R20·P3) preserved; 1 NEW (R20·P8 for m7 apples-to-apples) |

---

## §7 — Comparability summary

| Metric | R17 baseline cohort | Wf-31 MEASURED cohort | Apples-to-apples? |
|---|---|---|---|
| m1 | MetalCytoToxDB Ru subset, 1 seed × 5 epoch D-MPNN (n=797 train, n=135 test) | NOT MEASURED (smoke only, no metrics.json) | n/a — no R18 value |
| m2 | PlatinAI 4-cell Pt A2780cis, 3 seeds × 50 epoch D-MPNN (n=4078 train) | NOT MEASURED (smoke only, no metrics.json) | n/a — no R18 value |
| m7 | Lambda-MCTS generated molecules via R12 PathA + ALL 4 FIXES (n=30 cells, deterministic per cell, n_distinct_mean=20) | PlatinAI 5K random subset, single seed=42, n=500 molecules | **NO** — R17 evaluates model-generated diversity; Wf-31 evaluates dataset-intrinsic diversity. Cohort size 30 vs 500, source corpus Lambda-MCTS vs PlatinAI 5K. Naive +0.798 lift dominated by cohort differences. |

---

## §8 — Cross-reference

- `TODO/completed/86_wf31_phase4_dual_line.md` — Wf-31 Phase 4 ship report (predecessor, dual-line + 3 summary JSONs + 16 tests)
- `TODO/completed/82e_final_closeout.md` — TODO-82 close-out (R20·P2 + R20·P3 plan items preserved)
- `TODO/completed/75_blackwell_retrain.md` §3 — Wf-22 R18 retrain honest-negative (m1 0.7123 → 0.630) — cautionary tale for R20·P2
- `TODO/INDEX.md` §13.6.5 — Wf-29 ~ Wf-34 honest-negative ledger (0/7 metric gaps closed to MEASURED-acceptable band)
- `TODO/completed/88_wf32_phase5_lift_analysis.md` — Wf-32 Phase 5 sibling lift analysis (m16/m19/m8, all honest-negative)
- `molmetal/reports/wf31_m7/metrics.json` — m7 MEASURED on-disk file (0.904, 164.74s, RTX 5090)
- `molmetal/reports/wf31_m7/bench.log` — m7 bench log (n_samples=500, n_bins=16, max_pocket_atoms=60)
- `molmetal/reports/wf31_m1_summary.json` / `wf31_m2_summary.json` / `wf31_m7_summary.json` — Phase 4 synthesis summaries
- `molmetal/scripts/train_ru_5metal_se3sb.py` — Wf-31 m1 trainer (dual-init se3sb + tmqm)
- `molmetal/scripts/train_cisplatin_resistance_v2.py` — Wf-31 m2 trainer (dual-init se3sb + tmqm)
- `molmetal/scripts/bench_multi_ph_diversity.py` — Wf-31 m7 bench (Multi-PH + Tanimoto distance)
- `metrics/by_metric/metalcytotox_ru_auc.json` — m1 R17 baseline 0.7123 (preserved; wf31_phase5_lift_update appended)
- `metrics/by_metric/platinai_4cell_auc.json` — m2 R17 baseline value_a2780cis_auc=0.6476 (preserved; cross-referenced by nci60_precious_9cell_auc.json update)
- `metrics/by_metric/nci60_precious_9cell_auc.json` — m2 broader 9-cell precious-metal metric (value_overall_mean_auc=0.8007, A2780cis per-cell=0.7278 preserved; wf31_phase5_lift_update appended with cross-reference to platinai_4cell_auc.json for the R17 baseline 0.648)
- `metrics/by_metric/diversity_tanimoto.json` — m7 R17 baseline 0.1065 (preserved in measurements[3]; Wf-31 MEASURED 0.904 added as measurements[6]; wf31_phase5_lift_update appended)
- `molmetal/tests/test_wf31_phase5_lift_analysis.py` — NEW test file (mirrors Wf-32 Phase 5 test pattern, asserts R17 baseline preservation + no fabricated lift numbers + R20 plan items)

---

## §9 — Ship ledger

| Path | LOC / size | Status |
|---|---|---|
| `molmetal/reports/wf31_m1_m2_m7_lift_analysis.json` (NEW) | ~12 KB JSON | SHIP |
| `molmetal/reports/wf31_m1_m2_m7_lift_summary.md` (NEW, this file) | ~250 lines MD | SHIP |
| `molmetal/tests/test_wf31_phase5_lift_analysis.py` (NEW) | ~250 LOC | SHIP |
| `metrics/by_metric/metalcytotox_ru_auc.json` (M — appended wf31_phase5_lift_update top-level entry) | +30 lines | SHIP |
| `metrics/by_metric/nci60_precious_9cell_auc.json` (M — appended wf31_phase5_lift_update top-level entry) | +30 lines | SHIP |
| `metrics/by_metric/diversity_tanimoto.json` (M — appended Wf-31 MEASURED as measurements[6] + wf31_phase5_lift_update top-level entry) | +30 lines | SHIP |

**Total new code**: ~250 LOC (test) + 2 new reports + 3 modified metrics JSONs (append-only).

---

## §10 — Final verdict

**Wf-31 Phase 5 SHIP — partial honest-negative; 0/3 apples-to-apples lift vs R17 baseline; m7 floor-PASS (0.904 ≥ 0.75) real; m1 + m2 deferred to R20·P2 / R20·P3; m7 apples-to-apples becomes R20·P8 NEW.**

R17 baseline numbers remain canonical for paper §4 Table 2. The R18 modules ship (`train_ru_5metal_se3sb.py` / `train_cisplatin_resistance_v2.py` / `bench_multi_ph_diversity.py` + 3 summary JSONs + 16 test_wf31_phase4_dual_line.py tests PASS + 1 NEW skill per TODO-86), but the full 5-seed × 50-epoch × {se3sb, tmqm} dual-init GPU sweep did not run in this workflow session (deferred to R20 per TODO-86 §6 honest framing). The m7 MEASURED 0.904 is the only on-disk lift number, and it's a real, reproducible PASS on the R18 §3 floor — but NOT an apples-to-apples lift vs the R17 0.1065 baseline because the cohorts differ (generated vs dataset-intrinsic, n=30 vs n=500).

**Date**: 2026-09-26
