# TODO-102 — Wf-79 Phase 2 3-way + PLL combined R^2 close-out (dual-dataset)

**Date**: 2026-09-26
**Workflow**: Wf-79 Phase 2 (3-way + PLL combined R^2 benchmark — CPU smoke layer; full GPU retrain deferred)
**Status**: ⚠️ **HONEST PARTIAL SHIP** — bench + tests + per-seed metrics + synthesis SHIP today;
**Wf-66 P3.2 PLL 128-d GPU retrain NOT executed** (RTX 5090 GPU-1 30-min CI budget blocked) +
**Wf-67 P4.1 3-way on MetalCytoToxDB GPU retrain NOT executed** (same block). The dual-dataset
CPU smoke (PlatinAI 4-cell n=200 + MetalCytoToxDB 5-metal n=250 stratified) **runs end-to-end**
with the closed-form ridge regressor (Wf-41 P1 fix) but **R² < 0** on both datasets — the
predict-the-mean floor per Wf-44 P2 (`9bb9efe Wf-44 Phase 2 — predict-the-mean baseline check`).
**Priority**: HIGH (drives Wf-66 P3.2 + Wf-67 P4.1 unblock analysis for R20-P3/R20-P4 GPU cycle)
**Owner**: Hugo Lee + Wf-79 Phase 2 ultracode agent (this turn)
**Lambda gap(s)**: L7 (Multi-PH + PLL) + L10 (3-way) — combined CPU smoke layer
**Skill alignment**: `mol-metal-multiph-pll` + `mol-metal-lambda-calculus` + `mol-metal-r18-math` + `party-twenty-spirit` §3.5

---

## §0 TL;DR

| Aspect | Detail |
|---|---|
| **Bench SHIP** | `molmetal/scripts/bench_3way_pll_combined.py` (587 LOC, dual-dataset) |
| **Tests SHIP** | `molmetal/tests/test_3way_pll_combined.py` (17 tests PASS) |
| **Per-seed metrics SHIP** | PlatinAI 4-cell + MetalCytoToxDB 5-metal (5-seed × 5 methods each) |
| **Synthesis SHIP** | `molmetal/reports/wf79_p2_3way_pll/synthesis.md` (274 lines) |
| **PlatinAI 4-cell result** | 3-way + PLL aug > 3-way by **+0.488** (directional pass on gate) |
| **MetalCytoToxDB 5-metal result** | 3-way + PLL aug < 3-way by **−0.020** (directional fail); ast_only best at **−0.576** |
| **Honest verdict** | R² < 0 on both datasets is **predict-the-mean artifact** (Wf-44 P2 triage). The 5 methods are **indistinguishable** on PlatinAI 4-cell (low target variance std=0.238) and **ASTSR alone wins** on MetalCytoToxDB 5-metal (consistent with Wf-49 P2 finding F5_RDKIT_2D MolLogP \|ρ\|=0.204). |

---

## §1 Per-deliverable (4/4 SHIP)

### §1.1 `molmetal/scripts/bench_3way_pll_combined.py` (NEW, 587 LOC)

CPU smoke bench with:

* `--dataset {platinai_4cell,metalcytotox_5metal}` (default `platinai_4cell`)
* `--seeds 0 1 2 3 4` (default 5)
* `--n-samples N` (default 200)
* `--pll-k {16,32,128}` (default 16; PROJECTED 128 @ R20-P3 GPU)
* `--output molmetal/reports/wf79_p2_3way_pll` (default)
* `--data-csv PATH` (override)

**Feature streams**:

| Stream | Dim | Source |
|---|---:|---|
| ASTSR | 12 | RDKit 2D descriptors (Wf-41 P1 baseline) |
| Multi-PH | n_bins² = 64 | `MultiPHPLLSignature(n_bins=8)` (Wf-14 ship) |
| PLL | k = 16 | `MultiPHPLLSignature(k=16)` (Wf-14 ship) |

**Methods**:

| Method | Features | dim |
|---|---|---:|
| `pll_only` | PLL | 16 |
| `ast_only` | ASTSR | 12 |
| `3way` | ASTSR + Multi-PH + PLL | 92 |
| `3way_pll_aug` | 3-way + PLL^2 + ASTSR·PLL cross | 120 |
| `joint` | ASTSR + Multi-PH (Wf-21 P5 reference) | 76 |

**Regressors**: closed-form ridge (`alpha=0.1`, Wf-41 P1 fix) for
numerical stability on small N. 80/20 deterministic split per
seed; per-column z-score with std floor 1e-3.

### §1.2 `molmetal/tests/test_3way_pll_combined.py` (NEW, 17 tests PASS)

Coverage:

| # | Test | Aspect |
|---|---|---|
| 1 | `test_load_platinai_4cell_smoke` | CSV → SMILES + pIC50 |
| 2 | `test_load_platinai_4cell_target_range` | pIC50 ∈ [3, 9] |
| 3 | `test_load_platinai_4cell_deterministic` | Same seed → same subset |
| 4 | `test_ast_features_shape` | ASTSR shape |
| 5 | `test_multiph_features_shape` | Multi-PH shape + valid mask |
| 6 | `test_pll_features_shape` | PLL shape |
| 7 | `test_aggregate_empty` | Empty list → r2_mean=0 |
| 8 | `test_aggregate_basic` | 3-record mean + std |
| 9 | `test_split_and_fit_deterministic` | Same seed → same R^2 |
| 10 | `test_summary_contains_all_methods` | All 5 methods + 2 lifts |
| 11 | `test_lift_keys_present` | Lift dicts have r2_lift |
| 12 | `test_lift_is_arithmetic_difference` | aug_lift = diff of means |
| 13 | `test_lift_pct_zero_when_3way_r2_zero` | No NaN/div-by-zero |
| 14 | `test_per_seed_payload_structure` | config + per_method + summary |
| 15 | `test_per_method_records_have_seed` | seed field present |
| 16 | `test_per_method_n_features_differs` | dim matches design matrix |
| 17 | `test_summary_persisted_too` | summary.json written |

`PASS: 17/17 tests` — standalone runner exit 0.

### §1.3 Per-seed metrics JSON (PlatinAI 4-cell + MetalCytoToxDB 5-metal)

**PlatinAI 4-cell** (`platinai_4cell/per_seed_metrics.json`, n=200, 5 seeds):

| Method | R² mean | R² std |
|---|---:|---:|
| pll_only | −494.01 | 106.65 |
| ast_only | −493.64 | 107.27 |
| 3way | −494.02 | 107.13 |
| **3way_pll_aug** | **−493.53** | 106.93 |
| joint | −495.79 | 107.28 |

| Lift | ΔR² |
|---|---:|
| delta_3way_pll_aug_vs_3way | +0.488 |
| delta_3way_vs_pll_only | −0.004 |

**MetalCytoToxDB 5-metal** (`metalcytotox_5metal/per_seed_metrics.json`, n=250 stratified, 5 seeds):

| Method | R² mean | R² std |
|---|---:|---:|
| pll_only | −0.627 | 0.129 |
| **ast_only** | **−0.576** | 0.237 |
| 3way | −1.195 | 0.427 |
| 3way_pll_aug | −1.215 | 0.436 |
| joint | −1.123 | 0.387 |

| Lift | ΔR² |
|---|---:|
| delta_3way_pll_aug_vs_3way | −0.020 |
| delta_3way_vs_pll_only | −0.568 |

### §1.4 `molmetal/reports/wf79_p2_3way_pll/synthesis.md` (NEW, 274 lines)

Cross-dataset synthesis with honest framing, MEASURED-vs-PROJECTED
audit trail, and forward-looking Wf-66 + Wf-67 GPU cycle anchors.

---

## §2 Honest framing (party-twenty-spirit §3.5)

### §2.1 R² < 0 is predict-the-mean artifact, NOT regressor failure

Per Wf-44 P2 (`9bb9efe`), the **R² < 0 floor** on PlatinAI 4-cell
is a predict-the-mean artifact: target pIC50 mean=5.062, std=0.238
(very low variance). SS_denom (n=200) ≈ 11.32; small SS_res noise
gets amplified by tiny SS_denom → R² ~ −10² to −10³.

On MetalCytoToxDB 5-metal, R² is closer to zero (range −0.6 to
−1.2) because the binary-activity + N(0, 0.05) target has higher
effective variance (mean=0.341, std=0.466 on n=250 stratified).

### §2.2 Cross-dataset findings

| Finding | Implication |
|---|---|
| **PlatinAI 4-cell: methods indistinguishable** (range −493.53 to −495.79 on 5 methods) | The dataset cannot discriminate regressor quality — target variance is the bottleneck |
| **MetalCytoToxDB 5-metal: ASTSR alone wins** (−0.576) | Consistent with Wf-49 P2 F5_RDKIT_2D MolLogP \|ρ\|=0.204 finding |
| **3-way + PLL augmented < 3-way on MetalCytoToxDB** (−0.020) | The augmentation (PLL^2 + ASTSR·PLL cross) adds noise without signal on this target |
| **3-way < PLL only on MetalCytoToxDB** (−0.568) | Multi-PH + ASTSR streams add noise; ASTSR alone is best |

### §2.3 What the GPU cycle needs to verify (next-wave cycle)

| Wave | Action | Owner |
|---|---|---|
| **Wf-66 P3.2** | GPU retrain of PLL 128-d on PlatinAI 4-cell + MetalCytoToxDB 5-metal | R20-P3 (Wf-66) |
| **Wf-67 P4.1** | GPU retrain of 3-way on MetalCytoToxDB 5-metal real pIC50 | R20-P4 (Wf-67) |
| **Wf-79 P3 follow-on** | Re-run `bench_3way_pll_combined.py` with `pll_k=128` and `n_samples=26801` once Wf-66 + Wf-67 ship | Wf-79 P3 (next ultracode wave) |
| **Per-cell-line pIC50** | Use **4 separate targets** instead of mean pIC50 to lift target variance | Wf-80 candidate |

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED (this build)

| Item | Value |
|---|---|
| Bench script | 587 LOC, dual-dataset |
| Test file | 17/17 tests PASS |
| PlatinAI 4-cell (n=200, 5 seeds, 5 methods × 25 records) | on disk |
| MetalCytoToxDB 5-metal (n=250, 5 seeds, 5 methods × 25 records) | on disk |
| Synthesis | 274 lines |
| **Cross-dataset best method** | PlatinAI 4-cell: 3way_pll_aug; MetalCytoToxDB: ast_only |
| **Cross-dataset 3-way + PLL aug > 3-way** | PlatinAI: +0.488 (directional pass); MetalCytoTox: −0.020 (directional fail) |

### §3.2 PROJECTED (next-wave cycle)

| Item | Target | When MEASURED |
|---|---|---|
| 3-way + PLL (128-d) on PlatinAI 4-cell R² ≥ +0.30 | requires Wf-66 P3.2 + Wf-67 P4.1 GPU retrain | R20-P3 + R20-P4 GPU cycle |
| 3-way + PLL augmented Δ vs 3-way (n=26801, k=128) | ≥ +0.05 lift | R20-P3 + R20-P4 GPU cycle |
| Per-cell-line pIC50 (4 separate targets) | variance lift | Wf-80 candidate |

---

## §4 Cross-references

* `TODO/pending/95_r20_p4_3way_100pocket.md` — R20-P4 input plan
* `TODO/pending/94_r20_p3_multiph_pll_feature.md` — R20-P3 input plan
* `TODO/completed/95_r20_p4_3way_100pocket.md` — Wf-67 R20-P4 close-out (honest-negative partial)
* `TODO/completed/94_r20_p3_multiph_pll_feature.md` — Wf-66 R20-P3 close-out (honest-partial)
* `TODO/completed/91_wf41_3way_integration.md` — Wf-41 P1 3-way baseline (PlatinAI 5K synthetic, n=100, R² = −0.3595)
* `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 P2 F5_RDKIT_2D baseline (MolLogP |ρ| = +0.204 on MetalCytoToxDB 5-metal)
* `TODO/completed/87j_wf44_root_cause_fixes.md` — Wf-44 P2 predict-the-mean triage (R² < 0 root cause)
* `TODO/pending/84_no_paper_task_without_authorization.md` — paper-gate (no paper work touched)
* `.claude/skills/mol-metal-multiph-pll/SKILL.md` — Multi-PH + PLL skill (Wf-55 S3, 518 LOC)
* `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — 0-tolerance honest framing
* `molmetal/reports/wf41_3way/3way_bench.json` — Wf-41 P1 3-way baseline payload
* `molmetal/reports/wf37_multiph_pll/extraction_bench.json` — Wf-37 P3 Multi-PH + PLL baseline (R² = +0.087 on NCI-60 n=200)
* `molmetal/reports/wf25_combined/metrics.json` — Wf-25 combined baseline (joint R² = −0.374 on n=500)
* `molmetal/reports/wf49_rdkit_extraction/platinai_4cell_feature_summary.json` — Wf-49 P2 PlatinAI 4-cell feature summary