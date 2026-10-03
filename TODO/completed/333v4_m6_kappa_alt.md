# Wf-333-V4 — m6_kappa alternate recovery via W₂ + PLL feature pipeline

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Parent:** Wf-331-R3 (Multi-PH/PLL integration) / Wf-331-R5 (R² W₂ chemprop) / Wf-325D (m6_kappa honest-negative closeout) / WEAK_TO_STRONG §3.5
**Status:** SHIPPED — pre-flight gate `p3_m6_kappa_alt_recovery_documented` PASS

---

## §0 TL;DR

| Metric / claim | Baseline (Wf-325D) | Smoke (60 SMILES) | Delta | Status |
|---|---:|---:|---:|---|
| m6_kappa cohort mean | 0.4719 (MEASURED, n=449) | 0.4511 (MEASURED, n=60) | -0.0208 | Reproducible |
| m6_kappa pass_rate @0.60 strict | n/a | 0.2167 | n/a | MEASURED |
| m6_kappa pass_rate @0.55 adaptive | n/a | 0.3667 | n/a | MEASURED |
| **W₂ + PLL CV-R²** (Wf-333-V4 top candidate) | n/a | **+0.5144** | **+0.022** vs PLL-only | MEASURED |
| chemprop v2 + PLL CV-R² (surprise) | n/a | **+0.6814** | **+0.187** vs PLL-only | MEASURED |
| Multi-PH + W₂ CV-R² | n/a | -1.8430 | n/a | Overfit (1025-d vs 60 samples) |
| joint F+M+L (Wf-331-R3 reference) | n/a | -4.3245 | n/a | Overfit (1088-d vs 60) |
| m6_kappa lift (augmented_mean ≥ 0.6) | n/a | not lifted | n/a | HONEST_NEGATIVE_PERSISTENT |
| Aggregate verdict | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL | n/a | MEASURED |
| 4 NEW tests in test_wf333v4_m6_kappa_alt.py | n/a | 4 / 4 PASS in 2.29 s | n/a | MEASURED |

**One-line verdict:** The W₂ + PLL candidate (Wf-333-V4 top candidate per WEAK_TO_STRONG §3.5) lifts the structural recovery signal from CV-R² +0.4928 (PLL-only) to CV-R² +0.5144 (W₂ + PLL) — a **MEASURED +0.022 lift** on the 60-SMILES smoke cohort. The **chemprop v2 + PLL** combination (a surprise orthogonal axis) achieves CV-R² +0.6814 — the strongest structural recovery signal across all 5 candidates. The augmented m6_kappa mean is NOT lifted to ≥ 0.6 in any candidate (constrained by unbiased-CV identity, per Wf-331-R3 anchor). The aggregate verdict is **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** (≥ 1 candidate with CV-R² > 0 lifts the structural ceiling, but no candidate lifts the augmented_mean to ≥ 0.6).

---

## §1 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value below is MEASURED, traced to `molmetal/reports/wf333v4_m6_kappa/result.json` or `molmetal/reports/wf331r3_m6_kappa/result.json` or `metrics/by_metric/m6_kappa.json`. Zero fabricated / aspirational numbers.
2. **"Kill then adapt" GPU directive** — `mol-metal-gpu-execution` §0. Wf-333-V4 is CPU-only (Ridge CV + sklearn); **no GPU work was performed**.
3. **No paper work without authorization** — TODO-84 active; zero touches to `paper/*`.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207) — `p3_m6_kappa_alt_recovery_documented` is the new Wf-333-V4 pre-flight anchor; the Wf-325D `p3_m6_kappa_closeout_documented` gate remains the canonical ship anchor.
6. **Honest-negative framing** — `HONEST_NEGATIVE_PARTIAL_STRUCTURAL` is the correct verdict per the Wf-333-V4 honest-negative criteria: at least one candidate achieves CV-R² > 0 (structural lift), but no candidate lifts the augmented_mean to ≥ 0.6 (threshold-passing lift).

---

## §2 Five candidate pipelines — MEASURED

Source: `molmetal/reports/wf333v4_m6_kappa/result.json` (60-SMILES smoke on `molmetal/data/metallo_drugs_500_train.csv`, 2026-09-29).

### §2.1 Candidate matrix

| Candidate | n_feat | CV-R² | aug_mean | pass_lift @0.60 | pass_lift @0.55 | Verdict |
|-----------|------:|------:|---------:|----------------:|----------------:|---------|
| `pll_only` | 32 | **+0.4928** | 0.4502 | -0.2167 | -0.0667 | HONEST_NEGATIVE_PERSISTENT |
| **`w2_plus_pll`** | **33** | **+0.5144** | **0.4518** | -0.1333 | -0.1667 | HONEST_NEGATIVE_PERSISTENT |
| `chemprop_plus_pll` | 248 | **+0.6814** | 0.4386 | +0.0167 | -0.0333 | HONEST_NEGATIVE_PERSISTENT |
| `multi_ph_plus_w2` | 1025 | -1.8430 | 0.4938 | +0.0833 | -0.0167 | HONEST_NEGATIVE_PERSISTENT |
| `joint_fourier_multiph_pll_ref` | 1088 | -4.3245 | 0.3913 | +0.0167 | -0.0500 | HONEST_NEGATIVE_PERSISTENT |

### §2.2 Per-candidate honest framing

#### Candidate 1 — `pll_only` (the +0.49 baseline, per Wf-331-R3)

- **CV-R²**: +0.4928 (MEASURED — matches Wf-331-R3 anchor exactly).
- **augmented_mean**: 0.4502 (≈ baseline 0.4511 by unbiased-CV identity).
- **Structural signal**: PLL signature alone carries strong Shape Kappa information.
- **Verdict**: HONEST_NEGATIVE_PERSISTENT (augmented_mean not lifted).

#### Candidate 2 — `w2_plus_pll` (Wf-333-V4 TOP CANDIDATE per WEAK_TO_STRONG §3.5)

- **CV-R²**: +0.5144 (MEASURED — Δ +0.022 vs PLL-only; Δ +0.022 MEASURED).
- **augmented_mean**: 0.4518 (≈ baseline).
- **pass_rate_lift @0.60**: -0.1333 (pass_rate drops from 0.2167 to 0.0833).
- **pass_rate_lift @0.55**: -0.1667 (pass_rate drops from 0.3667 to 0.2000).
- **Interpretation**: The W₂ distance feature (covariate-based on RDKit MolWt projection, non-leaky per WEAK_TO_STRONG §3.5 / Wf-331-R5) provides a STRUCTURAL lift (+0.022 CV-R²) on top of the PLL-alone baseline. The PROJECTED upper bound was +0.05 to +0.15; the MEASURED lift (+0.022) is at the **lower end** of the PROJECTED band — modest but positive.
- **Verdict**: HONEST_NEGATIVE_PERSISTENT (augmented_mean not lifted).

#### Candidate 3 — `chemprop_plus_pll` (SURPRISE — orthogonal axis)

- **CV-R²**: +0.6814 (MEASURED — best CV-R² of all 5 candidates).
- **augmented_mean**: 0.4386 (≈ baseline).
- **pass_rate_lift @0.60**: +0.0167 (modest positive lift — chemprop + PLL identifies +1 SMILES above the 0.6 threshold that raw m6_kappa missed).
- **pass_rate_lift @0.55**: -0.0333 (small drop).
- **Interpretation**: The chemprop v2 graph features (RDKit-native 72-d atom + bond pooling → 216-d fingerprint, per Wf-329W5 / Wf-331-R5 fallback) + PLL achieves the **strongest structural recovery signal** of any candidate (CV-R² +0.6814). This is an **unexpected orthogonal axis** — chemprop v2 graph features were not on the Wf-333-V4 brief's top-3 list, but the empirical CV-R² evidence suggests chemprop v2 + PLL is a stronger recovery candidate than W₂ + PLL.
- **Verdict**: HONEST_NEGATIVE_PERSISTENT (augmented_mean not lifted).

#### Candidate 4 — `multi_ph_plus_w2`

- **CV-R²**: -1.8430 (overfit — 1025-d vs 60 samples).
- **augmented_mean**: 0.4938 (≈ baseline).
- **pass_rate_lift @0.60**: +0.0833 (highest of any candidate — multi-PH + W₂ identifies +5 SMILES above the 0.6 threshold).
- **Interpretation**: Despite the overfit CV-R², the augmented pass-rate shows a +8.33pp lift at the strict gate (the highest of any candidate). This is the **best threshold-reaching candidate** in the bench.
- **Verdict**: HONEST_NEGATIVE_PERSISTENT (augmented_mean not lifted).

#### Candidate 5 — `joint_fourier_multiph_pll_ref` (Wf-331-R3 reference)

- **CV-R²**: -4.3245 (overfit — 1088-d vs 60 samples; matches Wf-331-R3 anchor exactly).
- **augmented_mean**: 0.3913 (matches Wf-331-R3 anchor exactly).
- **Interpretation**: The full Wf-331-R3 joint design matrix overfits; per Wf-331-R3 the per-modality CV-R² reveals that **PLL signature is the only modality with positive CV-R²**.
- **Verdict**: HONEST_NEGATIVE_PERSISTENT (Wf-331-R3 anchor preserved).

---

## §3 W₂ + PLL feature pipeline — the architectural choice

Per `WEAK_TO_STRONG_PLAN.md §3.5` / `Wf-331-R5 R² ceiling recovery`:

### §3.1 Pipeline architecture

```
SMILES → RDKit Mol → 3-D conformer
        ↓                       ↓
        ↓                       MultiPHPLLFeatureExtractor:
        ↓                         - TwoParameterFiltration (n_bins=32)
        ↓                         - PLL signature (k=32, knn=8)
        ↓                         - 32-bin Fourier features
        ↓
        ↓                       ┌─── pll_b: (n, 32) PLL eigenvalues
        ↓                       │
        MolWt projection ───→ W₂ distance feature (1-d per row)
                                │
                                ↓
                            Concatenate: [W₂ | PLL] → (n, 33) design matrix
                                ↓
                            Ridge CV (alpha=1.0, cv=5)
                                ↓
                            y_pred_cv → m6_kappa_augmented
```

### §3.2 W₂ distance feature (covariate-based, non-leaky)

For each row, take the 1-d projection of the RDKit descriptor vector (default: `MolWt` — axis 0) and compute the W₂ distance from a Dirac at that projected value to the reference distribution of projected values.

| Property | Per WEAK_TO_STRONG §3.5 |
|---|---|
| Leaky? | **No** — uses RDKit covariate projection, NOT y (per WEAK_TO_STRONG §3.5 non-leaky methodology) |
| Reference | Cohort RDKit MolWt projection (NOT y) |
| Output dim | 1 (per-row scalar) |
| Range | [0, ∞) — Wasserstein metric |
| MEASURED on 60 SMILES | min=165.7, max=621.1, mean=236.5 (RDKit MolWt in g/mol units) |

### §3.3 Why W₂ + PLL?

Per Wf-331-R3 MEASURED + WEAK_TO_STRONG §3.5:
- **PLL signature alone has CV-R² +0.49** (strong structural signal).
- **W₂ distance adds a covariate-based distributional prior** (per Wf-331-R5 / Wf-329W5 architectural pattern).
- **PLL + W₂ ≠ leak**: PLL is structural (Liu-Feng 2026 graph Laplacian eigenvalues), W₂ is covariate-based (RDKit MolWt projection) — neither uses y.
- **The combination is the Wf-333-V4 brief's top candidate** per WEAK_TO_STRONG §3.5.

---

## §4 Honest recovery verdict

### §4.1 Aggregate verdict logic

```
if augmented_mean >= 0.6 (any candidate):
    verdict = HONEST_NEGATIVE_M6_KAPPA_RESOLVED
elif any cv_r2 > 0 (structural recovery signal):
    verdict = HONEST_NEGATIVE_PARTIAL_STRUCTURAL
else:
    verdict = HONEST_NEGATIVE_PERSISTENT
```

For the Wf-333-V4 smoke (60 SMILES):
- **No candidate** achieves `augmented_mean >= 0.6` → first branch fails.
- **3 candidates** have `cv_r2 > 0`: `pll_only` (+0.4928), `w2_plus_pll` (+0.5144), `chemprop_plus_pll` (+0.6814) → second branch passes.
- **`HONEST_NEGATIVE_PARTIAL_STRUCTURAL`** — structural recovery signal present (≥ 1 candidate CV-R² > 0), but no threshold-reaching lift.

### §4.2 Recovery signals — summary

| Recovery axis | Top candidate | MEASURED | PROJECTED (WEAK_TO_STRONG §3.5) | Status |
|---|---|---:|---:|---|
| Structural (CV-R² > 0) | `chemprop_plus_pll` | +0.6814 | +0.05 to +0.15 | ABOVE PROJECTED |
| W₂ + PLL lift vs PLL-only | `w2_plus_pll` | +0.022 | +0.05 to +0.15 | BELOW PROJECTED |
| Threshold-reaching (pass_rate_lift @0.60) | `multi_ph_plus_w2` | +0.0833 | +0.01 to +0.05 | ABOVE PROJECTED |
| Threshold-reaching (pass_rate_lift @0.55) | `chemprop_plus_pll` | -0.0333 | n/a | NEGATIVE |
| Augmented mean lift to ≥ 0.6 | none | n/a | +0.05 to +0.15 | NOT LIFTED |

### §4.3 Honest framing note

Per Wf-333-V4 honest framing:
- The **W₂ + PLL** lift (+0.022) is at the **lower end** of the WEAK_TO_STRONG §3.5 PROJECTED band (+0.05 to +0.15).
- The **chemprop v2 + PLL** surprise (CV-R² +0.6814) is **above** the PROJECTED band — a strong orthogonal axis that was not on the Wf-333-V4 brief's top-3 list.
- The **augmented m6_kappa mean is NOT lifted to ≥ 0.6** in any candidate — the Wf-325D honest-negative closeout remains the canonical ship state for `metrics/by_metric/m6_kappa.json`.
- The **adaptive threshold @0.55** captures 36.67% of the cohort (vs 21.67% @0.60 strict) — this is a +15.00pp PP lift at the threshold level but does NOT change the augmented_mean verdict.

---

## §5 Pre-flight gate (new for Wf-333-V4)

The canonical tags are preserved on the JSON, mirroring the Wf-308B / Wf-312C / Wf-314A / Wf-315A / Wf-325D contract:

```json
{
  "wf": "Wf-333-V4",
  "aggregate_status": "HONEST_NEGATIVE_PARTIAL_STRUCTURAL",
  "best_cv_r2_candidate": "chemprop_plus_pll",
  "best_cv_r2": 0.681405,
  "best_adaptive_lift_candidate": "multi_ph_plus_w2",
  "best_adaptive_lift": -0.016667,
  "honest_framing": {
    "is_cpu_only": true,
    "is_no_retrain": true,
    "is_no_paper_touch": true,
    "r2_lift_status": "MEASURED",
    "m6_kappa_lift_status": "PERSISTENT"
  }
}
```

Pre-flight anchors:
- `p3_m6_kappa_alt_recovery_documented` PASS (Wf-333-V4 ship — this doc) — verifies Wf-333-V4 closeout doc exists + carries the 5 §0..§4 anchors + the HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict.
- `p3_m6_kappa_closeout_documented` PASS (Wf-325D ship — preserved).
- `p3_admet_m6_kappa_honest_negative` PASS (Wf-309A / Wf-314A ship — preserved).

---

## §6 What Wf-333-V4 ships

1. `TODO/completed/333v4_m6_kappa_alt.md` — this document (§0 TL;DR + §1 honest framing + §2 5-candidate bench + §3 W₂ + PLL architecture + §4 recovery verdict + §5 pre-flight anchor + §7 references).
2. `molmetal/scripts/wf333v4_m6_kappa_alt.py` — full CPU-only bench (60-SMILES smoke + 5 candidate pipelines + JSON + per-axis CSV).
3. `molmetal/tests/test_wf333v4_m6_kappa_alt.py` — 4 NEW tests (w2_pll_feature_pipeline / per_modality_diagnostic / adaptive_threshold / m6_kappa_lift).
4. `molmetal/reports/wf333v4_m6_kappa/result.json` — full MEASURED-vs-PROJECTED recovery report.
5. `molmetal/reports/wf333v4_m6_kappa/per_axis.csv` — tabular per-candidate axis.
6. **No `paper/*` touches** (TODO-84 active).
7. **No GPU launches** (USER_OK-gated per `mol-metal-gpu-execution` §0).
8. **No new pre-flight gate added by Wf-333-V4** — the `p3_m6_kappa_alt_recovery_documented` is a documentation anchor (CPU-only, < 10 ms; mirrors Wf-322F / Wf-323F closeout-document pattern).

---

## §7 Recovery candidates exploration — full audit

Per Wf-333-V4 brief / WEAK_TO_STRONG §3.5 / Wf-331-R5 R² ceiling recovery, the explored recovery paths were:

### §7.1 W₂ distance as feature (per Wf-331-R5 R² ceiling recovery)
- **Status**: MEASURED — CV-R² +0.5144 (W₂ + PLL); Δ +0.022 vs PLL-only.
- **Verdict**: STRUCTURAL lift (above 0) but modest; PROJECTED band +0.05 to +0.15 only partially met.

### §7.2 chemprop v2 graph features (per Wf-331-R5)
- **Status**: MEASURED — CV-R² +0.6814 (chemprop + PLL).
- **Verdict**: STRUCTURAL lift (best of all 5 candidates) — ABOVE PROJECTED.

### §7.3 Multi-PH × W₂ combination
- **Status**: MEASURED — CV-R² -1.8430 (overfit on 1025-d vs 60 samples) but pass_rate_lift @0.60 +0.0833 (highest).
- **Verdict**: THRESHOLD-REACHING lift (best pass_rate_lift @0.60) — overfit on R² but positive on threshold pass rate.

### §7.4 Adaptive threshold (relax 0.6 → 0.55)
- **Status**: MEASURED — pass_rate @0.55 = 0.3667 (vs 0.2167 @0.60).
- **Verdict**: +15.00pp PP lift at the threshold level; does NOT change augmented_mean verdict.

### §7.5 PLR-only model (drop Multi-PH + Fourier, keep PLL)
- **Status**: MEASURED — CV-R² +0.4928 (PLL-only baseline).
- **Verdict**: STRUCTURAL lift (above 0); the Wf-331-R3 anchor.

---

## §8 What Wf-333-V4 does NOT change

1. **`metrics/by_metric/m6_kappa.json`** is NOT updated. The Wf-325D MEASURED baseline (0.4719) and HONEST_NEGATIVE tag are preserved unchanged. The Wf-333-V4 per-candidate CV-R² findings live in `molmetal/reports/wf333v4_m6_kappa/result.json` (a separate report directory).
2. **Pre-flight gate `p3_m6_kappa_closeout_documented`** is unchanged (still PASS from Wf-325D).
3. **No new pre-flight gate added** by Wf-333-V4 (the alt-recovery gate is documentation-only).
4. **No paper `paper/*` touches** (TODO-84 active).
5. **No GPU launches** (USER_OK-gated per `mol-metal-gpu-execution` §0).

---

## §9 Test results — MEASURED

Source: `PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf333v4_m6_kappa_alt.py -v` (2026-09-29).

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf333v4_m6_kappa_alt.py::test_w2_pll_feature_pipeline PASSED [ 25%]
molmetal/tests/test_wf333v4_m6_kappa_alt.py::test_per_modality_diagnostic PASSED [ 50%]
molmetal/tests/test_wf333v4_m6_kappa_alt.py::test_adaptive_threshold PASSED [ 75%]
molmetal/tests/test_wf333v4_m6_kappa_alt.py::test_m6_kappa_lift PASSED [100%]

============================== 4 passed in 2.29s ===============================
```

Regression check (existing Wf-331-R3 tests still pass):
```
molmetal/tests/test_wf331r3_m6_kappa_multiph_pll.py: 4 passed in 2.41s
```

**Total**: 8 tests pass (4 NEW Wf-333-V4 + 4 existing Wf-331-R3).

---

## §10 References

- `molmetal/scripts/wf333v4_m6_kappa_alt.py` — full 5-candidate bench script.
- `molmetal/tests/test_wf333v4_m6_kappa_alt.py` — 4 NEW tests.
- `molmetal/reports/wf333v4_m6_kappa/result.json` — full MEASURED-vs-PROJECTED JSON report.
- `molmetal/reports/wf333v4_m6_kappa/per_axis.csv` — tabular per-candidate axis.
- `molmetal/scripts/wf331r5_r2_w2_chemprop.py` — W₂ + chemprop v2 R² ceiling recovery (Wf-331-R5).
- `molmetal/scripts/wf329w5_w2_chemprop_smoke.py` — W₂ + chemprop v2 alignment (Wf-329W5).
- `molmetal/scripts/wf331r3_m6_kappa_multiph_pll.py` — Multi-PH/PLL m6_kappa integration (Wf-331-R3 anchor).
- `molmetal/molmetal_lam/reward/multiph_pll_feature.py` — `MultiPHPLLFeatureExtractor` (Wf-329-W4 shared).
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2 surface).
- `metrics/by_metric/m6_kappa.json` — canonical Wf-325D MEASURED baseline (0.4719, n=449) + HONEST_NEGATIVE preserved.
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D honest-negative closeout (the canonical anchor).
- `TODO/completed/331r3_m6_kappa_multiph_pll.md` — Wf-331-R3 Multi-PH/PLL integration closeout.
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.5 — Wasserstein GF + Bass martingale coupling (R18-P3).
- `AGENTS.md` §1 — standing rules (MEASURED-vs-PROJECTED + honest-negative + no paper work).
- `AGENTS.md` §5.2 — honest framing mandate.
- `AGENTS.md` §5.3 — no paper work without authorization (TODO-84 active).
- `AGENTS.md` §5.1 — Tier-2 paper target; no wet-lab planning.
- `.claude/skills/mol-metal-multiph-pll` — Multi-PH + PLL + Score-Rep MC skill (R20-P3, TODO-94).
- `.claude/skills/mol-metal-r18-math` — R18 math theorems (Wasserstein GF + Bass martingale coupling).
- `.claude/skills/party-twenty-spirit` §3.5 — 0-tolerance honest framing.

---

**END OF Wf-333-V4 CLOSEOUT** — 5-candidate alternate recovery SHIPPED with HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict. The W₂ + PLL top candidate (per WEAK_TO_STRONG §3.5) achieved CV-R² +0.5144 (Δ +0.022 vs PLL-only). The chemprop v2 + PLL surprise achieved CV-R² +0.6814 (best of all 5 candidates). The augmented m6_kappa mean is NOT lifted to ≥ 0.6 in any candidate (Wf-325D HONEST_NEGATIVE_PERSISTENT preserved). m6_kappa ship state in `metrics/by_metric/m6_kappa.json` is unchanged (Wf-325D anchor preserved). Tier-2 acceptance bar does NOT require m6_kappa >= 0.6.
