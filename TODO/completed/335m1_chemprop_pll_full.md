# Wf-335-M1 — m6_kappa chemprop_plus_pll full integration

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Parent:** Wf-333-V4 (m6_kappa alternate-recovery bench) / Wf-331-R3 (Multi-PH/PLL integration) / Wf-49 P2 (chemprop v2 graph features) / WEAK_TO_STRONG §3.5
**Status:** SHIPPED — pre-flight anchor `p3_m6_kappa_chemprop_pll_integration_documented` PASS

---

## §0 TL;DR

| Metric / claim | Value | Source |
|---|---:|---|
| **chemprop_plus_pll CV-R² (60 SMILES)** | **+0.6814** | `molmetal/reports/wf335m1_chemprop_pll/result.json` (MEASURED) |
| Wf-333-V4 MEASURED CV-R² anchor | +0.6814 | `molmetal/reports/wf333v4_m6_kappa/result.json` |
| chemprop_plus_pll rank vs 5 candidates | 1st (best) | Wf-333-V4 §2.2 |
| augmented_mean | 0.4386 | `molmetal/reports/wf335m1_chemprop_pll/result.json` |
| m6_kappa pass_rate_lift @0.60 strict | +0.0167 | MEASURED |
| m6_kappa pass_rate_lift @0.55 adaptive | -0.0333 | MEASURED |
| Wf-325D baseline anchor | 0.4719 | `metrics/by_metric/m6_kappa.json` |
| Wall (60 SMILES smoke) | 50.81 s | CPU-only |
| 4 NEW tests in `test_wf335m1_chemprop_pll.py` | 4 / 4 PASS in 1.49 s | MEASURED |
| Aggregate verdict | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** | structural ceiling lifted via +0.6814 CV-R²; augmented_mean constrained ~baseline by unbiased-CV identity |

**One-line verdict:** The Wf-333-V4 chemprop_plus_pll axis is promoted from a bench candidate to a **first-class integration module**. The integration reproduces the Wf-333-V4 MEASURED CV-R² +0.6814 exactly on the 60-SMILES smoke cohort (bit-identical to Wf-333-V4 result.json) and exposes a clean wire-in API (`ChempropPLLFeatureExtractor` + `ChempropPLLRidgeModel` + `wire_chemprop_pll_recovery`). The structural recovery signal is **above the Wf-333-V4 PROJECTED band** (+0.05 to +0.15); the augmented_mean is constrained to 0.4386 by the unbiased-CV identity and remains BELOW the 0.6 threshold. The Wf-325D HONEST_NEGATIVE_PERSISTENT ship state for `metrics/by_metric/m6_kappa.json` is preserved unchanged.

---

## §1 Honest framing (`party-twenty-spirit` §3.5)

Per `AGENTS.md` §1 standing rules:

1. **MEASURED-vs-PROJECTED tag discipline** — every numeric value below is MEASURED, traced to `molmetal/reports/wf335m1_chemprop_pll/result.json` or `molmetal/reports/wf333v4_m6_kappa/result.json` or `metrics/by_metric/m6_kappa.json`. Zero fabricated / aspirational numbers.
2. **"Kill then adapt" GPU directive** — `mol-metal-gpu-execution` §0. Wf-335-M1 is CPU-only (RDKit-native chemprop fallback + sklearn Ridge); **no GPU work was performed**.
3. **No paper work without authorization** — TODO-84 active; zero touches to `paper/*`.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207) — `p3_m6_kappa_chemprop_pll_integration_documented` is the new Wf-335-M1 pre-flight anchor (documentation gate only — no new gate logic added).
5. **Honest-negative framing** — the structural recovery signal is **above PROJECTED**; the threshold-reaching recovery signal remains **below PROJECTED** (augmented_mean constrained by unbiased-CV identity).

---

## §2 What Wf-335-M1 ships

### §2.1 Module — `molmetal/molmetal_lam/reward/chemprop_pll_integration.py`

A new 460-LOC module promoting the Wf-333-V4 chemprop_plus_pll candidate from inline bench code to a **first-class API surface**:

| Class / function | Purpose |
|---|---|
| `ChempropPLLFeatureExtractor` | Joint chemprop v2 graph + PLL signature extractor with `compute_batch` returning `(chemprop_feat, pll_feat, joint)` |
| `ChempropPLLRidgeModel` | sklearn-style Ridge wrapper with `fit` / `predict` / `cv_predict` / `cv_r2` / `pass_rate_at_threshold` / `score` |
| `ChempropPLLRecoveryReport` | Dataclass carrying the JSON contract (CV-R² + augmented mean + pass-rates + verdict + wall) |
| `wire_chemprop_pll_recovery` | Single-call wire-in to the m6_kappa pipeline |
| `write_recovery_report` | JSON serialiser with Wf-333-V4 / Wf-325D contract tags |
| Constants | `WF333V4_MEASURED_CV_R2 = +0.6814`, `WF325D_MEASURED_BASELINE = 0.4719`, `M6_KAPPA_THRESHOLD = 0.6`, `M6_KAPPA_THRESHOLD_ADAPTIVE = 0.55` |

**Architectural choice:** `ChempropPLLFeatureExtractor` defaults to `use_chemprop_fallback=True` (RDKit-native atom-pooled 216-d fingerprint), preserving bit-identity to the Wf-333-V4 fallback. Set `use_chemprop_fallback=False` to use the full 258-d chemprop v2 graph (atom + bond, 3 pools).

### §2.2 Smoke script — `molmetal/scripts/wf335m1_chemprop_pll_smoke.py`

CPU-only smoke runner that:
1. Loads the metallo cohort CSV (default cap 60 SMILES).
2. Computes m6_kappa baseline via `m6_kappa_from_smiles`.
3. Runs the joint chemprop+PLL recovery wire-in.
4. Writes `molmetal/reports/wf335m1_chemprop_pll/result.json` with the JSON contract.
5. Emits a 5-line plain-text OUTPUT (per `AGENTS.md` §8.3).

### §2.3 Tests — `molmetal/tests/test_wf335m1_chemprop_pll.py`

4 NEW tests (all CPU-only, deterministic, < 2 s combined):

1. `test_chemprop_v2_features_extraction` — chemprop v2 RDKit-native fallback produces well-formed (n, 216) graph fingerprint, atom vocab + pool contract.
2. `test_pll_signature_extraction` — PLL signature (delegated to `MultiPHPLLFeatureExtractor`) produces well-formed (n, 32) matrix; joint design matrix contract (chemprop leading, PLL trailing).
3. `test_combined_ridge_model` — `ChempropPLLRidgeModel` fit + predict + cross-validate; CV-R² finite; augmented mean constrained ~ baseline by unbiased-CV identity; pass-rate contract (adaptive ≥ strict).
4. `test_cv_r2_0_6814_verification` — End-to-end wire-in via `wire_chemprop_pll_recovery`; Wf-333-V4 MEASURED CV-R² +0.6814 anchor mirror; JSON contract roundtrip.

### §2.4 Report — `molmetal/reports/wf335m1_chemprop_pll/result.json`

JSON report carrying the Wf-333-V4 contract tags:
- `wf_333_v4_best_cv_r2_measured: 0.6814` — upper-bound anchor
- `recovery.cv_r2: 0.6814` — MEASURED on the 60-SMILES smoke cohort
- `recovery.verdict: HONEST_NEGATIVE_PERSISTENT` — augmented_mean 0.4386 < 0.6
- `honest_framing` block — CPU-only + no retrain + no paper touch

---

## §3 Wf-335-M1 vs Wf-333-V4 — bit-identity verification

The Wf-335-M1 integration reproduces the Wf-333-V4 MEASURED CV-R² +0.6814 **exactly** on the 60-SMILES smoke cohort:

| Axis | Wf-333-V4 | Wf-335-M1 | Δ |
|---|---:|---:|---:|
| `chemprop_plus_pll` CV-R² | +0.681405 | +0.681405 | 0.0 (bit-identical) |
| `chemprop_plus_pll` augmented_mean | 0.4386 | 0.4386 | 0.0 (bit-identical) |
| `chemprop_plus_pll` pass_rate_lift @0.60 | +0.0167 | +0.0167 | 0.0 (bit-identical) |
| chemprop_dim | 216 (atom-only) | 216 (atom-only) | 0 |
| pll_dim | 32 | 32 | 0 |
| joint_dim | 248 | 248 | 0 |
| Ridge alpha | 1.0 | 1.0 | 0 |
| CV folds | 5 | 5 | 0 |
| Verdict | HONEST_NEGATIVE_PERSISTENT | HONEST_NEGATIVE_PERSISTENT | 0 |

**Verification:** The Wf-335-M1 integration is a **promotion** (WEAK_TO_STRONG §3.5), not a redesign. The chemprop+PLL joint design matrix (216 + 32 = 248-d), Ridge alpha (1.0), CV folds (5), and verdict contract are all preserved bit-identically.

---

## §4 Pre-flight gate (new for Wf-335-M1)

The canonical tag is preserved on the JSON, mirroring the Wf-308B / Wf-312C / Wf-314A / Wf-315A / Wf-325D / Wf-333-V4 contract:

```json
{
  "wf": "Wf-335-M1",
  "no_gpu_launch": true,
  "no_retrain": true,
  "cpu_only": true,
  "wf_333_v4_best_cv_r2_measured": 0.6814054457018517,
  "wf_325d_baseline_measured": 0.4719079955824899,
  "recovery": {
    "candidate": "chemprop_plus_pll",
    "n_samples": 60,
    "n_features": 248,
    "cv_r2": 0.681405,
    "augmented_mean": 0.438605,
    "delta_mean": -0.012472,
    "pass_rate_baseline": 0.216667,
    "pass_rate_augmented": 0.233333,
    "pass_rate_lift": 0.016667,
    "verdict": "HONEST_NEGATIVE_PERSISTENT",
    "wall_seconds": 50.8144,
    "ridge_alpha": 1.0,
    "cv_folds": 5
  }
}
```

Pre-flight anchors (per Wf-335-M1 ship):
- `p3_m6_kappa_chemprop_pll_integration_documented` PASS (Wf-335-M1 ship — this doc).
- `p3_m6_kappa_alt_recovery_documented` PASS (Wf-333-V4 ship — preserved).
- `p3_m6_kappa_closeout_documented` PASS (Wf-325D ship — preserved).
- `p3_admet_m6_kappa_honest_negative` PASS (Wf-309A / Wf-314A ship — preserved).

---

## §5 Test results — MEASURED

Source: `PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf335m1_chemprop_pll.py -v` (2026-09-29).

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf335m1_chemprop_pll.py::test_chemprop_v2_features_extraction PASSED [ 25%]
molmetal/tests/test_wf335m1_chemprop_pll.py::test_pll_signature_extraction PASSED [ 50%]
molmetal/tests/test_wf335m1_chemprop_pll.py::test_combined_ridge_model PASSED [ 75%]
molmetal/tests/test_wf335m1_chemprop_pll.py::test_cv_r2_0_6814_verification PASSED [100%]

========================= 4 passed, 1 warning in 1.49s =========================
```

**Regression check** (existing Wf-333-V4 tests still pass):
```
molmetal/tests/test_wf333v4_m6_kappa_alt.py: 4 passed in 2.29s
```

**Total**: 8 tests pass (4 NEW Wf-335-M1 + 4 existing Wf-333-V4).

---

## §6 Smoke run — MEASURED

Source: `PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/wf335m1_chemprop_pll_smoke.py --output-dir molmetal/reports/wf335m1_chemprop_pll/` (2026-09-29).

```
[Wf-335-M1] Loading cohort from /home/hugo/codes/mol-metal/molmetal/data/metallo_drugs_500_train.csv (max 60) ...
[Wf-335-M1] Loaded 60 SMILES in 0.00s
[Wf-335-M1] Computing m6_kappa baseline on cohort ...
[Wf-335-M1] m6_kappa baseline: n_parsed=60, n_failed=0, mean=0.4511, (0.12s)
[Wf-335-M1] Running chemprop+PLL recovery wire-in ...
[Wf-335-M1] Extractor: chemprop_dim=216, pll_dim=32, joint_dim=248
[Wf-335-M1] chemprop+PLL wire-in: 51.72s

================================================================================
Wf-335-M1 SUMMARY
  n_samples          : 60
  n_features         : 248
  cv_r2              : +0.6814 (Wf-333-V4 MEASURED upper bound = +0.6814)
  augmented_mean     : 0.4386 (baseline = 0.4719)
  delta_mean         : -0.0125
  pass_rate_strict   : 0.2167 → 0.2333 (lift=+0.0167)
  pass_rate_adaptive : 0.3667 → 0.3333 (lift=-0.0333)
  verdict            : HONEST_NEGATIVE_PERSISTENT
  wall_seconds       : 50.81
================================================================================
[Wf-335-M1] Wrote molmetal/reports/wf335m1_chemprop_pll/result.json
```

---

## §7 Recovery axis — summary

| Axis | Top candidate | MEASURED (Wf-335-M1) | Wf-333-V4 MEASURED | PROJECTED (WEAK_TO_STRONG §3.5) | Status |
|---|---|---:|---:|---:|---|
| **Structural (CV-R² > 0)** | `chemprop_plus_pll` | **+0.6814** | +0.6814 | +0.05 to +0.15 | **ABOVE PROJECTED** |
| chemprop+PLL rank (of 5 candidates) | n/a | 1st (best) | 1st (best) | n/a | MEASURED |
| chemprop+PLL bit-identity to Wf-333-V4 | n/a | 0.0 | (anchor) | n/a | BIT-IDENTICAL |
| Threshold-reaching (pass_rate_lift @0.60) | `chemprop_plus_pll` | +0.0167 | +0.0167 | +0.01 to +0.05 | MEASURED |
| Threshold-reaching (pass_rate_lift @0.55) | `multi_ph_plus_w2` | -0.0333 | -0.0333 | n/a | NEGATIVE |
| Augmented mean lift to ≥ 0.6 | none | 0.4386 | 0.4386 | +0.05 to +0.15 | NOT LIFTED |
| Aggregate verdict | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL | HONEST_NEGATIVE_PARTIAL_STRUCTURAL | n/a | MEASURED |

---

## §8 What Wf-335-M1 does NOT change

1. **`metrics/by_metric/m6_kappa.json`** is NOT updated. The Wf-325D MEASURED baseline (0.4719) and HONEST_NEGATIVE tag are preserved unchanged. The Wf-335-M1 chemprop+PLL CV-R² +0.6814 lives in `molmetal/reports/wf335m1_chemprop_pll/result.json` (a separate report directory).
2. **Pre-flight gate `p3_m6_kappa_closeout_documented`** is unchanged (still PASS from Wf-325D).
3. **Pre-flight gate `p3_m6_kappa_alt_recovery_documented`** is unchanged (still PASS from Wf-333-V4).
4. **No new pre-flight gate logic added** by Wf-335-M1 (the chemprop_pll_integration gate is documentation-only).
5. **No paper `paper/*` touches** (TODO-84 active).
6. **No GPU launches** (USER_OK-gated per `mol-metal-gpu-execution` §0).

---

## §9 References

- `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` — Wf-335-M1 full integration module (NEW).
- `molmetal/scripts/wf335m1_chemprop_pll_smoke.py` — Wf-335-M1 smoke runner (NEW).
- `molmetal/tests/test_wf335m1_chemprop_pll.py` — 4 NEW tests (NEW).
- `molmetal/reports/wf335m1_chemprop_pll/result.json` — full MEASURED JSON report (NEW).
- `TODO/completed/335m1_chemprop_pll_full.md` — this document (NEW).
- `molmetal/scripts/wf333v4_m6_kappa_alt.py` — Wf-333-V4 5-candidate bench (anchor source).
- `molmetal/tests/test_wf333v4_m6_kappa_alt.py` — Wf-333-V4 4 tests (anchor regression).
- `molmetal/reports/wf333v4_m6_kappa/result.json` — Wf-333-V4 MEASURED CV-R² +0.6814 anchor.
- `molmetal/scripts/wf331r5_r2_w2_chemprop.py` — W₂ + chemprop v2 R² ceiling recovery (Wf-331-R5).
- `molmetal/scripts/wf329w5_w2_chemprop_smoke.py` — W₂ + chemprop v2 alignment (Wf-329W5).
- `molmetal/scripts/wf331r3_m6_kappa_multiph_pll.py` — Multi-PH/PLL m6_kappa integration (Wf-331-R3 anchor).
- `molmetal/molmetal_lam/reward/multiph_pll_feature.py` — `MultiPHPLLFeatureExtractor` (Wf-329-W4 shared).
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2 surface).
- `metrics/by_metric/m6_kappa.json` — canonical Wf-325D MEASURED baseline (0.4719, n=449) + HONEST_NEGATIVE preserved.
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D honest-negative closeout (canonical anchor).
- `TODO/completed/333v4_m6_kappa_alt.md` — Wf-333-V4 alternate-recovery closeout.
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.5 — Wasserstein GF + Bass martingale coupling (R18-P3).
- `AGENTS.md` §1 — standing rules (MEASURED-vs-PROJECTED + honest-negative + no paper work).
- `AGENTS.md` §5.2 — honest framing mandate.
- `AGENTS.md` §5.3 — no paper work without authorization (TODO-84 active).
- `AGENTS.md` §5.1 — Tier-2 paper target; no wet-lab planning.
- `.claude/skills/mol-metal-multiph-pll` — Multi-PH + PLL + Score-Rep MC skill (R20-P3, TODO-94).
- `.claude/skills/mol-metal-r18-math` — R18 math theorems (Wasserstein GF + Bass martingale coupling).
- `.claude/skills/party-twenty-spirit` §3.5 — 0-tolerance honest framing.

---

**END OF Wf-335-M1 CLOSEOUT** — chemprop_plus_pll full integration SHIPPED with bit-identical Wf-333-V4 CV-R² +0.6814 anchor. The chemprop+PLL axis is promoted to a first-class integration module (`ChempropPLLFeatureExtractor` + `ChempropPLLRidgeModel` + `wire_chemprop_pll_recovery`) with 4 NEW tests (4 / 4 PASS in 1.49 s) and a JSON contract that mirrors the Wf-333-V4 / Wf-325D anchors. The structural ceiling is lifted (CV-R² +0.6814 ABOVE the WEAK_TO_STRONG §3.5 PROJECTED band +0.05 to +0.15), but the augmented m6_kappa mean is constrained to 0.4386 < 0.6 by the unbiased-CV identity. The Wf-325D HONEST_NEGATIVE_PERSISTENT ship state for `metrics/by_metric/m6_kappa.json` is preserved unchanged. m6_kappa Tier-2 acceptance is computational-only and does not require m6_kappa >= 0.6 (per AGENTS.md §5.4).
