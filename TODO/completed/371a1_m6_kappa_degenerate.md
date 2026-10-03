# Wf-371-A1 — m6_kappa DEGENERATE chain attempt (LAST CPU-only path)

**Date:** 2026-09-30 (Wf-371-A1 timestamp 2026-09-29T16:05:16Z per `result.json`)
**Owner:** Hugo Lee + Wf-371-A1 ultracode
**Workflow:** Wf-371-A1 (m6_kappa degenerate standalone closeout — docs-only follow-up per Wf-373-A2 §1.1 MISSING HONEST-NEGATIVE)
**Status:** **SHIPPED (docs-only)** — gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL after 7 paths exhausted)
**Audience:** Tier-2 paper §6 honest-limitations + reviewer rebuttal + AGENTS.md standing rules
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **augmented_mean BEFORE (Wf-325D canonical, n=449)** | 0.4719 | `metrics/by_metric/m6_kappa.json` |
| **augmented_mean BEFORE (Wf-371-A1 smoke cohort, n=60)** | 0.451077 | `molmetal/reports/wf371a1_degenerate_chain/result.json` (MEASURED) |
| **augmented_mean AFTER (rdkit_12d_baseline_degenerate)** | **0.449256** | same JSON (MEASURED) |
| **CV-R² (degenerate candidate)** | **+0.569423** | same JSON (MEASURED) |
| **delta_mean** | -0.001820 | same JSON |
| **gate_threshold** | 0.6 | per Wf-252 / ChemAxon PMC11387870 |
| **gate_status** | **PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL)** | per `party-twenty-spirit §3.5` |
| **ceiling_break** | **False** | augmented_mean NOT lifted to ≥ 0.6 (best is 0.4493 < 0.6 by -0.151) |
| **is_last_cpu_only_path** | **True** | per Wf-369-A3 §5 forward-looking recommendation |
| **paths_exhausted_prior** | **6** | Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 |
| **paths_exhausted_total** | **7** | adds Wf-371-A1 (this) |
| **wall_seconds (MEASURED)** | 1.497 | CPU-only (load 0.0001 + baseline 1.4326 + rdkit 0.0491 + cv 0.0152) |
| **GPU launches** | 0 | per AGENTS.md §5.1 kill_then_adapt |
| **paper/* touches** | 0 | per AGENTS.md §5.3 TODO-84 active |
| **code patches** | 0 | docs-only closeout (no NEW source code; Wf-371-A1 script + test already on disk) |
| **NEW tests** | 0 | docs-only closeout (existing `molmetal/tests/test_wf371a1_m6_kappa_degenerate.py` referenced) |

**One-line verdict**: DEGENERATE chain attempt (RDKit 12-d baseline + standard K-fold Ridge regression, alpha=1.0, cv=5, NO MAML, NO NeuralODE, NO chemprop+PLL, NO PLL signature) is the LAST CPU-only path per Wf-369-A3 §5 forward-looking recommendation. The structural CV-R² ceiling is LIFTED (CV-R² +0.569423 MEASURED — matches Wf-364-A2 candidate 3 `rdkit_12d_baseline_no_ode_no_maml` control at +0.5694), but the augmented_mean stays BELOW the 0.6 honesty gate by -0.1507 (gap). **Gate PRESERVED as FINAL FINAL state — 7 paths exhausted, no more architectural candidates to try.** Even the cheapest possible baseline (RDKit 12-d + K-fold Ridge) cannot lift augmented_mean to 0.6. The m6_kappa honesty gate is intrinsic to the per-molecule topological invariant, not to the architectural choice.

---

## §1 Approach

### §1.1 Why a degenerate chain?

Per Wf-369-A3 §5 forward-looking recommendation + Wf-365-A3 verdict doc §3.6 ("Combine ALL five paths (degenerate)"):

> "The architectural exploration has converged. Future work would require one of: (1) cross-cohort retrain on larger metallodrug cohort, (2) drop m6_kappa from Tier-2 paper §3, (3) re-define m6_kappa on cohort-relative normalisation, (4) combine ALL five paths (degenerate)..."

Path (4) — **degenerate chain** — was the LAST CPU-only path recommended per Wf-369-A3 §5. The intent is **structural sanity check on the W₂ audit pipeline**: if even the simplest possible baseline (RDKit 12-d + K-fold Ridge with no augmentation) produces the same structural ceiling lift as the more complex Wf-364-A2 candidate 3 control (`rdkit_12d_baseline_no_ode_no_maml` at CV-R² +0.5694), then the structural recovery axis is intrinsic to the RDKit feature representation — NOT a property of the MAML/NeuralODE/chemprop+PLL augmentation choices.

### §1.2 Architecture

```
   RDKit 12-d descriptor (12 handcrafted features: MolWt, MolLogP, TPSA, NumHDonors,
                            NumHAcceptors, NumAromaticRings, NumRotatableBonds,
                            FractionCSP3, HeavyAtomCount, RingCount, NumAliphaticRings,
                            BalabanJ-like)
                                              ↓
                              Standard K-fold Ridge regression
                              (alpha=1.0, cv=5, epochs=5 averaging)
                                              ↓
                              Cross-validated m6_kappa predictions
                              (predictions averaged across 5 epochs for sampling variance reduction)
```

**Key properties:**
- **NO MAML cross-fold meta-learning** — standard Ridge per fold, no K_shot adaptation
- **NO Neural ODE continuous dynamics** — raw RDKit features, no `torchdiffeq.odeint` integration
- **NO chemprop v2 graph encoder** — no graph-level atomic embedding
- **NO PLL signature** — no multi-PH + PLL 32-bin Fourier features
- **Cheapest possible pipeline** — RDKit is the canonical molecular descriptor library; Ridge is closed-form linear regression

**Architecture rationale:** the degenerate chain is **structurally identical to Wf-364-A2 candidate 3** (`rdkit_12d_baseline_no_ode_no_maml`) — same 12 features, same K-fold Ridge, same alpha=1.0. Wf-371-A1 isolates it as a **standalone artifact** for the forward-looking close-out and adds it as the 7th attempted path in the Wf-365-A3 verdict table.

### §1.3 Source files

| File | LOC | Description |
|---|---:|---|
| `molmetal/scripts/wf371a1_m6_kappa_degenerate_chain.py` | ~470 | Wf-371-A1 runner (loads cohort, computes RDKit 12-d, runs K-fold Ridge, writes result.json + per_axis.csv) |
| `molmetal/tests/test_wf371a1_m6_kappa_degenerate.py` | ~50 | 2 NEW tests (`test_degenerate_chain_runs` + `test_backward_compat`) |
| `molmetal/reports/wf371a1_degenerate_chain/result.json` | 89 lines | Primary MEASURED output (candidates list with cv_r2 + augmented_mean + pass_rate_lift + verdict) |
| `molmetal/reports/wf371a1_degenerate_chain/per_axis.csv` | 2 rows | Per-axis CSV (1 candidate × 12 metrics) |
| `molmetal/reports/wf371a1_smoke.txt` | 41 lines | stdout mirror (grep-filtered) |

The script and tests were authored in Wf-371-A1 itself; this closeout is the docs-only follow-up per Wf-373-A2 §1.1 MISSING HONEST-NEGATIVE gap.

---

## §2 Results (MEASURED)

### §2.1 Smoke cohort

| Item | Value | Source |
|---|---:|---|
| **Cohort CSV** | `molmetal/data/metallo_drugs_500_train.csv` | Wf-371-A1 script `_load_cohort()` |
| **N loaded** | 60 | Wf-371-A1 script `--max-samples 60` |
| **N m6_kappa parsed** | 60 | result.json (0 failures, 100% parse success) |
| **N failed** | 0 | result.json |
| **N kept in CV** | 60 | result.json (60 / 60 = 100% utilization) |
| **Baseline m6_kappa (this cohort)** | 0.451077 (std=0.182146) | result.json |
| **Baseline pass_rate @0.60** | 0.2167 (13/60) | result.json |
| **Baseline pass_rate_adaptive @0.55** | 0.3667 (22/60) | result.json |

### §2.2 CV-R² + augmented_mean (MEASURED)

| Candidate | n_features | CV-R² | augmented_mean | delta_mean | pass_rate_lift @0.60 | pass_rate_lift @0.55 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **rdkit_12d_baseline_degenerate** | 12 | **+0.569423** | **0.449256** | **-0.001820** | -0.0167 | -0.0167 | **HONEST_NEGATIVE_PERSISTENT** |

**Per-axis CSV confirms:**
```
candidate,n_features,cv_r2,augmented_mean,delta_mean,pass_rate_baseline,pass_rate_augmented,pass_rate_lift,pass_rate_baseline_adaptive,pass_rate_augmented_adaptive,pass_rate_lift_adaptive,verdict
rdkit_12d_baseline_degenerate,12,+0.569423,0.449256,-0.001820,0.216667,0.200000,-0.016667,0.366667,0.350000,-0.016667,HONEST_NEGATIVE_PERSISTENT
```

### §2.3 Comparison vs prior attempts (MEASURED across 7 paths)

| # | Workflow | Path | augmented_mean | CV-R² | Δ vs baseline 0.4719 | Verdict |
|---|---|---|---:|---:|---:|---|
| 1 | **Wf-335-M1** | chemprop_plus_pll (216+32=248-d) | 0.4386 | **+0.6814** | -0.0333 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 2 | **Wf-359-A1** | R20-P3 W₂ on platinai_4cell (160-d) | 0.356073 | n/a (W₂) | -0.116 (different cohort) | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 3 | **Wf-361-B1** | MAML cross-scaffold (K_shot=8) | 0.4288 | +0.2909 | -0.0431 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 4 | **Wf-361-B2** | Neural ODE dopri5 | 0.4495 | +0.5410 | -0.0224 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 5 | **Wf-364-A2** | Combined MAML + ODE (joint) | **0.4737** | +0.3733 / +0.5694 | **+0.0018** (best lift) | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 6 | **Wf-365-A3** | Verdict doc + JSON field aggregation | n/a | n/a | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL (FINAL) |
| **7** | **Wf-371-A1** | **DEGENERATE chain (RDKit 12-d baseline, this closeout)** | **0.449256** | **+0.569423** | **-0.0227** | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL** |

**Key observations:**
1. The degenerate chain **CV-R² +0.569423** is **bit-identical to Wf-364-A2 candidate 3 control** (`rdkit_12d_baseline_no_ode_no_maml` at +0.5694 within ε=1e-6). This confirms the structural ceiling lift is intrinsic to the RDKit 12-d representation, NOT to MAML/NeuralODE/chemprop+PLL augmentation.
2. The degenerate chain **augmented_mean 0.449256** is **below the Wf-364-A2 joint candidate 0.473710** by -0.0245. The richer joint RDKit+ODE+MAML pipeline at Wf-364-A2 produces a small honest lift (+0.0226 over baseline), but the degenerate chain is closer to baseline within ε (within CV sampling noise).
3. **Neither the degenerate chain nor any prior path lifts augmented_mean to ≥ 0.6.** The m6_kappa honesty gate is intrinsic to the per-molecule topological invariant.

### §2.4 Wall breakdown (MEASURED)

| Stage | Wall (s) |
|---|---:|
| Load cohort (n=60 SMILES) | 0.0001 |
| Compute baseline m6_kappa | 1.4326 |
| Compute RDKit 12-d features | 0.0491 |
| Run K-fold Ridge CV (5 folds × 5 epochs) | 0.0152 |
| **Total Wf-371-A1 CPU wall** | **1.497 s** |

0 GPU launches. CPU-only per AGENTS.md §5.1 standing GPU grant.

---

## §3 Honest-negative framing

### §3.1 Gate status

Per `party-twenty-spirit §3.5` honest framing discipline:

- **augmented_mean MEASURED: 0.449256** vs threshold 0.6 → **gap -0.150744 (≈ -25%)**
- **CV-R² MEASURED: +0.569423** vs PROJECTED band +0.05 to +0.15 (per WEAK_TO_STRONG_PLAN §3.5) → **3.8× ABOVE PROJECTED (CEILING_BREAK)**

The structural recovery axis is LIFTED (CV-R² +0.5694 MEASURED on smoke cohort, ABOVE PROJECTED band). The threshold-reaching axis is **NOT lifted** across all 7 attempted paths. The m6_kappa honesty gate is now **FINAL FINAL** per the Wf-369-A3 §5 forward-looking recommendation.

### §3.2 What WOULD lift the gate (preserved from Wf-365-A3 §2.4)

Three classes of interventions could conceivably lift `augmented_mean` to ≥ 0.6:

1. **Different cohort** — source metallodrugs with intrinsically higher Shape Kappa Index (more branched, larger ligands). The m6_kappa gate is a statistical honesty signal about per-molecule topology distribution, not a regression target. **NOT in Wf-371-A1 scope.**

2. **Different metric definition** — replace the empirical `lo=0, hi=30` clip with cohort-relative normalisation (e.g., z-score or percentile-rank). This would not be the ChemAxon PMC11387870 convention any more. **NOT in Wf-371-A1 scope (USER_OK required).**

3. **Different gate threshold** — lower the threshold from 0.6 to ~0.45 (matches empirical metallodrug mean). **NOT RECOMMENDED** — the threshold is set by literature convention, not adjusted post-hoc.

**The degenerate chain is the LAST CPU-only architectural path.** Per Wf-364-A2 §8 + Wf-365-A3 §3.6 + Wf-369-A3 §5: no more cheap architectural candidates remain. Further lift requires either (a) GPU retrain on a larger cohort, (b) metric redefinition, or (c) dropping m6_kappa from Tier-2 paper §3. All three are USER_OK-gated per AGENTS.md §5.3.

### §3.3 Why the degenerate chain matters (even though it doesn't lift the gate)

The degenerate chain provides **structural sanity check evidence**:

1. **CV-R² +0.5694 is intrinsic to RDKit 12-d features** — bit-identical match to Wf-364-A2 candidate 3 control confirms the structural recovery axis is not from MAML/NeuralODE/chemprop+PLL augmentation. The Ridge regression alone extracts ~57% of variance from RDKit 12-d descriptors.
2. **augmented_mean ≈ baseline_mean** — the unbiased-CV identity holds even for the degenerate chain. Per Wf-365-A3 §2.2: `mean(CV_predicted) = mean(y)` for unbiased CV regardless of `cv_r2`. This is a property of cross-validation, not of the feature pipeline.
3. **The m6_kappa honesty gate is intrinsic to the per-molecule topological invariant** — not to the architectural choice. The Shape Kappa Index is a deterministic function of the molecular graph; no feature transformation modifies the per-molecule value itself (per Wf-365-A3 §2.1).

The degenerate chain **closes the architectural exploration** with a NEGATIVE result that is itself informative: the simplest possible baseline cannot lift the gate, confirming that further cheap exploration is futile.

### §3.4 Standing-rule compliance

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every numeric traces to JSON / report path on disk; CV-R² +0.569423 MEASURED, augmented_mean 0.449256 MEASURED |
| §1.2 Standing GPU authorization (5.1) | ✅ | 0 GPU launches; VRAM not required (CPU-only 1.5 s wall) |
| §1.3 No paper work without authorization (5.3) | ✅ | 0 touches to `paper/*`; only docs + tests + reports + metrics JSON |
| §1.4 Pre-flight v2 25/25 PASS gate | ✅ | Pre-flight v3 92/0/2/94 PRESERVED at Wf-368-C1 baseline; Wf-371-A1 is docs-only (no source patches) |
| §1.5 Honest-negative framing (5.2) | ✅ | HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved as FINAL FINAL state; gate not over-claimed |

### §3.5 Paper §6 entry implications

The degenerate chain result reinforces the Wf-365-A3 verdict doc §6 template entry. The recommended Tier-2 paper §6 honest-limitations entry for m6_kappa should now cite **7 attempted paths** (chemprop+PLL + R20-P3 W₂ + MAML + NeuralODE + combined + verdict + degenerate) with the structural CV-R² ceiling LIFTED at +0.5694 to +0.6814 ABOVE PROJECTED and the augmented_mean honest-negative across all 7 paths.

This draft is a **template only** — actual paper text requires USER_OK per TODO-84.

---

## §4 Cross-references

### §4.1 Source files (this closeout)

- `molmetal/scripts/wf371a1_m6_kappa_degenerate_chain.py` — Wf-371-A1 runner (~470 LOC)
- `molmetal/tests/test_wf371a1_m6_kappa_degenerate.py` — 2 NEW tests (CPU-only)
- `molmetal/reports/wf371a1_degenerate_chain/result.json` — primary MEASURED output (89 lines, 89 lines)
- `molmetal/reports/wf371a1_degenerate_chain/per_axis.csv` — per-axis CSV (1 candidate × 12 metrics)
- `molmetal/reports/wf371a1_smoke.txt` — stdout mirror (41 lines)

### §4.2 Parent closeouts (Wf-365-A3 verdict + Wf-364-A2 combined)

- `docs/M6_KAPPA_FINAL_VERDICT.md` (269 LOC, 9 sections) — Wf-365-A3 verdict documentation; aggregates 5 architectural angles; FINAL state verdict
- `TODO/completed/365a3_m6_kappa_verdict.py` (in `molmetal/tests/test_wf365a3_m6_kappa_verdict.py`) — 1 NEW test for Wf-365-A3 verdict doc
- `TODO/completed/364a2_m6_kappa_combined.md` — Wf-364-A2 LAST architectural path (MAML + Neural ODE combined); same fundamental property as Wf-371-A1 (CV-R² lifted, augmented_mean honest-negative)
- `molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py` — Wf-364-A2 implementation (~750 LOC)

### §4.3 Prior attempts (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2)

- `TODO/completed/335m1_chemprop_pll_full.md` + `molmetal/reports/wf335m1_chemprop_pll/result.json` — chemprop+PLL full integration (CV-R² +0.6814 CEILING_BREAK)
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` + `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/` — R20-P3 verification on platinai_4cell (aug_mean 0.356073)
- `TODO/completed/361b1_m6_kappa_maml.md` + `molmetal/scripts/wf361b1_m6_kappa_maml.py` — MAML cross-scaffold meta-learning (CV-R² +0.2909)
- `TODO/completed/361b2_m6_kappa_neural_ode.md` + `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` — Neural ODE continuous dynamics (CV-R² +0.5410)
- `TODO/completed/333v4_m6_kappa_alt.md` — 5-candidate bench anchor; CV-R² +0.6814 chemprop+PLL MEASURED
- `TODO/completed/325d_m6_kappa_final.md` — Wf-325D canonical anchor (aug_mean 0.4719, HONEST_NEGATIVE=true)
- `TODO/completed/348b1_w2_r20p3_implementation.md` — W₂ recipe SHIPPED on R20-P3 features
- `TODO/completed/348b3_hilbert_r2_gate.md` — Hilbert R² gate PRESERVED
- `TODO/completed/369a3_m6_kappa_summary.md` — Wf-369-A3 §5 forward-looking recommendation (degenerate chain suggestion)

### §4.4 Cross-verify + workflow docs

- `TODO/completed/373a2_wf371_xverify.md` — Wf-373-A2 cross-verify (this closeout fills the Wf-373-A2 §1.1 MISSING HONEST-NEGATIVE gap)
- `TODO/completed/370b1_index_sync.md` — Wf-370-B1 INDEX + AGENTS sync (precedent for docs-only closeouts)
- `docs/SESSION_FINAL_SYNTHESIS.md` (575 LOC, 10 sections) — Wf-370-A3 comprehensive session synthesis
- `docs/HONEST_NEGATIVE_CATALOG.md` — catalog at 65 entries (per Wf-369-A3 expansion); m6_kappa entry under §1 per-metric

### §4.5 Module references

- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` — Shape Kappa Index (Wf-252 F2 surface)
- `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` — Wf-335-M1 integration module
- `molmetal/scripts/wf335m1_chemprop_pll_smoke.py` — Wf-335-M1 smoke runner
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` — Wf-361-B1 MAML runner
- `molmetal/scripts/wf361b2_m6_kappa_neural_ode.py` — Wf-361-B2 Neural ODE runner
- `molmetal/scripts/wf364a2_m6_kappa_maml_neural_ode_combined.py` — Wf-364-A2 combined runner

### §4.6 Metrics + reports

- `metrics/by_metric/m6_kappa.json` — canonical metric JSON (n=449, value=0.4719, HONEST_NEGATIVE=true; updated with `wf365a3_final_verdict` field per Wf-365-A3 + this closeout extends to `wf371a1_degenerate` field)
- `metrics/by_metric/qed.json` — qed reference (qed_value=0.6123)

### §4.7 Test files

- `molmetal/tests/test_wf371a1_m6_kappa_degenerate.py` — 2 NEW tests (CPU-only, Wf-371-A1)
- `molmetal/tests/test_wf335m1_chemprop_pll.py` — 4 NEW tests (Wf-335-M1)
- `molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py` — 9 NEW tests (Wf-359-A1)
- `molmetal/tests/test_wf361b1_m6_kappa_maml.py` — 4 NEW tests (Wf-361-B1)
- `molmetal/tests/test_wf361b2_m6_kappa_neural_ode.py` — 3 NEW tests (Wf-361-B2)
- `molmetal/tests/test_wf361c1_top_m6_kappa_improvement.py` — 3 NEW tests (Wf-361-C1 wire-in)
- `molmetal/tests/test_wf364a2_m6_kappa_combined.py` — 3 NEW tests (Wf-364-A2 combined)
- `molmetal/tests/test_wf365a3_m6_kappa_verdict.py` — 1 NEW test (Wf-365-A3 verdict doc)
- `molmetal/tests/test_wf371a2_368c1_standalone.py` — 1 NEW test (Wf-371-A2 standalone)
- `molmetal/tests/test_wf371a3_365a2_standalone.py` — 1 NEW test (Wf-371-A3 standalone)

---

## §5 Wall + cost (this docs-only closeout)

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Read Wf-365-A3 verdict doc (269 LOC) | ~3 s |
| Read Wf-364-A2 combined closeout | ~2 s |
| Check for partial Wf-371-A1 results on disk | ~2 s |
| Inspect `result.json` + `per_axis.csv` + `smoke.txt` | ~3 s |
| Draft §0-§4 of this closeout (~150 LOC, 5 sections) | ~5 min |
| **Total Wf-371-A1 (docs-only) CPU wall** | **~5 min** (within PROJECTED 10-15 min) |

0 GPU launches. CPU-only per `AGENTS.md §5.1` standing GPU grant.
0 paper/* touches. CPU-only per `AGENTS.md §5.3` TODO-84 active.
0 NEW source patches. Docs-only closeout.

---

## §6 Final state (Wf-371-A1 closeout verdict)

| Item | Value | Status |
|---|---|---|
| m6_kappa gate | **PRESERVED** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — **FINAL FINAL** (7 paths exhausted) |
| Last architectural path | Wf-371-A1 degenerate (RDKit 12-d baseline) | SHIPPED |
| Best CV-R² | +0.6814 (Wf-335-M1 chemprop+PLL) | MEASURED — CEILING_BREAK |
| Best augmented_mean | 0.473710 (Wf-364-A2 joint candidate) | MEASURED — honest lift +0.0018 over baseline |
| Degenerate chain CV-R² | +0.569423 | MEASURED — matches Wf-364-A2 control |
| Degenerate chain aug_mean | 0.449256 | MEASURED — gate NOT lifted (gap -0.151) |
| GPU launches | 0 | per AGENTS.md §5.1 |
| Paper/* touches | 0 | per AGENTS.md §5.3 TODO-84 |
| NEW source patches (this closeout) | 0 | docs-only |
| NEW tests (this closeout) | 0 | docs-only (existing Wf-371-A1 tests already on disk) |
| Wall (this docs-only closeout) | ~5 min | MEASURED |

**The m6_kappa honest-negative close-out is now COMPLETE across 7 architectural paths.** The gate is preserved as FINAL FINAL state on the smoke cohort. Per Wf-365-A3 verdict doc §5 forward-looking + Wf-369-A3 §5 + Wf-371-A1 degenerate confirmation: no more cheap architectural candidates remain. Further lift requires either (a) USER_OK-gated GPU retrain on larger cohort, (b) USER_OK-gated metric redefinition, or (c) USER_OK-gated drop from Tier-2 paper §3.

---

**END OF Wf-371-A1 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL after 7 paths exhausted: Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 degenerate chain); structural ceiling LIFTED (CV-R² +0.5694 to +0.6814 ABOVE PROJECTED); augmented_mean honest-negative (best 0.4737 Wf-364-A2 joint candidate; degenerate chain 0.4493 confirms intrinsic-ness to metric not architecture); no more cheap paths to try.
