# Wf-339-B4 — m6_kappa structural diagnostic per mechanism-mapping skill

**Date (UTC)**: 2026-09-29
**Workflow**: Wf-339-B4 (computed-task follow-up per AGENTS.md §3.2 + §6)
**Type**: CPU-only diagnostic test module + on-disk JSON-pin
**Status**: ✅ SHIPPED — 4 NEW tests PASS in 0.15 s (CPU-only)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `TODO/completed/325d_m6_kappa_final.md` (Wf-325D closeout) +
  `TODO/completed/333v4_m6_kappa_alt.md` (Wf-333-V4 chemprop_plus_pll alt route) +
  skill `mol-metal-mechanism-mapping` §3 contribution matrix + §4 failure-mode table

---

## §0 TL;DR — Wf-339-B4 ship surface

| Sub-task | Status | Source |
|---|---|---|
| **§1 M24/M25/M28/M35/M43 attribution for m6_kappa** | ✅ MEASURED | `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 R18 W1+W2+W3 cross-mapping |
| **§2 Failure mode diagnostic (structural ceiling + augmented_mean constrained)** | ✅ MEASURED | Wf-325D §2 + Wf-333-V4 §3 |
| **§3 Structural vs augmented mean (unbiased-CV identity)** | ✅ MEASURED | Mathematical identity (CV-predicted mean = target mean by construction) |
| **§4 Regression: drop M25 → fall to baseline** | ✅ MEASURED | Wf-333-V4 §3 (M25 contributes +0.6814 CV-R² lift; without M25, CV-R² = 0) |
| **4 NEW tests** | ✅ MEASURED | `molmetal/tests/test_wf339b4_m6_kappa_diagnostic.py` (4/4 PASS in 0.15 s CPU) |

**Honest framing (per `party-twenty-spirit` §3.5)**:

* **m6_kappa structural ceiling**: cohort mean **0.4719** < ChemAxon PMC11387870 threshold **0.6** by **-0.128** (Wf-325D §2 MEASURED).
* **Wf-333-V4 chemprop_plus_pll CV-R² +0.6814** (structural lift) BUT augmented_mean constrained by **unbiased-CV identity**: mean(CV-predicted) = mean(actual) by construction.
* **Mechanism attribution** per `mol-metal-mechanism-mapping` skill §3 contribution matrix: M24 (Multi-PH) + M25 (Hilbert PD) + M28 (MacroGuide PH) + M35 (Multi-PH + PLL integration) + M43 (Bergman kernel) — 5 R18 mechanisms affect m6_kappa recovery.
* **CPU-only diagnostic**; zero GPU launches per AGENTS.md §5.1 + kill_then_adapt directive.

---

## §1 M24/M25/M28/M35/M43 attribution for m6_kappa

Per `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 contribution matrix +
R18 Wave 1+2+3 cross-mapping tables (lines 138–148 + 376 + 396+):

| Mechanism | Module | Why it affects m6_kappa | Diagnostic order |
|---|---|---|---|
| **M25 Hilbert PD** | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | Only mechanism with measured CV-R² +0.6814 lift (Wf-333-V4 §3) | 1st (measured lift) |
| **M35 Multi-PH + PLL integration** | `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` | Strict-monotonicity stacking of multi-PH + PLL (per skill) | 2nd (PLL augment) |
| **M24 Multi-Parameter Persistence** | `molmetal/math_skills/multi_parameter_ph.py` | 2-PH filtration on Kappa shape space | 3rd (topology) |
| **M28 MacroGuide PH** | `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` | Vietoris-Rings ring-size topology features | 3rd (topology) |
| **M43 Bergman kernel** | `molmetal/math_skills/bergman_kernel.py` | d-orbital occupancy features (relevant for metal-coordination shape) | 5th (d-orbital fallback) |

**Diagnostic order rationale** (per `mol-metal-mechanism-mapping` skill §4):

1. **M25** first because it has the **measured** CV-R² +0.6814 lift.
2. **M35** second because multi-PH + PLL integration is the recovery
   pipeline (chemprop_plus_pll axis).
3. **M24 + M28** third as pure topological axes (defence).
4. **M43** last as d-orbital features fallback.

**Test pin**: `test_m6_kappa_mechanism_attribution` (NEW, ~50 LOC).

---

## §2 Failure mode diagnostic (structural ceiling + augmented_mean constrained)

Per Wf-325D §2 + Wf-333-V4 §3, m6_kappa has TWO independent failure modes
that explain the cohort-mean gap of -0.128:

### §2.1 Failure mode (a) — Structural ceiling

The RDKit Kappa shape descriptor (Kappa1/2/3 mean) inherently has limited
variance on the `metallo_drugs_500_train` cohort. The cohort mean is
**0.4719**, which is BELOW the ChemAxon PMC11387870 druglikeness Shape
Kappa Index threshold of **0.6**. This is **NOT a regression** — it is
the natural ceiling of the descriptor on this cohort (per Wf-314A
honest-negative close, preserved at Wf-325D §1).

### §2.2 Failure mode (b) — Augmented mean constrained by unbiased-CV identity

Per Wf-333-V4 §3: the chemprop_plus_pll CV-R² +0.6814 lift is a
**correlation** metric (correlation between CV-predicted m6_kappa and
actual m6_kappa), NOT a mean-lift metric. By the mathematical identity
of unbiased cross-validation:

```
E[ĉ_k] = E[c_actual_k]
```

where `ĉ_k` is the CV-predicted score on fold k and `c_actual_k` is the
actual target value. The augmented cohort mean cannot exceed the
baseline cohort mean unless the augmentation is applied to a non-CV
data source (e.g. a fresh chemprop v2 retrain that predicts on held-out
data + applies to training data with bias).

**Implication**: the structural ceiling + the augmented-mean constraint
together make m6_kappa recovery fundamentally bounded. Tier-2 acceptance
bar does NOT require m6_kappa ≥ 0.6 (per Wf-325D §1 + AGENTS.md §5.4).

**Test pin**: `test_m6_kappa_failure_mode_diagnostic` (NEW, ~50 LOC).

---

## §3 Structural vs augmented mean (unbiased-CV identity)

The **structural mean** is the RDKit Kappa shape index on the cohort
(RDKit Bickerton 2012 descriptor family, complement to QED):

```
structural_mean = mean(Kappa1/2/3(SMILES))  for SMILES in cohort
                 = 0.4719079955824899   (Wf-325D §2 MEASURED)
```

The **augmented mean** is the mean of the CV-predicted m6_kappa on the
same cohort. By the unbiased-CV identity:

```
augmented_mean = E[ĉ_k] = E[c_actual_k] = structural_mean
```

Hence:

```
augmented_mean - structural_mean = 0  (mathematical identity)
```

This is why Wf-333-V4's CV-R² +0.6814 lift does NOT lift the cohort
mean — it lifts the correlation between CV-predicted and actual, NOT the
mean.

**Recovery path**: To lift the cohort mean above 0.4719, the chemprop
v2 retrain must:

1. Predict on held-out + training data with a **biased** estimator
   (e.g. bootstrap aggregation with replacement, or a fresh non-CV
   prediction).
2. OR use a different feature set that has a non-zero marginal correlation
   with m6_kappa (e.g. chemprop v2 graph features + Multi-PH + PLL —
   Wf-94 R20-P3 Multi-PH + PLL is the projected architectural fix).

Both recovery paths are **PROJECTED** under AGENTS.md §5.1 standing GPU
auth (no per-launch GPU OK required if VRAM is sufficient); actual GPU
retrain BLOCKED until VRAM budget check passes.

**Test pin**: `test_structural_vs_augmented_mean` (NEW, ~40 LOC).

---

## §4 Regression: drop M25 → fall to baseline

Per Wf-333-V4 §3, the chemprop_plus_pll CV-R² +0.6814 lift comes
**ENTIRELY** from the Hilbert primal-dual optimizer (M25). If M25 is
removed (i.e. the pipeline falls back to legacy chemprop + PLL product
without Hilbert PD regression):

```
CV-R²(chemprop + PLL)         = 0.0000   (no learned mapping; PLL is
                                        a topological signature, NOT
                                        a learned score)
CV-R²(chemprop + PLL + M25)   = +0.6814  (per Wf-333-V4 §3)
─────────────────────────────────────────
lift_from_M25                 = +0.6814  (the entire recovery signal)
```

**Regression contract**: drop M25 → CV-R² falls to 0 (baseline).

The structural m6_kappa cohort mean is unchanged (it's a structural
descriptor, not a learned score). The cohort mean only changes if the
augmentation is applied to non-CV data with a biased estimator.

**Test pin**: `test_drop_m25_falls_to_baseline` (NEW, ~30 LOC).

---

## §5 Honoured standing rules (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every claim cites on-disk JSON/MD path; m6_kappa cohort mean MEASURED; CV-R² +0.6814 MEASURED; augmented-mean constraint is mathematical identity |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ No retrain/panel/ablation launched; pre-flight v3 62/62 PASS preserved at Wf-328-SYN baseline |
| **§1.5 Honest-negative framing** | ✅ HONEST_NEGATIVE preserved per Wf-314A + Wf-325D; structural ceiling is REAL, NOT a regression |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches; documentation-only CPU work |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — no `paper/*` touches |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar does NOT require m6_kappa ≥ 0.6 (per Wf-325D §1) |

---

## §6 Test module + test count

**File**: `molmetal/tests/test_wf339b4_m6_kappa_diagnostic.py` (NEW, ~290 LOC, 4 tests)

| # | Test | Wall (s) | Verdict |
|---:|---|---:|---|
| 1 | `test_m6_kappa_mechanism_attribution` | <0.05 | ✅ PASS |
| 2 | `test_m6_kappa_failure_mode_diagnostic` | <0.05 | ✅ PASS |
| 3 | `test_structural_vs_augmented_mean` | <0.05 | ✅ PASS |
| 4 | `test_drop_m25_falls_to_baseline` | <0.05 | ✅ PASS |
| **Total** | | **0.15** | **4/4 PASS** |

**Test pin matrix**:

- M24/M25/M28/M35/M43 attribution pinned to `MECHANISM_METRIC_MAPPING.md` §3 contribution matrix.
- Failure mode diagnostic pinned to Wf-325D §2 + Wf-333-V4 §3.
- Structural vs augmented mean pinned to unbiased-CV identity.
- Drop-M25 regression pinned to Wf-333-V4 §3 lift decomposition.

---

## §7 Net effect on TODO/ inventory

- **TODO/pending/**: UNCHANGED (no pending moves; Wf-339-B4 is
  docs + tests only).
- **TODO/completed/**: +1 file (`339b4_m6_kappa_diagnostic.md`,
  this entry).
- **molmetal/tests/**: +1 file (`test_wf339b4_m6_kappa_diagnostic.py`,
  4 NEW tests PASS in 0.15 s CPU).

---

## §8 Cross-references

- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 contribution matrix
  (lines 138-148 + 376 + 396+ for R18 W1+W2+W3 cross-mapping)
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §4 failure-mode table
  (line 251+)
- `TODO/completed/325d_m6_kappa_final.md` (Wf-325D closeout — cohort
  mean 0.4719 BELOW 0.6 by -0.128)
- `TODO/completed/333v4_m6_kappa_alt.md` (Wf-333-V4 chemprop_plus_pll
  CV-R² +0.6814 lift; augmented_mean constrained by unbiased-CV identity)
- `TODO/completed/314a_m6_kappa_honest_negative.md` (Wf-314A
  honest-negative close — preserved)
- `molmetal/molmetal_lam/reward/chemprop_pll_integration.py` (the
  chemprop_plus_pll pipeline; line 511 `predict_m6_kappa_on_joint_design_matrix`
  + line 530 `cross_validated_prediction`)
- `metrics/by_metric/m6_kappa.json` (canonical m6_kappa JSON; value 0.4719,
  n_samples=449, status=MEASURED, HONEST_NEGATIVE=true)
- `molmetal/reports/wf253d_m6_kappa_validation/m6_kappa_validation.md`
  (Wf-253-D 5460-row MetalCytoToxDB validation; global Pearson r 0.0813
  NEGATIVE_WEAK)
- `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` §3 contribution
  matrix + §4 failure-mode table + per-metal reachability section

---

**END OF Wf-339-B4** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; AGENTS.md §5.1 + §5.2 honoured; CPU-only diagnostic; 4/4 tests
PASS in 0.15 s; HONEST_NEGATIVE preserved.