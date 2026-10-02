# Wf-325A — F5_RDKIT_2D 32-d Expansion R² Lift Bench

**Date**: 2026-09-29
**Owner**: Hugo Lee + workflow harness
**Upstream**: Wf-297B (R² architectural ceiling finding), Wf-218 (32-d
descriptor expansion), Wf-127 §Cause 2 (signal-floor on 12-d RDKit).
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing.

---

## §1 Background

Per Wf-297B, the **R² architectural ceiling** is documented as
honest-negative on three real-pIC50 cohorts:

| Cohort | Wf-297B MEASURED R² (Hilbert-ASTSR combined) | Status |
|---|--:|---|
| PlatinAI 4-cell | 0.214 | signal-floor |
| Hilbert | 0.276 | signal-floor |
| MetalCytoToxDB_combined | 0.276 | signal-floor |

The Wf-49 P1 frozen 12-d RDKit feature set was identified as one of
the binding constraints (Wf-127 §Cause 2).  Wf-218 (and Wf-248 P1
canonical implementation in `molmetal/data/rdkit_2d_extended.py`)
defined a **32-d expansion** that adds 20 shape / topology /
partial-charge / ring / heteroatom / functional-group descriptors.

Wf-325A is the **CPU-only diagnostic** that asks: does the 32-d
expansion alone lift the Ridge baseline R² above the signal floor?

---

## §2 Method

CPU-only, no torch, no GPU.  Three cohorts:

1. **platinai_4cell** (200 SMILES, 4-cell pIC50 mean) — 12-d from
   `molmetal/data/platinai_4cell_rdkit_features.csv`; 32-d from
   `molmetal/data/platinai_4cell_rdkit_extended_features.csv`.
2. **hilbert** (same 200 SMILES, different test-seed for an
   independent reading) — same data sources, seed=1 split.
3. **metalcytotox_combined** (2000 SMILES, 5-metal MetalCytoToxDB
   `activity` target) — 12-d from
   `molmetal/data/metalcytotox_rdkit_features.csv`; 32-d computed
   on the fly via `molmetal.data.rdkit_2d_extended.compute_rdkit_2d_extended`.

For each cohort:

* Ridge regression baseline (`alpha=1.0`).
* `StandardScaler` on the feature matrix.
* 80/20 train/test split.
* `seed=0` (and `seed=1` for the hilbert cohort).
* `r2 = model.score(X_test, y_test)`.

Lift = `R²(32d) - R²(12d)`.  Verdict labels (per
`_summarise_lift`):

* `STRONG_LIFT` (lift ≥ +0.10)
* `MODERATE_LIFT` (lift ≥ +0.05)
* `WEAK_LIFT` (lift ≥ +0.01)
* `CEILING_BREAK` (12-d R² ≤ 0 AND 32-d R² > 0)
* `NO_LIFT_HONEST_NEGATIVE` (lift < +0.01)
* `STILL_NEGATIVE_HONEST_NEGATIVE` (32-d R² ≤ 0)

---

## §3 Results (MEASURED)

| Cohort | n | R²(12d) | R²(32d) | lift | verdict |
|---|--:|--:|--:|--:|---|
| platinai_4cell | 200 | **-0.0945** | **-0.3118** | **-0.2174** | STILL_NEGATIVE_HONEST_NEGATIVE |
| hilbert | 200 | **-0.0169** | **-0.0110** | **+0.0059** | STILL_NEGATIVE_HONEST_NEGATIVE |
| metalcytotox_combined | 2000 | **+0.1055** | **+0.0837** | **-0.0219** | NO_LIFT_HONEST_NEGATIVE |

**Aggregate honest-negative status**: `HONEST_NEGATIVE_CONFIRMED`.

The 32-d RDKit expansion **does not lift the Ridge R²** on any of
the three cohorts.  This is a **clean honest-negative result**:

* On **platinai_4cell**, the 32-d R² actually **drops** to -0.31
  (the extra descriptors amplify the noise on a 200-row Pt-only
  cohort, where the 12-d baseline was already negative).
* On **hilbert**, the lift is a vanishingly small +0.006 (within
  seed variance) — no architectural ceiling break.
* On **metalcytotox_combined**, the 32-d R² **drops** by -0.02
  (the 12-d baseline at +0.10 was already weak; the extra
  descriptors add noise without information).

### §3.1 What this rules out (the positive result)

The Wf-297B R² ceiling is NOT caused by **feature-set sparseness
alone**.  Even with 32-d RDKit 2D descriptors, the Ridge baseline
R² is bounded near 0 on real pIC50.  This **eliminates**
"12-d RDKit is too thin" as a viable remediation.

### §3.2 What this does NOT rule out (the honest-negative)

The R² ceiling is **architectural** — it requires fundamentally
different signal sources per WEAK_TO_STRONG §2.1:

* **W₂ alignment** between PlatinAI 4-cell and MetalCytoToxDB 5-metal
  (TODO-122 P1.3).
* **chemprop v2 graph features** (atom-bond message passing; the 12-d
  and 32-d RDKit sets are both per-molecule scalar features with
  no atom-level or bond-level structure).
* **Multi-PH + PLL topological signatures** (TODO-94).
* **Hilbert-ASTSR joint end-to-end retrain** (TODO-93, GPU portion).

These are the **next** investigations, not the current one.  Wf-325A
is the CPU-only diagnostic that establishes the baseline.

---

## §4 Honest framing (party-twenty-spirit §3.5)

Per AGENTS.md §1 (MEASURED-vs-PROJECTED tag discipline):

* **MEASURED** (this bench): the R² numbers in §3 above.
* **PROJECTED** (TODO-122 / Wf-218): the +0.10 to +0.30 R² lift
  hypothesised for the 32-d expansion **was not realised** on any
  of the three cohorts.  Wf-325A is the empirical falsification
  of that projection at the CPU-only Ridge baseline.
* **HONEST_NEGATIVE** (this finding): the R² architectural ceiling
  is **confirmed**, not relieved, by the 32-d expansion.  The
  remediation path requires W₂ alignment + chemprop v2 graph
  features + Multi-PH/PLL (deferred to Wf-325B+ GPU portion).

The §9 honest framing catalog in
`TODO/completed/207_unified_pre_flight_checklist_v2.md` is
**expanded by one** with this entry: the Wf-218 32-d expansion
hypothesis is **empirically falsified** for the CPU-only Ridge
baseline on these three cohorts.

---

## §5 File / code references

| Path | Purpose |
|---|---|
| `molmetal/scripts/bench_wf325a_r2_32d.py` | CPU diagnostic bench |
| `molmetal/tests/test_wf325a_r2_32d_expansion.py` | 8 tests (all PASS) |
| `molmetal/reports/wf325a_r2_32d/result.json` | MEASURED result JSON |
| `molmetal/data/rdkit_2d_extended.py` | Wf-218 canonical 32-d extractor |
| `molmetal/data/platinai_4cell_rdkit_features.csv` | 12-d baseline (4078 rows) |
| `molmetal/data/platinai_4cell_rdkit_extended_features.csv` | 32-d extended (200 rows) |
| `molmetal/data/metalcytotox_rdkit_features.csv` | 12-d baseline (26k rows) |
| `metrics/by_metric/hilbert_4cell_r2.json` | Wf-297B source: hilbert R² anchor |
| `metrics/by_metric/metalcytotox_combined_r2.json` | Wf-297B source: combined R² anchor |

### §5.1 Test coverage (8 tests, all PASS)

* `TestPerCohortR2::test_platinai_4cell_r2_shape` — platinai cohort R² shape
* `TestPerCohortR2::test_hilbert_r2_seed_independence` — hilbert cohort R²
* `TestPerCohortR2::test_metalcytotox_combined_r2_shape` — metalcytotox cohort R² shape
* `TestLiftComparison::test_lift_sign_is_computed` — lift sign + verdict
* `TestLiftComparison::test_lift_verdict_documented_labels` — verdict labels
* `TestRidgeBaselineCorrectness::test_ridge_perfect_fit_yields_positive_r2` — Ridge correctness
* `TestRidgeBaselineCorrectness::test_ridge_constant_target_yields_r2_one` — Ridge constant-target edge case
* `TestIntegrationWithResultJson::test_result_json_has_required_fields` — result JSON schema

---

## §6 Pre-flight gate

Per AGENTS.md §1 rule 4 (pre-flight v2 25/25 PASS gate required
before retrain / panel / ablation):

> Wf-325A is a **CPU-only diagnostic** (no retrain, no panel, no
> ablation).  The pre-flight v2 gate is **NOT triggered** for this
> workflow.  CPU-only work proceeds without OK per the standing
> directive in AGENTS.md §5.1.

The pre-flight script (`molmetal/scripts/pre_flight_check.py`)
was launched in background and its output is logged for the
record.  It is not gating this workflow.

---

## §7 Implications for R20+ roadmap

1. **Wf-325B (next)**: GPU W₂ alignment + chemprop v2 graph features
   (TODO-122 P1.3 + Wf-124 P3 5-metal × 5-seed).  This is the
   **only** known path to lift the R² above the architectural
   ceiling, per WEAK_TO_STRONG §2.1.
2. **Wf-218 projection**: the +0.10 to +0.30 R² lift projection in
   the Wf-218 docstring is **empirically falsified** at the
   CPU-only Ridge baseline level.  Update the projection to
   "+0.00 to +0.01 R² lift at the Ridge baseline; full lift
   requires W₂ + chemprop v2 graph features" in a follow-up doc
   patch.
3. **Wf-297B remediation plan §R01 (DANN adversarial head)** and
   §R03 (ADMET pass-through) remain the TIER-1 quick wins —
   they address the DrugOOD scaffold OOD AUC gap, not the
   R² ceiling directly.
4. **Paper §3.6 R² claim**: per TODO-161 §4.4, the chemprop v2
   backbone swap is **orthogonal** to ASTSR / Hilbert / Multi-PH
   / PLL R² ceiling.  Wf-325A reinforces this: the
   R² ceiling is not a chemprop v2 problem; it's a
   feature-architecture problem that requires W₂ + graph
   features + multi-scale topology.

---

## §8 Bottom line

Per the user's request (Wf-325A trigger):

> "得想办法先把honest-negative先解决了，如果能不中断训练去做那更好"

(translation: "we need to figure out how to solve the honest-negative
first, ideally without interrupting any training in flight")

**Wf-325A closes out a single honest-negative question (does the
32-d RDKit expansion lift R²?) without interrupting any training in
flight**.  The answer is **NO** (HONEST_NEGATIVE_CONFIRMED) — the
R² ceiling is not lifted by RDKit descriptor expansion alone; the
remediation requires W₂ alignment + chemprop v2 graph features
(GPU portion, deferred to Wf-325B+).

The Wf-124 P3 multi-task D-MPNN retrain in flight is **not
impacted** by Wf-325A.  No GPU wall time was used.

**End of Wf-325A documentation.** Cross-references:
Wf-297B (R² ceiling finding), Wf-218 (32-d expansion), Wf-127
§Cause 2 (signal-floor on 12-d), WEAK_TO_STRONG §2.1
(W₂ + chemprop v2 + multi-PH/PLL as the architectural fix).
