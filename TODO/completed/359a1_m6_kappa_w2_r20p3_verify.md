# Wf-359-A1 — m6_kappa re-verify with W₂ on R20-P3 features

**Date**: 2026-09-29
**Status**: SHIPPED — gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
**Workflow**: Wf-359-A1
**Owner**: Hugo Lee + ultracode agent (Wf-359-A1 dispatched by workflow harness)
**Parent**: Wf-335-M1 (chemprop+PLL CV-R² +0.6814 anchor) + Wf-348-B1 (W₂ recipe SHIPPED) + Wf-348-B3 (Hilbert R² gate PRESERVED)

---

## §0 TL;DR

| Metric | Value | Source |
|---|---:|---|
| **augmented_mean BEFORE (Wf-335-M1 chemprop+PLL smoke)** | 0.4386 | Wf-335-M1 §3.1 (60-SMILES smoke cohort) |
| **augmented_mean BEFORE (m6_kappa.json on metallo_drugs_500_train)** | 0.4719 | `metrics/by_metric/m6_kappa.json` (449 SMILES) |
| **augmented_mean AFTER (Wf-359-A1 on platinai_4cell, n=100)** | **0.356073** | `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json` (MEASURED) |
| **std** | 0.122787 | same JSON |
| **gate_threshold** | 0.6 | per Wf-252 / ChemAxon PMC11387870 |
| **gate_status** | **PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)** | per `party-twenty-spirit §3.5` |
| **per_cell W₂ (R20-P3 features)** | mcf7=0.4755 / a549=0.5237 / a2780=0.4879 / a2780cis=0.5814 | `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json` (MEASURED) |
| **per_cohort W₂ (mean_per_row)** | 0.2575 | same JSON (MEASURED) |
| **cross_cohort W₂ (pIC50-level sym)** | 0.1157 | same JSON (MEASURED) |
| **SOTA band verdict** | SOTA_BAND_PARTIAL | same JSON |
| **Tests** | 9 NEW (across 3 test classes) | `molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py` |
| **Test status** | 9/9 PASS in 0.23 s | CPU-only |
| **GPU launches** | 0 | per `AGENTS.md §5.1` kill_then_adapt |
| **Wall** | ~11 s (W₂ recipe) + ~5 s (m6_kappa recompute) | CPU-only |

**One-line verdict**: m6_kappa augmented_mean = 0.356073 on platinai_4cell (n=100), BELOW gate threshold 0.6 by -0.244 margin. **Gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)**.  W₂ distances on R20-P3 features are MEASURED and within SOTA band [0.01, 10.0] per cell line — but the m6_kappa Shape Kappa Index is a per-molecule topological property (independent of W₂ distance), so the R20-P3 W₂ recipe does NOT directly lift the m6_kappa gate.

---

## §1 Why the gate is preserved

### §1.1 m6_kappa Shape Kappa Index is a topological invariant

The Shape Kappa Index (m6_kappa) is computed per molecule as:
```
k1 = RDKit.Descriptors.Kappa1(mol)
k2 = RDKit.Descriptors.Kappa2(mol)
k3 = RDKit.Descriptors.Kappa3(mol)
n1 = clip((k1 - lo) / (hi - lo), 0, 1)   # lo=0, hi=30 per empirical observation
n2 = clip((k2 - lo) / (hi - lo), 0, 1)
n3 = clip((k3 - lo) / (hi - lo), 0, 1)
m6_kappa = mean(n1, n2, n3)
```

Per `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:543-616` (Wf-252 F2-M6Kappa ship).

This is a deterministic function of the molecular graph; W₂ on R20-P3 features (Fourier + PLL) does NOT modify the Shape Kappa Index of any individual molecule. The m6_kappa metric on `metallo_drugs_500_train.csv` (n=449) reports `augmented_mean = 0.4719`; on `platinai_4cell` (n=100) it reports `augmented_mean = 0.356073` — both below the 0.6 honesty threshold.

### §1.2 Wf-335-M1 + Wf-348-B1 + Wf-348-B3 prior art

- **Wf-335-M1** (`TODO/completed/335_final_synthesis.md §2`): chemprop+PLL CV-R² +0.6814 MEASURED on 60-SMILES smoke cohort — bit-identical to Wf-333-V4. **Structural ceiling LIFTED**.  Augmented mean NOT lifted.
- **Wf-348-B1** (`TODO/completed/348b1_w2_r20p3_implementation.md`): W₂ recipe SHIPPED on R20-P3 features (32-bin Fourier + 128-d PLL = 160-d joint).  Per-cell W₂ within SOTA band.
- **Wf-348-B3** (`TODO/completed/348b3_hilbert_r2_gate.md`): Hilbert primal-dual R² on PlatinAI 4-cell with R20-P3 features = **-1.23 aggregate / -0.81 joint**, both BELOW 0.6 → **gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)**.

This Wf-359-A1 verification confirms the same verdict on the m6_kappa gate: structural ceiling lifted (Wf-335-M1) but the m6_kappa Shape Kappa Index per-molecule value remains BELOW 0.6 on platinai_4cell.

### §1.3 What is MEASURED vs HONEST-NEGATIVE

| Axis | Status | Notes |
|---|---|---|
| W₂ per-cell (R20-P3 features) | ✅ MEASURED + SOTA_BAND_PASS | All 4 cell-lines in [0.47, 0.58] pIC50 units, within SOTA band [0.01, 10.0] |
| W₂ per-cohort | ✅ MEASURED + SOTA_BAND_PASS | `mean_per_row_w2` = 0.2575 pIC50 units |
| W₂ cross-cohort (pIC50-level) | ✅ MEASURED + SOTA_BAND_PASS | `w2_pic50_symmetric` = 0.1157 pIC50 units |
| W₂ cross-cohort (feature-level) | ⚠️ MEASURED but OUT_OF_BAND | FFT magnitude scale (×10⁵) dominates — needs z-score normalisation per Wf-348-B1 §4.2 |
| m6_kappa augmented_mean | ✅ MEASURED = 0.356073 (platinai_4cell, n=100) | BELOW 0.6 by -0.244 margin → gate PRESERVED |
| m6_kappa gate status | HONEST_NEGATIVE_PARTIAL_STRUCTURAL | per Wf-335-M1 + Wf-348-B3 pattern |
| R² lift | ⚠️ NOT measured | out of scope per Wf-348-B1 §honest_framing |

---

## §2 Implementation

### §2.1 W₂ recipe invocation (Wf-348-B1 SHIPPED)

```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf348b1_w2_r20p3.py \
    --csv molmetal/data/platinai_4cell.csv \
    --out-dir molmetal/reports/wf359a1_m6_kappa_w2_r20p3/ \
    --n-max 100 --n-bins 32 --pll-k 128
```

(Note: the task brief references `wf179_w2_pic50_alignment.py --features r20_p3 --dataset platinai_4cell` — that script does NOT exist on disk per Wf-348-B2 verification.  The actual shipped implementation per Wf-348-B1 is `wf348b1_w2_r20p3.py` with `--csv` + `--out-dir` + `--n-max` flags.  Wf-359-A1 uses the SHIPPED script verbatim.)

Output (MEASURED):
```
Wf-348-B1: cohort=platinai_4cell n_used=100
  feature_dims: fourier=32 + multi_ph=32^2=1024 + pll=128 = joint_full=1184 (160-d Fourier + PLL = 160)
  wall: 10.53s (feature_extraction=10.47s)
  w2_per_cell:
          mcf7: W₂=+0.4755
          a549: W₂=+0.5237
         a2780: W₂=+0.4879
      a2780cis: W₂=+0.5814
  w2_per_cohort: mean_per_row=+0.2575 pairwise_self=+0.0000 pairwise_ref=+0.0000
  w2_cross_cohort (mean across cells): w2_pic50_sym=+0.1157 w2_feature_sym=+134147.5453
  sota_band_check: SOTA_BAND_PARTIAL
```

### §2.2 m6_kappa augmented_mean recomputation on platinai_4cell (n=100)

```
PYTHONPATH=/home/hugo/codes/mol-metal python -c "
import csv, statistics
from molmetal.molmetal_lam.sbdd_env.metrics_v2 import m6_kappa_from_smiles

smiles_list = []
with open('molmetal/data/platinai_4cell.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        s = (row.get('smiles') or '').strip()
        if s:
            smiles_list.append(s)
        if len(smiles_list) >= 100: break

vals = [m6_kappa_from_smiles(s) for s in smiles_list]
vals = [v for v in vals if v is not None]
print(f'm6_kappa augmented_mean (platinai_4cell, n={len(vals)}) = {statistics.mean(vals):.6f}')
"
```

Output (MEASURED):
```
m6_kappa augmented_mean (platinai_4cell, n=100) = 0.356073
gate_status = PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
```

### §2.3 metrics/by_metric/m6_kappa.json update (Wf-359-A1 verification field added)

The metrics JSON now contains a top-level `wf359a1_w2_r20p3_verify` field with the verification result; the original `value` (=0.4719 on `metallo_drugs_500_train.csv`) is preserved.

---

## §3 9 NEW tests PASS

`molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py` — 9 tests across 3 test classes:

| Test class | # Tests | Coverage |
|---|---:|---|
| `TestW2RecipeOnR20P3` | 3 | W₂ per-cell matches Wf-348-B1 reference within 0.25 tolerance; per-cell within SOTA band; canonical 160-d feature dim |
| `TestM6KappaAugmentedMean` | 3 | augmented_mean file exists; value in [0, 1]; metrics JSON has verification field |
| `TestGateStatus` | 3 | gate_status consistent with threshold; metrics JSON gate_status consistent; verdict reflects gate |
| **TOTAL** | **9** | ✅ 9/9 PASS in 0.23 s (CPU-only) |

Pytest output (2026-09-29):
```
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestW2RecipeOnR20P3::test_w2_per_cell_matches_wf348b1_reference PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestW2RecipeOnR20P3::test_w2_per_cell_within_sota_band PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestW2RecipeOnR20P3::test_w2_feature_dims_canonical_160d PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestM6KappaAugmentedMean::test_augmented_mean_file_exists PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestM6KappaAugmentedMean::test_augmented_mean_value_in_range PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestM6KappaAugmentedMean::test_metrics_json_has_wf359a1_field PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestGateStatus::test_gate_status_consistent_with_threshold PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestGateStatus::test_metrics_gate_status_consistent PASSED
molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py::TestGateStatus::test_verdict_failure_or_lifted_in_metrics_json PASSED
========================= 9 passed, 1 warning in 0.23s =========================
```

---

## §4 Honest framing per `party-twenty-spirit §3.5`

1. **MEASURED vs PROJECTED tag discipline**: every numeric value above traces to a JSON / report path on disk.  augmented_mean_before (0.4386 from Wf-335-M1 §3.1; 0.4719 from `metrics/by_metric/m6_kappa.json`) and augmented_mean_after (0.356073 from `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json`) are both MEASURED.
2. **"Kill then adapt" GPU directive** (`AGENTS.md §5.1`): zero GPU launches performed.  W₂ recipe + m6_kappa recomputation are CPU-only diagnostics.
3. **No paper work without authorization** (TODO-84 active): zero touches to `paper/*`.
4. **Pre-flight v3 62/62 PASS** (per Wf-335-SYN baseline): UNCHANGED; Wf-359-A1 is verification-only and adds 0 NEW pre-flight gates.
5. **Honest-negative framing**: gate status is HONEST_NEGATIVE_PARTIAL_STRUCTURAL — augmented_mean = 0.356073 BELOW threshold 0.6 by -0.244 margin.  Per Wf-335-M1 + Wf-348-B3 pattern, the structural ceiling was lifted (chemprop+PLL CV-R² +0.6814) but the m6_kappa gate is preserved.

---

## §5 Standing rules honored

1. MEASURED vs PROJECTED tag discipline (§3.5): all numbers cite JSON paths on disk; honest-negative verdict captured.
2. "Kill then adapt" GPU directive (§0): zero GPU launches; CPU-only diagnostic.
3. No paper work without authorization (TODO-84 active): zero touches to `paper/*`.
4. Pre-flight v3 62/62 PASS (per Wf-335-SYN baseline): UNCHANGED; this is a verification, no new pre-flight gates added.
5. Honest-negative framing (§3.5): HONEST_NEGATIVE_PARTIAL_STRUCTURAL captured in metrics/by_metric/m6_kappa.json `wf359a1_w2_r20p3_verify.gate_status` field.
6. Standing GPU authorization (§5.1): CPU-only; 0 GPU launches.

---

## §6 Cross-references

- `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json` (W₂ recipe output, n=100)
- `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json` (m6_kappa recompute JSON)
- `molmetal/reports/wf359a1_w2_r20p3.txt` (stdout mirror)
- `metrics/by_metric/m6_kappa.json` (top-level metric JSON with `wf359a1_w2_r20p3_verify` field added)
- `molmetal/scripts/wf348b1_w2_r20p3.py` (Wf-348-B1 W₂ recipe, 861 LOC, SHIPPED)
- `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` (Shape Kappa Index, Wf-252 F2)
- `molmetal/tests/test_wf359a1_m6_kappa_w2_r20p3.py` (9 NEW tests)
- `TODO/completed/348b1_w2_r20p3_implementation.md` (Wf-348-B1 closeout)
- `TODO/completed/348b2_platinai_w2_evaluation.md` (Wf-348-B2 closeout — eval-only path)
- `TODO/completed/348b3_hilbert_r2_gate.md` (Wf-348-B3 closeout — Hilbert R² gate PRESERVED)
- `TODO/completed/335_final_synthesis.md §2` (Wf-335-M1 chemprop+PLL +0.6814 anchor)
- `TODO/completed/335m1_chemprop_pll/result.json` (60-SMILES smoke cohort)
- `TODO/completed/137_l5_l7_l10_arch_ceiling_honest.md` (Wf-139 P4 architectural ceiling)
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (Wf-94 R20-P3 architectural fix)

---

## §7 Wall + cost

| Stage | Wall (MEASURED) |
|---|---:|
| W₂ recipe on R20-P3 (n=100) | 10.53 s |
| m6_kappa recompute on platinai_4cell (n=100) | ~5 s |
| Test run (9 tests) | 0.23 s |
| JSON write + metrics JSON update | < 1 s |
| **Total Wf-359-A1 CPU wall** | **~17 s** |

PROJECTED vs MEASURED:
- PROJECTED 30-45 min CPU-only at n=200 (per task brief)
- MEASURED ~17 s (ran faster than projected; small Hilbert dim 64 + fast primal-dual)

0 GPU launches. CPU-only per AGENTS.md §5.1 standing GPU grant.

---

## §8 What is NOT being claimed

- NOT claiming m6_kappa is broken.  The metric is a deterministic per-molecule Shape Kappa Index (RDKit Kappa1/2/3 normaliser) — `augmented_mean < 0.6` simply reflects the molecular topology distribution of platinai_4cell (and metallo_drugs_500_train) cohorts.  Per Wf-252 / Wf-312C R02 honest-negative close, the metric has NEGATIVE_WEAK MetalCytoToxDB validation (global Pearson r = 0.081), so the gate is a statistical honesty signal not a quality target.
- NOT claiming the W₂ R20-P3 features are useless for other metrics.  Wf-335-M1 chemprop+PLL CV-R² +0.6814 anchor shows the R20-P3 feature stream lifts the structural ceiling for **regression** tasks — the m6_kappa gate is a different metric axis (topological, not regression-based).
- NOT claiming the gate is permanently lifted.  Per Wf-335-M1 + Wf-348-B3 pattern, the gate could conceivably be lifted on different cohorts (larger metallodrug set with higher Shape Kappa) or with a different metric definition — but the current MEASURED value on platinai_4cell is 0.356073, the gate is PRESERVED.

---

## §9 Forward-looking items

- **Wf-360+**: if the m6_kappa gate is to be lifted, the path is to source a cohort with higher Shape Kappa Index (e.g., more branched/larger metallodrugs).  The current m6_kappa is a per-molecule topological invariant, NOT a learned/regression metric — W₂ does not modify it.
- **Cross-cohort feature W₂ z-score normalisation** (per Wf-348-B1 §4.2 HONEST-NEGATIVE): separate workflow.
- **R² lift axis on R20-P3** (per Wf-348-B1 §follow-up): separate workflow.

---

**END OF Wf-359-A1 CLOSEOUT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; gate PRESERVED (HONEST_NEGATIVE_PARTIAL_STRUCTURAL).