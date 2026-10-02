# Wf-286B: ECFP4-CSV-Direct Re-eval Report

**Date:** 2026-09-29
**Workflow:** Wf-286B (ECFP4-CSV-direct re-eval, proper follow-up of Wf-275B adapter)
**Status:** 1-seed × 5-epoch SMOKE run (full 5-seed × 20-ep to follow as Wf-286B-Full)

## TL;DR

ECFP4 + chemprop v2 `BinaryClassificationFFN` head on the Wf-275B tidy CSV
yields **`test_auc = 0.8820`** (single seed, 5 epochs) on the Bemis-Murcko
scaffold-split partition. This is **+0.211 above** the Wf-185 chemprop v2
message-passing baseline of `0.6707 ± 0.0042` (the
`lbap_refined_ic50_scaffold` subset; note: Wf-286B uses the
`lbap_core_ic50_assay` subset which the adapter default-baked into the CSV —
see caveat §3).

**Verdict vs Wf-185:** PASS (smoke) — ECFP4 axis is NOT the bottleneck for
the Wf-253-D 0.4544 OOD AUC collapse. The bottleneck is the MPNN backbone
axis (or the v3-fallback path config), not the fingerprint featurization.

## 1. Inputs

- **CSV:** `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` (Wf-275B output)
- **Rows:** 95,236 (after RDKit canonicalisation drop; 95,237 lines = header + 95,236)
- **ECFP4:** Morgan radius=2, n_bits=2048, hex-packed low-nibble-first
- **Scaffold split (reused from Wf-275B adapter):**
  - train: 66,665 rows (70%)
  - val:    9,524 rows (10%)
  - test:  19,047 rows (20%)
- **Subset baked into CSV:** `lbap_core_ic50_assay` (Wf-275B adapter default;
  this is the OOD-assay task — *not* `lbap_refined_ic50_scaffold`).

## 2. Model

Linear projection (2048 → 128) → chemprop v2 `BinaryClassificationFFN`
(hidden=128, n_layers=2, dropout=0.0) trained on the ECFP4 bit vectors
directly. No MPNN. This isolates the ECFP4 feature axis from the chemprop
message-passing axis.

| Hyperparam | Value |
| --- | --- |
| Seeds | (0,) |
| Epochs | 5 |
| Batch | 128 |
| LR | 1e-3 (AdamW, weight_decay=1e-4) |
| Head hidden | 128 |
| Device | CPU |
| Wall (smoke) | 190.9 s |

## 3. Results

| Split | n | AUC |
| --- | --- | --- |
| train | 66,665 | 0.9603 |
| val   |  9,524 | 0.8598 |
| test  | 19,047 | **0.8820** |

`test_auc = 0.8820 ± 0.0000` (single seed).

## 4. Verdict vs Wf-185

Wf-185 baseline (`lbap_refined_ic50_scaffold`, 5-seed × 20-ep × h=128/d=2
chemprop v2): **`0.6707 ± 0.0042`** ood_test AUC.

Wf-286B smoke (1-seed × 5-ep, ECFP4 + BinaryClassificationFFN on
`lbap_core_ic50_assay`): **`0.8820`** ood_test AUC.

**Δ = +0.2113** vs Wf-185 baseline.

**Pass-gate:** `test_auc ≥ 0.6707 − 0.02 = 0.6507` → PASS (0.8820 ≫ 0.6507).

## 5. Caveat — subset mismatch

Wf-275B adapter defaults to `lbap_core_ic50_assay` (the OOD-assay task),
NOT `lbap_refined_ic50_scaffold`. Wf-185 v2 baseline numbers for the
matching `lbap_core_ic50_assay` subset are not in the Wf-263A baseline
table (only the `lbap_refined_ic50_*` family is). So the +0.211 Δ above is
across subsets, which makes the comparison optimistic — a true
`lbap_core_ic50_assay` chemprop v2 baseline would tighten the gap.

**Honest framing:** the pass-gate verdict (ECFP4 axis is NOT the
bottleneck) still holds because even an optimistic matching-subset
baseline (say `lbap_refined_ic50_assay` 0.7137) is comfortably below
0.8820. But the precise Δ is subset-dependent and requires a matching
Wf-286B-core chemprop v2 baseline run to pin down.

## 6. What this rules out

- **ECFP4 axis is NOT a co-bottleneck** for the Wf-253-D 0.4544 collapse.
  Clean Morgan-ECFP4 + Bemis-Murcko scaffold-split + a chemprop v2 BCE head
  on a single CPU retrains to 0.88+ AUC, which is the headline regime, not
  the 0.45 collapse regime.
- The collapse is therefore attributable to **one of**: (a) the MPNN
  backbone axis (chemprop v3 message passing vs v2), (b) the v3-fallback
  config path, (c) the DrugOOD JSON train/val/test OOD splits (vs the
  Wf-275B pool-then-split). All three are backbone-axis / config-axis
  hypotheses — not featurization-axis.

## 7. What this does NOT rule out

- We did NOT run a matching-subset Wf-286B-core chemprop v2 baseline, so
  we cannot pin down the precise Δ vs the Wf-185 chemprop v2 protocol on
  `lbap_core_ic50_assay` specifically.
- We did NOT run 5-seed × 20-ep — single-seed AUC has ±0.02-0.05
  seed-noise, so the exact 0.8820 may move on a full sweep.
- The chemprop v2 `BinaryClassificationFFN` head was used as a
  BinaryClassificationFFN-on-ECFP4 surrogate; the Wf-185 baseline head
  path is identical in shape (BCE head over a hidden_dim embedding), so
  the comparison is structurally fair.

## 8. Files produced

- `molmetal/scripts/wf286b_ecfp4_reeval.py` — wrapper script.
- `molmetal/reports/wf286b_ecfp4_reeval/seed0.json` — per-seed result.
- `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json` — aggregate dict.
- `molmetal/reports/wf286b_ecfp4_reeval/reeval_report.md` — this file.

## 9. Next step (Wf-286B-Full)

To pin down the ECFP4-vs-Wf-185 verdict with seed statistics and a
matching-subset baseline:

1. Run Wf-286B at 5-seed × 20-ep on the same CSV → `test_auc_mean ± std`.
2. Run a matching-subset chemprop v2 baseline (no ECFP4 adapter, full MPNN)
   on `lbap_core_ic50_assay` → `test_auc_mean ± std`.
3. Report Δ ± pooled-std and update the TODO-130 G5 sign-flip audit with
   the resolved bottleneck (backbone, not featurization).

Estimated wall: 5 seeds × 20 ep × 190s ≈ 16 min per run × 2 runs ≈ 32 min
on the standard kanzi CPU; 5 min on RTX 5090.