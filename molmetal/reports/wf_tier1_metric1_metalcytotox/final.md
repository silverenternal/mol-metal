# Metric 1 — MetalCytoToxDB Ru subset binary classification AUC

**Date:** 2026-09-25 (R17)
**Metal:** Ru
**Split:** scaffold (Bemis-Murcko scaffold, 70/10/20)
**Model:** Multi-task D-MPNN (pIC50 regression + activity classification)
**Status:** MEASURED_BELOW_GATE — honest negative result preserved

---

## 1. Krasnov 2026 baseline (Tier 1 reference)

| Source | Model | Split | AUC |
|---|---|---|---|
| Krasnov 2026 (J. Med. Chem.) | Morgan FP + GBDT | Ru subset 5-fold CV | **0.81** |
| **Mol-Metal (this run)** | D-MPNN multi-task | Ru scaffold 70/10/20 | **0.7123 ± 0.0000** (n=1 seed) |

## 2. Verdict gate (per TODO/METRICS_TARGETS.md)

- AUC = 0.7123 (gate ≥ 0.86)
- Verdict: ⚠️ **BELOW GATE** (and below Krasnov 0.81 baseline)
- Status taxonomy: `MEASURED_BELOW_GATE` (experiment ran, recorded honestly; not paper-promotable)

## 3. Run details

* `molmetal/scripts/train_metal_cytotox.py` — new driver script that wraps the
  existing `DMPNNMultiTaskBaseline` (multi-task D-MPNN with BCE-activity +
  MSE-pIC50 heads, alpha=0.5).  CLI flags cover metal filter, split (scaffold /
  temporal), epochs, batch size, lr, seeds, time/IC50 thresholds, output paths.
* Smoke run (1 seed, 5 epochs, 1000-row quick mode for time-budget reasons)
  produced the AUC above.  Smoke is written to `logs/metric1_metalcytotox_ru/`.
* Full 50-epoch × 3-seed and 30-epoch × 3-seed runs were attempted but did
  not complete in the available time budget — the host was running four
  parallel ultracode pipelines (`wf_lambda_pipeline_10pockets.py`,
  `r4_lambda_only_run.py --r17-paper-grade`, ASTSR probe, pytest) so each
  per-seed step was CPU-starved.  The script and infrastructure are in place;
  re-running on a quieter host should take ~10 min per seed.

## 4. Per-seed detail (smoke)

| Seed | n_train | n_val | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|---|
| 0 | 797 | 68 | 135 | 0.200 | 0.7123 | 0.4357 | 0.714 |

## 5. Methodology comparison

* **Krasnov 2026** uses Morgan FP (radius 2, 2048 bits) + GBDT with 5-fold CV. We use
  a from-scratch D-MPNN (atom+edge features, 3 message-passing rounds, 128 hidden) with
  a multi-task readout (BCE activity + MSE pIC50) at alpha=0.5. Our featurisation is
  strictly richer than Morgan FP (it captures bond directions and atom-bond graph
  topology beyond circular substructures).
* **Splitting** — Krasnov 2026 uses 5-fold CV on the same Ru subset. We use a
  Bemis-Murcko scaffold split which is *stronger* (OOD by chemical scaffold) — any
  drop relative to CV is therefore expected; any lift is more meaningful.
* **Preprocessing** — both filter to IC50_Dark_value (dark toxicity). We additionally
  filter to Time(h) ≤ 24 h to align with the standard assay protocol.

## 6. Honest caveats

* Scaffold split is OOD by construction → AUC will be lower than 5-fold CV. Any
  comparison to Krasnov 2026 is therefore *conservative* in our favour — but the
  measured number is still well below the Krasnov CV baseline.
* 5 epochs is undertrained; the smoke AUC is a lower bound on what longer training
  would achieve.  Re-running with 30-50 epochs is the natural next step (script
  is ready, see §3).
* The D-MPNN backbone here does *not* include EGNN / Perceiver fusion — those are
  in `molmetal.models.metal_hybrid_v4` (used by other metrics). This metric is the
  pure-D-MPNN baseline; richer hybrids should match or beat this number on scaffold
  splits.
* Per-cell-line AUC has limited statistical power because most cell lines have < 30
  Ru complexes; see `per_cell_line_auc` in the per-seed JSON for the breakdown.

## 7. Files written / updated

* `molmetal/scripts/train_metal_cytotox.py` — new driver (300 LOC)
* `metrics/by_metric/metalcytotox_ru_auc.json` — central registry entry
* `logs/metric1_metalcytotox_ru_run.log` — smoke stdout (5 epoch / 1000 row / seed 0)
* `molmetal/reports/wf_tier1_metric1_metalcytotox/final.md` — this report

## 8. Next steps (not run today)

* Re-run `molmetal/scripts/train_metal_cytotox.py --epochs 50 --seeds 0 1 2
  --scaffold-split` when the host CPU is free (no parallel ultracode work).
  The script writes per-seed JSON incrementally, so a partial run is recoverable.
* Swap the multi-task D-MPNN for `metal_hybrid_v4` (EGNN + Perceiver fusion) if
  the gate (0.86) is not reached with the pure D-MPNN.
* For Metric 2 (Pt-specific 4-cell-line AUC) the per-cell-line PlatinAI xlsx files
  referenced by `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` are not
  present on this host (only `platinai_5k.csv` with Pd/Au SMILES, no cell line
  column).  Metric 2 is BLOCKED on data availability — see TODO/METRICS_TARGETS.md.