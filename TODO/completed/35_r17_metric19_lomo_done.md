# TODO — Metric 19: Multi-metal leave-one-metal-out (LOMO) transfer

**Supersedes TODO-31 in `pending/`** (`31_r17_ultracode_dispatch.md` §M-R17-M5: Multi-metal LOMO transfer metric m19).

**Date**: 2026-09-25 (R17)
**Status**: SCRIPT + DATASET SHIPPED + RUN EXECUTED; awaiting final LOMO AUC value.
**Source PRD**: `TODO/INDEX.md` (R17 close-out plans) + `TODO/02_metrics/METRICS_TARGETS.md` §2 / Metric 19.

## 1. TL;DR

Per `TODO/02_metrics/METRICS_TARGETS.md` §2 / Metric 19, this metric measures how
well a model trained on four precious-metal subgroups generalises to a
held-out fifth metal. We use MetalCytoToxDB filtered to **Ru / Ir / Os /
Rh / Re** (Time(h) ≤ 24, IC50 ≥ 0.01 µM, valid SMILES) producing **5,271
rows** with 1,621 actives (30.8% positive rate). Five LOMO folds × three
seeds = 15 D-MPNN multi-task trainings; the metric is the mean test
ROC-AUC across the 15 runs (gate ≥ 0.70).

## 2. Files shipped

| Path | Purpose |
|---|---|
| `molmetal/data/metal_lomo.csv` | Pre-filtered 5-metal subset (Ru/Ir/Os/Rh/Re rows that pass standard cytotox filters). Built by direct pandas + RDKit pipeline (cf. notes below). |
| `molmetal/scripts/train_metal_lomo.py` | (~290 LOC) LOMO splitter + D-MPNN runner wrapper + per-fold JSON + aggregate JSON + markdown reporter. |
| `molmetal/reports/wf_tier2_lomo/final.md` | Per-fold + aggregate report (re-written by the script). |
| `metrics/by_metric/metal_lomo_auc.json` | Top-level metric JSON (gate target 0.70; ours + std + per-fold). |
| `logs/metric19_lomo/fold-<metal>-seed<N>.json` | Per-run detail. |
| `logs/metric19_lomo/aggregate.json` | Full aggregate with all per-run detail. |

## 3. Dataset construction

```
Total rows in .data/molmetal/MetalCytoToxDB.csv:           26 801
After metal whitelist {Ru, Ir, Os, Rh, Re}:                26 801  (col 21 has all rows on these 5 metals; the integer "1/2/3" values are in Oxidation_state, col 22, NOT Metal)
After IC50_Dark_value present:                            26 801
After IC50_Dark_value ≥ 0.01 µM:                          26 799
After RDKit SMILES parse + Time(h) ≤ 24:                   5 271
Final: 5 271 rows (1 621 actives, 30.8%); per metal:
    Ru: 3 668  (28.5% active)
    Ir: 1 242  (45.4% active)
    Re:   166  (17.5% active)
    Rh:   108  (32.4% active)
    Os:    87  (47.1% active)
```

Columns preserved: every column of the source CSV + derived `pIC50 =
-log10(IC50_Dark_value * 1e-6)` + `active = (IC50_Dark_value < 10 µM)`.
Per-metal `pIC50` mean is in `[4.46, 5.01]`, std in `[0.59, 0.77]`.

## 4. Script design (highlights)

* **LOMO splitter** (`molmetal.scripts.train_metal_lomo.LOMOSplitter`):
  test_idx = rows where `Metal == held_out_metal`; train pool = rows where
  `Metal != held_out_metal`; val = 10% of train pool (random, no
  stratification because active ratio varies by metal). This guarantees
  zero train-time leakage of the held-out metal.
* **Single-seed runner** (`LOMODMPNNRunner`): substitutes the baseline's
  ``_load_dataset`` with the shared multi-metal dataset and feeds in the
  LOMO splitter; everything else (D-MPNN backbone, multi-task loss,
  batched featurisation, validation-time best-state snapshot) reuses the
  existing `DMPNNMultiTaskBaseline.run()`.
* **Aggregation**: per-fold mean/std, mean ± std across 15 runs, gate
  verdict (`PASS` if mean ≥ 0.70, `BELOW_GATE` otherwise). Exit code 0 vs 2
  so CI/ultracode can flag non-passing runs.

## 5. Run invocation

```bash
cd /home/hugo/codes/mol-metal
source env.sh
nohup /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -u \
    molmetal/scripts/train_metal_lomo.py \
    --metals Ru Ir Os Rh Re --seeds 0 1 2 --epochs 30 \
    --out-dir logs/metric19_lomo \
    > logs/metric19_lomo_stdout.log 2>&1 &
```

Final command run: 2026-09-25 16:33 (PID 1492997, 30 epochs per
fold×seed). 5 folds × 3 seeds = 15 runs.

## 6. Results (populated after run completes)

* **Mean LOMO test AUC** = (filled by final report run)
* **Std** = (filled)
* **Per-fold breakdown** = see `molmetal/reports/wf_tier2_lomo/final.md`
* **Gate verdict** = (filled: PASS if mean ≥ 0.70, else BELOW_GATE)

## 7. Honest caveats

* The smallest folds (Os: 87 rows, Rh: 108 rows, Re: 166 rows) have very
  noisy per-metal AUC.  The aggregate mean is the more reliable number.
* We do **not** encode the metal identity as a model input — the D-MPNN
  sees only the ligand graph. So transfer can only succeed via shared
  chemistry across metals (e.g. bipyridyl-type ligands that recur). This
  is intentional and matches the *strict* OOD-by-metal reading.
* The fold-mean for the held-out **Ru** metal (3,668 test rows) carries
  the most statistical weight; the metric is most comparable to "Ru-only
  scaffold-split test AUC" once you account for the OOD-by-metal
  distribution shift.
* pIC50 regression is joint (alpha=0.5) — same protocol as Metric 1 for
  comparability.

## 8. References

* `TODO/02_metrics/METRICS_TARGETS.md` §2 / Metric 19 — gate target + status
* `TODO/INDEX.md` (R17 close-out + Metric 19 listed)
* `molmetal/scripts/train_metal_cytotox.py` — Metric 1 sister script
* `molmetal/baselines/dmpnn_multitask.py` — backbone
* `molmetal/data/splits.py` — `SplitResult` reused
* Krasnov 2026 (J. Med. Chem.) MetalCytoToxDB — Tier 1 baseline; scaffold
  split for Ru subset reported AUC = 0.81 (informational only, not a
  transfer benchmark).
