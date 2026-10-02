# Wf-278A — TODO-180 G4 Aggregation Path-Mismatch Fix

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Workflow:** Wf-278A
**Parent:** Wf-276C (TODO-180 G4 14 pairs MEASURED COMPLETE but `training_summary.json` showed `0 pairs aggregated`)

## 1. Symptom

Wf-276C closed TODO-180 G4 with the following observation:
> TODO-180 G4 14 pairs MEASURED COMPLETE (Wave A 8 + Wave B 6, ~108 min wall)
> BUT `training_summary.json` shows "0 pairs aggregated" — path mismatch.

`training_summary.json` BEFORE fix:
```json
{
  "todo_180_g4_total_pairs": 0,
  "runs": []
}
```

## 2. Root cause (path mismatch)

The original aggregator `molmetal/scripts/wf196_aggregate.py` was hard-coded for
the **flat** Wf-124 P3 layout:

```
molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed{seed}.json
```

But the Wf-124 P3 trainer (when run from the G4 config
`configs/wf124_p3_remaining_metals_g4.yaml`) writes to a **nested** layout:

```
results/wf124_p3_remaining_metals_g4/{wave}/{Metal}/seed{N}/{metal}/per_seed_metrics_seed{N}.json
                                 ^^^^^^  ^^^^^^         ^^^^   ^^^^^^
```

The `glob("per_seed_metrics_seed*.json")` against `SWEEP_DIR / metal.lower()`
matched **zero** files because the metal-named directory only existed one level
*deeper* under `wave/{Metal}/seed{N}/{metal}/`. Hence `0 pairs aggregated`.

## 3. Fix

Created `molmetal/scripts/wf278a_g4_aggregate.py` — a CLI-driven aggregator that:

1. Walks the **nested** tree with `root.glob("**/per_seed_metrics_seed*.json")`.
2. De-duplicates by `(metal, seed)` tuple derived from path parents (the bare
   `per_seed_metrics.json` aggregator wrapper files are skipped because they
   do not match the `seed{N}` suffix pattern).
3. Verifies payload `metal`/`seed` match the path-derived key (WARN log on
   mismatch).
4. Computes per-metal mean ± std for `test_auc`, `val_auc`, `test_pearson`,
   `test_rmse`, `test_mae`, `w_final`, and a 5-metal aggregate.
5. Writes `training_summary.json` with `todo_180_g4_total_pairs` set to the
   aggregated count.

### Diff summary (vs `wf196_aggregate.py`)

| Concern | `wf196_aggregate.py` (OLD) | `wf278a_g4_aggregate.py` (NEW) |
|---|---|---|
| Path layout | flat `{metal}/per_seed_metrics_seed*.json` | nested `**/{Metal}/seed{N}/{metal}/per_seed_metrics_seed{N}.json` |
| Input dir | hard-coded `molmetal/reports/wf196_50ep_sweep` | CLI `--input-dir` |
| Output path | hard-coded `SWEEP_DIR / training_summary.json` | CLI `--output` |
| Dedup key | n/a (flat) | `(metal_dir.lower(), seed_n)` tuple |
| Skips wrapper | n/a | yes — only files matching `per_seed_metrics_seed*.json` |
| Scope | Wf-196 50-ep CPU sweep (preserved) | Wf-278A TODO-180 G4 14 pairs |

## 4. Verification

`training_summary.json` AFTER fix (`/home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/training_summary.json`):

```
todo_180_g4_total_pairs = 14
expected_pairs          = 14
n_pairs_total           = 14
aggregate 5-metal test_auc = 0.7209 +- 0.0458  (n=14)

Ru test_auc = 0.7483 +- 0.0030  (n=2: seeds 1, 2)
Ir test_auc = 0.6702 +- 0.0056  (n=3: seeds 0, 1, 2)
Os test_auc = 0.7025 +- 0.0205  (n=3: seeds 0, 1, 2)
Rh test_auc = 0.7227 +- 0.0540  (n=3: seeds 0, 1, 2)
Re test_auc = 0.7700 +- 0.0244  (n=3: seeds 0, 1, 2)
```

Per-seed test_auc breakdown (sanity check vs Wf-276C):

```
Ru seed1 = 0.7453  (Wf-276C: status=OK)
Ru seed2 = 0.7514  (Wf-276C: status=OK)
Ir seed0 = 0.6659  (Wf-276C: status=OK)
Ir seed1 = 0.6782  (Wf-276C: status=OK)
Ir seed2 = 0.6666  (Wf-276C: status=OK)
Os seed0 = 0.7137  (Wf-276C: status=OK)
Os seed1 = 0.6738  (Wf-276C: status=OK)
Os seed2 = 0.7201  (Wf-276C: status=OK)
Rh seed0 = 0.7970  (Wf-276C: status=OK)
Rh seed1 = 0.6702  (Wf-276C: status=OK)
Rh seed2 = 0.7008  (Wf-276C: status=OK)
Re seed0 = 0.7756  (Wf-276C: status=OK)
Re seed1 = 0.7967  (Wf-276C: status=OK)
Re seed2 = 0.7378  (Wf-276C: status=OK)
```

All 14 pairs: file present, status=OK, test_auc finite, payload-metal/seed
matches path-derived key (no WARN log lines emitted).

## 5. Result

| | Before | After |
|---|---|---|
| pairs_before_fix | 0 | — |
| pairs_after_fix | — | 14 |
| per-metal status | all `NO_SEEDS` | 5/5 `OK` |
| 5-metal aggregate test_auc | `nan` | 0.7209 +- 0.0458 |

## 6. Honest framing

- Wf-276C's MEASURED COMPLETE claim was correct (14 seed payloads present).
- This fix only restores the aggregator; no GPU retraining.
- `wf196_aggregate.py` is **untouched** and still works for the flat Wf-196
  50-ep CPU sweep layout.
- The new aggregator is CLI-driven (`--input-dir`, `--output`) so future
  wave-laid-out results (Wf-265 G5, Wf-266 G6) can reuse it without code
  changes.

## 7. Cross-references

- `molmetal/scripts/wf196_aggregate.py` — original (flat-path) aggregator
- `molmetal/scripts/wf278a_g4_aggregate.py` — NEW (nested-path) aggregator
- `configs/wf124_p3_remaining_metals_g4.yaml` — G4 config (nested layout)
- `molmetal/reports/wf276c_*/` — Wf-276C audit (claim source)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` — TODO-180 G4