# Wf-292A — TODO-180 G4 Ru seed=0 cascade via G5 per_seed_metrics.json

**Date:** 2026-09-29
**Author:** Hugo Lee + Claude Code (subagent)
**Goal:** Verify G5 Ru seed=0 emit test_auc=0.7716 (per Wf-281A + Wf-284C) and close TODO-180 G4 n=14 → n=15 gap.

---

## 1. G5 Ru seed=0 file verification

**Expected path** (per Wf-281A + Wf-284C):
`results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/per_seed_metrics_seed0.json`

**Actual location** (nested `ru/` dir inside seed0, matching the G4 nested layout):
`results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics_seed0.json`

```
$ ls -la /home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/
total 8
drwxrwxr-x 1 hugo hugo   96 Sep 29 00:28 .
drwxrwxr-x 1 hugo hugo   70 Sep 29 00:28 ..
-rw-r--r-- 1 hugo hugo 2474 Sep 29 00:28 per_seed_metrics.json
-rw-r--r-- 1 hugo hugo 2105 Sep 29 00:28 per_seed_metrics_seed0.json
```

## 2. Schema verification (vs `wf278a_g4_aggregate.py` expectations)

| Field | Expected | Actual | Match |
|-------|----------|--------|-------|
| `metal` | "Ru" or "ru" | `"Ru"` | YES |
| `seed` | 0 | `0` | YES |
| `n_train` | ~12471 (5-metal cohort) | `12471` | YES |
| `epochs` | 50 (G5 full sweep) | `50` | YES |
| `test_metrics.roc_auc` | ~0.7716 | `0.7715537578862466` | YES (0.7716) |
| `val_metrics.roc_auc` | (any) | `0.8176661495862474` | OK |
| `status` | "OK" | `"OK"` | YES |
| `wasserstein_history` | list of 50 (epochs 1-50) | list of 50 values | YES |
| `ckpt_path` | results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/... | exact match | YES |

**Verdict:** Schema fully matches the `wf278a_g4_aggregate.py` expectations. The file is a valid input.

## 3. Cascade wire-up

The G5 file lives at a separate input_dir (`wf124_p3_50epoch_full_sweep_g5/`) than the G4 aggregator scans (`wf124_p3_remaining_metals_g4/`). The aggregator's `collect_seed_metrics_nested` uses a recursive `**/per_seed_metrics_seed*.json` glob, so any file under the input-dir tree is picked up.

**Strategy:** Create a symlink under G4's input-dir at `C/Ru/seed0/ru/per_seed_metrics_seed0.json` pointing to the G5 source of truth (no copy — symlink keeps G5 as the canonical record).

```
$ mkdir -p /home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru
$ ln -s /home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics_seed0.json \
       /home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json
```

Note the metal-dir (`ru`) is the parent of the filename per the script's
`metal_dir = p.parent.name  # "ru"` convention.

## 4. Re-run aggregator

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/wf278a_g4_aggregate.py \
    --input-dir /home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4 \
    --output   /home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/training_summary.json
```

**Console output:**

```
[wf278a-agg] wrote /home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/training_summary.json

============================================================
Wf-278A — TODO-180 G4 nested-path re-aggregation
============================================================
  Ru: test_auc = 0.7561 ± 0.0112 (n=3)  val_auc=0.7771±0.0310  test_pearson=0.458±0.064  w_final=0.3226
  Ir: test_auc = 0.6702 ± 0.0056 (n=3)  val_auc=0.8513±0.0027  test_pearson=-0.007±0.042  w_final=0.5359
  Os: test_auc = 0.7025 ± 0.0205 (n=3)  val_auc=0.7746±0.0458  test_pearson=0.355±0.131  w_final=0.5245
  Rh: test_auc = 0.7227 ± 0.0540 (n=3)  val_auc=0.7450±0.0619  test_pearson=0.151±0.037  w_final=0.3534
  Re: test_auc = 0.7700 ± 0.0244 (n=3)  val_auc=0.8812±0.0622  test_pearson=0.395±0.113  w_final=0.3903

  5-metal aggregate: test_auc = 0.7243 ± 0.0460 (n=15)
```

## 5. Pair count verification

| Metal | n_seeds | ckpt_paths | Status |
|-------|---------|------------|--------|
| Ru | 3 (0, 1, 2) | G5 seed0 + G4 A/seed1 + G4 A/seed2 | OK |
| Ir | 3 (0, 1, 2) | G4 A/seed0,1,2 | OK |
| Os | 3 (0, 1, 2) | G4 A/seed0,1,2 | OK |
| Rh | 3 (0, 1, 2) | G4 B/seed0,1,2 | OK |
| Re | 3 (0, 1, 2) | G4 B/seed0,1,2 | OK |
| **Total** | **15** | — | OK |

From the updated `training_summary.json`:

```json
"n_pairs_total": 15,
"todo_180_g4_total_pairs": 15,
"aggregate_5metal": {
  "n_pairs": 15,
  "test_auc_mean": 0.7243,
  "test_auc_std": 0.0460,
  ...
}
```

(Note: `expected_pairs` field still shows `14` because it's a hard-coded
constant in the aggregator script — the *actual* n_pairs_total=15 is the
authoritative count.)

**Gap closed: n=14 → n=15** (Ru seed=0 added).

## 6. Ru per-seed AUC cascade

| Seed | test_auc | Source |
|------|----------|--------|
| 0 | **0.7716** | G5 (`wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics_seed0.json`) — 50 epochs |
| 1 | 0.7453 | G4 (`wf124_p3_remaining_metals_g4/A/Ru/seed1/`) — 30 epochs |
| 2 | 0.7514 | G4 (`wf124_p3_remaining_metals_g4/A/Ru/seed2/`) — 30 epochs |
| Mean | **0.7561 ± 0.0112** | n=3 |

The G5 50-epoch Ru seed=0 (test_auc=0.7716) is the highest of the 3 Ru seeds.
This is consistent with the Wf-281A / Wf-284C observation that the longer
G5 run pushed the Ru test_auc above the G4 30-epoch ceiling. The
honest-framing note is that seed=0 alone does not yet support a "50-epoch
beats 30-epoch" claim (need all 3 G5 seeds to compare populations).

## 7. `wave_budget_tracker.py` update

Added authoritative `TODO_180_G4_N_PAIRS = 15` constant + updated the G4
aggregate trigger tokens to recognize "15_pairs" / "15pairs" alongside
"14_pairs" / "14pairs" (the legacy tokens are kept for backward compat
with existing wave names that haven't been renamed).

**Files modified:**

- `molmetal/scripts/wave_budget_tracker.py`:
  - Added `TODO_180_G4_N_PAIRS = 15` constant with Wf-292A docstring citing the cascade provenance.
  - Updated aggregate_name_hits tuple to include `"15_pairs"` + `"15pairs"` triggers (kept `"14_pairs"` + `"14pairs"` for backward compat).
  - Updated `Wave-3:chemprop_v3_G4_14_pairs` → `Wave-3:chemprop_v3_G4_15_pairs` wave name.
  - Updated docstring "TODO-180 G4 14 PENDING" → "TODO-180 G4 15 PENDING".

The TODO_180_G4_ACTUAL_HR_BY_METAL per-metal hours are unchanged (Ru still
28.9 hr / Ir 7.7 hr / Os 1.9 hr / Rh 1.7 hr / Re 1.6 hr — the G5 cascade
absorbs into Ru 28.9 which already covered 3 seeds).

## 8. Verdict

| Check | Result |
|-------|--------|
| `g5_ru_seed0_path` | `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics_seed0.json` (existed at task start) |
| `per_seed_metrics_match` | YES — metal="Ru", seed=0, n_train=12471, test_auc=0.7716 (matches Wf-281A + Wf-284C) |
| `pairs_after_cascade` | **15** (was 14, +1 from G5 Ru seed=0) |
| `wave_budget_tracker_updated` | YES — `TODO_180_G4_N_PAIRS = 15` constant added + trigger tokens expanded |
| `cascade_verdict` | **SUCCESS** — TODO-180 G4 n=14 → n=15 gap closed via G5 cascade |

## 9. Files referenced

- `/home/hugo/codes/mol-metal/results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru/per_seed_metrics_seed0.json` (G5 source of truth, 2.1 KB)
- `/home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json` (symlink into G4 input-dir)
- `/home/hugo/codes/mol-metal/results/wf124_p3_remaining_metals_g4/training_summary.json` (updated, n_pairs_total=15)
- `/home/hugo/codes/mol-metal/molmetal/scripts/wf278a_g4_aggregate.py` (re-run, console output captured in §4)
- `/home/hugo/codes/mol-metal/molmetal/scripts/wave_budget_tracker.py` (TODO_180_G4_N_PAIRS=15 + trigger tokens updated)
