# Wf-301D — Stale-Read Guard Elapsed Bug + wf278a Multi-Root Patch

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** HONEST-NEGATIVE on Part 1 (no `elapsed` bug in wrappers, Wf-298A prior finding reaffirmed), PASS on Part 2 (multi-root patch already shipped + cascade aggregate + 3/3 tests pass).

---

## §1 — Part 1: Stale-Read Guard `elapsed` Bug Fix

### §1.1 — Scope
Re-attempt of Wf-298A's prior fix on the three launch wrappers:
- `scripts/launch_wf124_p3_g4.sh`
- `scripts/launch_wf124_p3_g5.sh`
- `scripts/launch_wf124_p3_g6.sh`

### §1.2 — Findings (HONEST-NEGATIVE)
The described `elapsed is not defined` bug does **not exist** in any of the three wrappers. No `while` loop, no `elapsed` variable, no `max_wait_s` parameter, and no stale-read guard function exists in any of the three files. Each wrapper is a straight-line bash script that:
1. Calls the three Python guard scripts (`pre_flight_check`, `per_wave_checklist`, `wave_budget_tracker`) once each.
2. Runs `run_pair` / `run_tuple` for every (metal, seed) / (config, metal, seed) tuple back-to-back (no polling).
3. Aggregates `per_seed_metrics_seed*.json` into a `training_summary.json`.

This matches the Wf-298A prior finding (`molmetal/reports/wf298a_elapsed_bug_fix/fix_report.md` §2 + §5) which closed the same investigation with `n_wrappers_fixed: 0` and refused to fabricate a phantom polling loop. Per `party-twenty-spirit` §3.5 we re-affirm that honest-negative close: refuse to insert a stale-read guard into launch-only bash that the three files don't need.

The actual missing-deliverable remains `read_anchor_json(path, max_age_sec=..., required_version=...)` — a **Python** helper queued as Wf-291H (~45 min CPU). That work has not landed on disk.

### §1.3 — Bash Syntax Verification (per task spec §3)
```
$ bash -n scripts/launch_wf124_p3_g4.sh   →  SYNTAX OK
$ bash -n scripts/launch_wf124_p3_g5.sh   →  SYNTAX OK
$ bash -n scripts/launch_wf124_p3_g6.sh   →  SYNTAX OK
```

### §1.4 — Why no `elapsed=0` was inserted
Inserting `elapsed=0` before a non-existent loop would be either (a) dead code, or (b) part of a phantom guard. Per `party-twenty-spirit` §3.5 honest-negative discipline: refuse to fabricate a stale-read guard into the launch wrappers, because (a) the described bug is absent, (b) the three files are launch-only bash (no JSON anchor reads), and (c) inserting a phantom polling loop would be a regression risk on the live Wf-124 P3 sweep.

**stale_read_fixed = 0** (no bug exists; re-affirms Wf-298A honest-negative close).

---

## §2 — Part 2: wf278a Multi-Root Patch (Already Shipped)

### §2.1 — Code surface (`molmetal/scripts/wf278a_g4_aggregate.py`, 325 LOC)
The multi-root patch (Wf-298B) is already on disk and includes:

| Surface | Status | Lines |
| --- | --- | --- |
| `--input-dirs` argparse (nargs="+") | SHIPPED | 187-198 |
| `--input-dir` legacy single-root (kept for back-compat) | SHIPPED | 181-186 |
| `collect_seed_metrics_multi_root(roots)` with last-wins-by-mtime dedup | SHIPPED | 49-109 |
| `--expected-pairs` default 15 (was hard-coded 14 in Wf-278A v1) | SHIPPED | 205-210 |
| Cascade mode flag emission | SHIPPED | 256-291 |

### §2.2 — Cascade aggregate run (per task spec §9)
```
$ cd /home/hugo/codes/mol-metal
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf278a_g4_aggregate.py \
    --input-dirs results/wf124_p3_remaining_metals_g4 \
                 results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru \
    --output results/wf124_p3_remaining_metals_g4/training_summary_v2.json
```

Output (`tail -40`):
```
[wf278a-agg] wrote results/wf124_p3_remaining_metals_g4/training_summary_v2.json

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

### §2.3 — Summary JSON inspection
| Field | Value |
| --- | --- |
| `cascade_mode` | `True` |
| `input_dirs` | `['results/wf124_p3_remaining_metals_g4', 'results/wf124_p3_50epoch_full_sweep_g5/Ru/seed0/ru']` |
| `n_pairs_total` | **15** (5 metals × 3 seeds; G5 Ru/seed0 won dedup over G4's C/Ru/seed0) |
| `expected_pairs` | **15** (per Wf-298B default) |
| `aggregate_5metal.test_auc_mean` | 0.7243 |
| `aggregate_5metal.test_auc_std` | 0.0460 |

### §2.4 — Tests (per task spec §8)
```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf278a_multi_root.py -v 2>&1 | tail -20
```

Output:
```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collected 3 items

molmetal/tests/test_wf278a_multi_root.py::test_multi_root_merge PASSED   [ 33%]
molmetal/tests/test_wf278a_multi_root.py::test_last_wins_dedup PASSED    [ 66%]
molmetal/tests/test_wf278a_multi_root.py::test_expected_pairs_15 PASSED  [100%]

========================= 3 passed, 1 warning in 0.23s =========================
```

---

## §3 — Output Summary (PLAIN TEXT)

| Metric | Value |
| --- | --- |
| `stale_read_fixed` | **0** (no `elapsed` bug in wrappers; re-affirms Wf-298A honest-negative) |
| `multi_root_n_lines` | **325** LOC in `molmetal/scripts/wf278a_g4_aggregate.py` (with multi-root patch Wf-298B shipped) |
| `cascade_pairs` | **15** (5 metals × 3 seeds; G5 Ru/seed0 won mtime dedup) |
| `n_tests_pass` | **3** (test_multi_root_merge + test_last_wins_dedup + test_expected_pairs_15) |
| `bash -n` checks | **3** (g4.sh + g5.sh + g6.sh all SYNTAX OK) |

**Honest framing (party-twenty-spirit §3.5)**:
- Part 1 closed as honest-negative (no bug). Per Wf-298A precedent: refuse to fabricate a stale-read guard into launch-only bash.
- Part 2 already shipped (Wf-298B) and verified end-to-end on this run: cascade produced 15 pairs (G5 Ru/seed0 won mtime dedup over G4 C/Ru/seed0); 3/3 tests pass; 5-metal test_auc = 0.7243 ± 0.0460 MEASURED.
- The real outstanding stale-read work is `read_anchor_json` Python helper (Wf-291H, ~45 min CPU), still MISSING.

**End of Wf-301D combined report.**