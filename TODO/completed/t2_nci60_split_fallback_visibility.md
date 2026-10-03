# T2-FIX — NCI-60 random-split fallback made visible (helper + callers)

**Status: SHIPPED (2026-10-03).** Task: T2-FIX (sibling of T1 `--max-joint-rows`).
Scope: make the silent scaffold->random fallback loud + machine-readable, WITHOUT
changing the split itself (nci60 split stays byte-identical to m21 / 0.8007 baseline).

## §0 Disk evidence (AGENTS.md §5.5)

```
$ ls -la molmetal/scripts/_nci60_scaffold_helpers.py \
        molmetal/scripts/nci60_9cell_chemprop_driver.py \
        molmetal/tests/test_t2_nci60_split_visibility.py
-rw-r--r-- 1 hugo hugo 43347 Oct  3 15:06 molmetal/scripts/nci60_9cell_chemprop_driver.py
-rw-r--r-- 1 hugo hugo 10664 Oct  3 15:05 molmetal/scripts/_nci60_scaffold_helpers.py
-rw-r--r-- 1 hugo hugo  9022 Oct  3 15:07 molmetal/tests/test_t2_nci60_split_visibility.py
$ git diff --stat -- molmetal/scripts/_nci60_scaffold_helpers.py molmetal/scripts/nci60_9cell_chemprop_driver.py
 helper: 105 ++-   driver: 150 ++-   (224 insertions, 31 deletions)
```

## §1 What changed

**Helper `_nci60_scaffold_helpers.py` (my lane):**
1. `None` fallback is LOUD: `logging.warning` + `RuntimeWarning` naming seed,
   `n_test_min`, n_rows, n_groups, max_group_size, and
   `path_taken=random_fallback`. Was a silent `return None`.
2. NEW `write_split_manifest(path, records, extra=...)` — JSON
   `nci60_split_manifest_v1` with `path_taken` per (cell, seed) so a gate can
   assert `path_taken == "scaffold_ood"`. Returned/unknown values are `None`,
   never NaN. ~55 LOC.
3. NEW `SPLIT_PATH_{SCAFFOLD_OOD,RANDOM_FALLBACK,RANDOM_TEMPORAL,UNKNOWN}`
   constants (exported).
4. Docstring arithmetic fix: MCF7 `28 -> 29`, DU145 `9 -> 8` (banker's
   rounding; MEASURED 2026-10-03; matches the wf319e test values).

**Driver `nci60_9cell_chemprop_driver.py` (caller — see §4 justification):**
5. `Nci60CellRun.path_taken` field (default `"unknown"`), set in every branch:
   `scaffold_ood` / `random_fallback` / `random_temporal`. Flows into
   `per_seed_records[]` automatically via `asdict`.
6. Driver-level `logger.warning("SPLIT FALLBACK ...")` in both fallback
   branches (seed-level; plus per-cell ACTUAL warning with cell name and the
   plan it contradicts).
7. `per_cell_split_actual` (OUTCOME) dict + `split_planned`/`path_taken` keys
   in every `per_cell_summary[cell]` + `split_manifest_path` field. The
   misleading `per_cell_split_choice` (PLAN, said "scaffold" for cells that
   actually ran random) is kept for compat and now labelled "(PLAN)" next to
   "(OUTCOME)".
8. Split manifest written next to the sweep JSON
   (`<out stem>.split_manifest.json`) on every execute_sweep.
9. `honest_framing_note` corrected: the old text claimed `--split auto` with
   auto-tuning "actually triggers scaffold-OOD for A549/MCF7/A2780" — MEASURED
   FALSE (see §2; helper is never consulted in auto mode because
   n_test_random_est >= per-cell threshold for 9/9 cells).

**Split construction: UNTOUCHED** (no index, threshold, or seed logic changed).

## §2 Per-cell measurement (the deliverable)

Command: `.venv/bin/python` on `molmetal/data/nci60_precious_9cell.csv`,
real helper, `seed=0`, wall 12.3 s for 3 Murcko passes x 9 cells:

```
cell       rows maxgrp path@50(n_test_min=50) autoThr path@auto auto-mode outcome
A549       2647     37 random_fallback          31 scaffold_ood random_temporal
MCF7       1964     34 random_fallback          29 scaffold_ood random_temporal
A2780      1095     37 random_fallback          31 scaffold_ood random_temporal
HCT116      701     20 random_fallback          17 scaffold_ood random_temporal
A2780cis    564     30 random_fallback          26 scaffold_ood random_temporal
PC3         275      9 random_fallback           8 scaffold_ood random_temporal
HL60        225     18 random_fallback          15 scaffold_ood random_temporal
SKOV3       191     18 random_fallback          15 scaffold_ood random_temporal
DU145       161     10 random_fallback           8 scaffold_ood random_temporal
```

**MEASURED: at the default `--split scaffold --n-test-min 50`, ALL NINE cells
fall back to random 80/10/20 (9/9).** Every historical `--split scaffold` run
was silently random while labelled scaffold. In `--split auto` (auto-tuning ON,
default) every `n_test_random_est` >= its per-cell threshold, so all 9 cells
resolve to temporal and the helper is never consulted; with
`--no-n-test-min-auto`, HL60/SKOV3/DU145 (n_test_random_est 45/38/32 < 50) plan
scaffold_fallback and then also land on random_fallback. Auto-tuning applies
only in auto mode — `--split scaffold` uses `--n-test-min` verbatim.

## §3 Tests + end-to-end smoke

NEW `molmetal/tests/test_t2_nci60_split_visibility.py` (6 tests):
fallback warns on both channels (with seed/threshold in the message);
success path stays quiet; manifest round-trip + gate assertion; constants
exported; **real-CSV tripwire: 9/9 cells return None + emit exactly 9 warnings
at n_test_min=50**; and an in-process `execute_sweep` smoke on a 60-row
synthetic cohort proving the caller records
`per_cell_split_actual=random_fallback`, a `SPLIT FALLBACK` log line, and the
manifest.

```
$ .venv/bin/python -m pytest test_t2_nci60_split_visibility.py \
      test_wf319e_...py test_wf315f_...py -q
26 passed, 12 warnings in 16.70s     (6 new + 20 pre-existing)
```

End-to-end driver smoke (subprocess, CPU, real `--execute`):
- fallback run (`--n-test-min 50`): exit 0; stderr contains BOTH the helper
  `NCI-60 scaffold split FALLBACK` and driver `SPLIT FALLBACK (seed=0)`;
  `per_cell_split_actual={'T2SMOKE':'random_fallback'}`;
  `manifest path_taken=['random_fallback']`; gate(scaffold_ood) -> False (as it
  must be). Success run (`--n-test-min 5`): no fallback warning,
  `path_taken=scaffold_ood`, gate -> True; 5.4 s wall.

## §4 Constraints honored / honest notes

- **m21 comparability**: split bytes untouched; changes are warning/record-only.
- **Lane**: `pre_flight_check.py` (other workflow, live-edited) NOT touched, nor
  hooks/configs. `train_dmpnn_multitask_wasserstein.py` NOT touched by me (it
  was already modified by the sibling T1 agent). The driver WAS edited —
  required by the task's "make the caller state the path in its own output";
  it was git-clean at edit time. If a parallel task also owns the driver, this
  is the overlap surface to watch.
- **§5.7: N/A** — no trainer flag wire-in; no trainer touched. No
  decode_ratio claim (D-MPNN driver, no decoder) and none fabricated.
- No commit, no GPU, no paper, no wet-lab. `.venv/bin/python` used throughout.
- **Manifest index-md5 deferred**: adding md5 of train/val/test index arrays
  would require plumbing indices out of `run_one_per_cell` (+ field + record
  edits, ~10 LOC); cost stated, not done — counts + path + label are shipped.
- Pre-existing, unrelated: `test_wf320c_todo220_full_impl.py` has 2 failures
  (missing legacy docs `TODO/completed/220_wf220_temporal_split_acceptance.md`
  and `220_wf219_nci60_9cell_acceptance.md`) — file-existence checks, not
  affected by this diff.

**Cross-refs**: review findings (T2) in the T1/T2 workflow brief;
`molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py` (threshold
auto-tuning); `molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py`
(split labels); `TODO/completed/330b2_optimization_integration.md` §1.1
(sibling T1 context).
