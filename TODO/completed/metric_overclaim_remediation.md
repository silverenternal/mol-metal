# Metric overclaim remediation (B2) — 6 `metrics/by_metric/*.json` re-verified by recompute

**Date:** 2026-10-03
**Scope:** `metrics/by_metric/{nci60_precious_9cell_auc, nci60_precious_9cell_auc_3seed_5epoch, decode_ratio, metal_lomo_chemprop_v2, docked_pose_rmsd, platinai_combined_r2}.json` + this closeout.
**Rules honored:** no `paper/*` / `docs/PAPER_*` touches (§5.3); no wet-lab (§5.4/§10); no commit (orchestrator commits); no `molmetal/reports/*` edits. Every number below was recomputed from the file's own contents with the command shown in §2; every changed claim is recorded inside its JSON as a `corrections` entry (old value + new value + reason + date) with matching `honest_framing` text (§5.2 — nothing overwritten silently).

## §1 Per-file findings (recomputed vs stated)

| File | Old stated summary | Recomputed from its own data | Changed in file | Cited in metrics/+TODO/+docs/ |
|---|---|---|---|---|
| `nci60_precious_9cell_auc.json` (E1) | "9-cell-line" mean AUC 0.8007, `n_cells=9` | 0.80072017249287 over the **8** non-null cells; DU145 `auc_mean` null across all 3 seeds | `n_cells` 9→8 + panel/measured-count fields + description/framing labels. Headline value unchanged (it was always the 8-cell mean). MEASURED by §2-E1 | **41 files** (incl. `metrics/by_metric/_audit/m5_real_per_metric.json`, `TODO/completed/35_nci60_precious_9cell_shipped.md` — which already said "across 8 cells") |
| `nci60_precious_9cell_auc_3seed_5epoch.json` (E2) | 0.688 ± 0.0429, `n_cells=9` | 8 unique lines: mean **0.688325**, mean-of-per-cell-stds **0.0448375**. The 9-key mean (0.687967) double-counts HCT-116/HCT116 (byte-identical per-seed arrays); SKOV3 absent | value 0.688→0.688325, std 0.0429→0.0448375, `n_cells` 9→8, alias/absent fields, duplicate dashboard row removed | **0 files** (never cited outside itself — the double-count did not propagate) |
| `decode_ratio.json` (E3) | `aggregate.all_measured_cfm_decode_ratio = 0.998`, presented as an all-measurements aggregate | 12 measurements, sum 5.4725 → **unweighted mean 0.456042**; **sample-weighted 0.881302** (N=7641); pre-YuelBond-era mean 0.1875; YuelBond-era mean 0.993125 | value **retained** (claim corrected — see §3): explicit definition added, all true recomputes stored as separate fields, `all_measured_cfm_decode_ratio_REAL` alias added, `corrections` record added | **56 files** (wide citation) |
| `metal_lomo_chemprop_v2.json` (E4) | `n_completed=1` beside `n_runs_completed=19` | 19 non-null entries in `per_fold_seed_auc` (Ru 5 + Ir 5 + Os 4 + Rh 5 + Re 0) | `n_completed` 1→19 + definition field | **14 files** (incl. `TODO/completed/321f_chemprop_v2_status_reconciliation.md`, which already reported "n_completed: 19 / 25") |
| `docked_pose_rmsd.json` (E5) | note: "15 runs = 5 pockets x 3 seeds" | **13** measured per-seed runs (3+3+1+3+3; 3ny8 succeeded 1/3 seeds) | note + description run count; `n_runs_measured=13`, `n_runs_planned=15`, `runs_missing` | **55 files** |
| `platinai_combined_r2.json` (E6) | `n_methods=6` | len(`methods`)=**7** (= len(`per_method`); generator `promote_metrics.py::build_platinai_combined_r2` defines `n_methods=len(per_source)`); **5** entries carry a measured R² (wf79_p2 null, wf125_p3 deferred) | `n_methods` 6→7 + explicit measured-count fields | **14 files** |

## §2 Recompute commands (each run 2026-10-03 with `.venv/bin/python`, output shown)

- E1: `import json; d=json.load(open('metrics/by_metric/nci60_precious_9cell_auc.json')); m=[v['auc_mean'] for v in d['per_cell_line'].values() if v['auc_mean'] is not None]; print(len(m), sum(m)/len(m))` → `8 0.8007201724928701`
- E2: same pattern over keys minus `HCT-116` → `8 0.6883250000000001 0.0448375` (mean, mean-of-stds)
- E3: `v=[m['value'] for m in d['measurements']]; …` → `12 5.4725 0.4560416666666667 0.8813023164507263` (n, sum, unweighted mean, sample-weighted mean)
- E4: count non-null `per_fold_seed_auc` → `19`
- E5: sum of `len(m['per_seed_best_rmsd_angstrom'])` over `measurements` → `13`
- E6: `len(d['methods'])` and count of populated R² → `7 5`

## §3 E3 verdict — stated plainly

- **The audit's core point stands.** A field named `all_measured_cfm_decode_ratio` carried a single hand-set measurement. The hand-set is disclosed inside the file (`wf398a3_fix_note`: 0.0 → 0.998, Wf-398-A3), so it was not a silent invention — but presenting one measurement under an aggregate name with no definition is a real §5.2-alignment defect, and it was **live**: hand-set 2026-09-30 per the in-file note, first committed `bc29aa90d6` (2026-10-02 14:37 +0800), fixed today 2026-10-03. Before that the same field held a stale 0.0 (also not an aggregate of the then-list) from the file's earliest git record (`4527e4f34e`, 2026-09-18) through `9b12e6f574` (2026-09-28).
- **The audit's arithmetic is wrong and I did not adopt it.** "mean ≈ 0.3727" is not reproducible: recomputing over all 12 stored values gives 5.4725/12 = **0.456042**. 0.3727 only appears by dropping the Path B bond-bearing 1.0 (pre-RDKit-strict) entry from the *numerator* while still dividing by 12. (I also stored the sample-weighted mean 0.881302; neither pooled number is a meaningful "current decode ratio" because the list deliberately spans a superseded architecture (5 zeros) and two metric variants (Path B bond-bearing vs RDKit-strict).)
- **Why the value 0.998 was retained (claim corrected instead).** Repo code defines this exact field as "latest MEASURED value": `pre_flight_check.py::gate_p3_mock_to_real_m14_decode_ratio_REAL_30_probes` (now reads the explicit `_REAL` alias first) and `molmetal/tests/test_wf397a1_decode_ratio_fix.py::test_wf397a1_aggregate_metric_matches_latest_measurement` (asserts plain field == latest post-Wf-124 measurement). Setting the plain field to the pooled mean would fail that regression test and misstate the current decoder state (0.998 is a genuine MEASURED value, Wf-148 5979/5991). Verified post-fix: `pytest molmetal/tests/test_wf397a1_decode_ratio_fix.py` → **4 passed**. If the project wants the plain field to carry the pooled mean, the gate + test must be updated in the same change — flagged here; that edit was outside this task's explicit scope (metrics + closeout only).

## §4 Evidence (§5.5 filesystem-truth + §5.6 atomic write)

```
$ ls -la metrics/by_metric/nci60_precious_9cell_auc.json metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json metrics/by_metric/decode_ratio.json metrics/by_metric/metal_lomo_chemprop_v2.json metrics/by_metric/docked_pose_rmsd.json metrics/by_metric/platinai_combined_r2.json
-rw-r--r-- 1 hugo hugo 15258 Oct  3 09:29 metrics/by_metric/decode_ratio.json
-rw-r--r-- 1 hugo hugo  9853 Oct  3 09:29 metrics/by_metric/docked_pose_rmsd.json
-rw-r--r-- 1 hugo hugo  7109 Oct  3 09:29 metrics/by_metric/metal_lomo_chemprop_v2.json
-rw-r--r-- 1 hugo hugo  7736 Oct  3 09:29 metrics/by_metric/nci60_precious_9cell_auc_3seed_5epoch.json
-rw-r--r-- 1 hugo hugo 12505 Oct  3 09:29 metrics/by_metric/nci60_precious_9cell_auc.json
-rw-r--r-- 1 hugo hugo 19595 Oct  3 09:29 metrics/by_metric/platinai_combined_r2.json
```

- `git status --short metrics/by_metric/` shows exactly these 6 files modified (no other metric touched).
- All 6 JSONs re-parse via `json.load`; each carries its `corrections` list (counts: E1 2, E2 4, E3 1, E4 1, E5 2, E6 1).
- All 6 writes used `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` (§5.6); this closeout was written via `write_closeout`. Note: `atomic_write_file` leaves 0600 perms (mkstemp default); the 6 files were chmod'd back to 644 to match their prior state. Worth fixing in the helper in a future wave.
- Work-tree only — **no commit** (orchestrator commits).

## §5 Honest framing / caveats

- All six reported items were **real** inconsistencies (no false positives in the set), except that E1's headline *number* was already correct as an 8-cell mean — only its "9-cell" label was wrong; and E3's number is a genuine measurement whose *claim* was wrong, not an invented value, plus the audit's expected 0.3727 was itself wrong (true 0.456042).
- Corrections by magnitude: E1/E4/E5/E6 = label/count fixes, headline values unchanged; E2 = value 0.688→0.688325 and std 0.0429→0.0448375 on the correct denominator (8 unique lines); E3 = claim corrected, headline current-state decode ratio unchanged at 0.998 with the pooled recomputes recorded.
- `platinai_combined_r2.json` is partly regenerated by `molmetal/scripts/promote_metrics.py` (line 741); the added `corrections`/count fields would not survive a re-promotion unless the generator is updated (out of scope here — noted for a future wave).
