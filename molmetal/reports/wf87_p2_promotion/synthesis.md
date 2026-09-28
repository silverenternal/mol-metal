# Wf-87 Phase 2 — Real JSON Promotion to `metrics/by_metric/`

**Date**: 2026-09-26  
**Workflow**: Wf-87 Phase 2 — promote MEASURED JSONs from `molmetal/reports/`
to `metrics/by_metric/`  
**Constraint**: CPU-only; no GPU retrain; paper-gate (TODO-84) honored;
party-twenty-spirit §3.5 honest framing applied throughout

## 1. Aggregate

- Promoter rules registered: **8**
- Promoted (PROMOTED): **8**
- Skipped (SKIPPED_EXISTS / SKIPPED_NO_SOURCE): **0**
- Build errors: **0**
- New files written: **8** (all in `metrics/by_metric/`)

## 2. Promotion ledger

| Target file | Source(s) | n_sources | Status | Headline value |
|---|---|---:|---|---:|
| `hilbert_4cell_r2.json` | `wf83_p1_phase2_4cell/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json` | 4 | PROMOTED | aggregate_combined_r2_mean = **-6.7126** (MEASURED_BELOW_GATE) |
| `platinai_combined_r2.json` | `wf77_p1_hilbert_baseline/platinai_4cell` + `wf77_p2_astr_baseline/platinai_4cell` + `wf79_p2_3way_pll/platinai_4cell` | 3 | PROMOTED | baseline_astr_r2_mean = **-0.4028** (MEASURED) |
| `metalcytotox_combined_r2.json` | `wf77_p2_astr_baseline/metalcytotox_5metal` + `wf79_p2_3way_pll/metalcytotox_5metal` | 2 | PROMOTED | per-method baseline **-0.2841** (MEASURED) |
| `tmqm_combined_r2.json` | `wf71_p3_wf49_cross_round2/tmqm_dblock/per_cohort_metrics` + `wf71_p3_wf49_cross_round2/cross_dataset_summary` | 2 | PROMOTED | cross_dataset_macro_r2_mean = **+0.1248** (MEASURED) |
| `ood_per_metal_reachability.json` | `wf73_p2_ood_permetal/{Ru,Ir,Os,Rh,Re}/ood_reachability` | 5 | PROMOTED | aggregate_ood_pass_rate = **1.000** (MEASURED) |
| `ood_per_element_reachability.json` | `wf73_p3_ood_perelement/{Co,Ni,Cu,Au,Hg}/ood_reachability` | 5 | PROMOTED | aggregate_ood_pass_rate = **0.650** (MEASURED) |
| `dmpnn_ckpt_status.json` | `.ckpt/` directory scan | 0 | PROMOTED | n_checkpoints = **2** (affinity_platinai_seed0/1) |
| `vina_status.json` | `metrics/by_metric/vina_kcal_per_mol.json` + filesystem probe | 0 | PROMOTED | meeko_installed = **false**; vina_binary_available = **false** |

## 3. Honest-framing notes

- **All values come from on-disk MEASURED evidence.** No fabrication, no
  GPU retrain, no PROJECTED token leaked into the central metric files
  (`test_promote_no_projected_in_central` PASS).
- **Collision detection is honoured.** First run produces zero collisions;
  any subsequent run without `--force` preserves the existing target
  file (`test_promote_existing_target_preserved` PASS).
- **MEASURED_BELOW_GATE is preserved.** `hilbert_4cell_r2.json` carries
  `aggregate_combined_r2_mean = -6.7126` with `status =
  MEASURED_BELOW_GATE` and `verdict = MEASURED_HONEST_NEGATIVE_R2`. We
  do not hide the negative R² — per `party-twenty-spirit §3.5` the
  learned predictor is worse than the constant-mean baseline on this
  cohort and that signal is load-bearing for the honest framing.
- **`metalcytotox_combined_r2.json` per-method rows may carry `null`
  values** when the source JSON has a different schema (e.g. Wf-79
  per_method records 5-seed mean R² but the Wf-79 source JSON omits a
  top-level `overall_r2_mean`). The honest `null` is preserved and the
  file-level `honest_framing_note` calls this out.
- **`dmpnn_ckpt_status.json`** is a directory listing, not a model
  inference. The file ships `n_checkpoints` + per-file
  `filename / size_bytes / mtime_utc` for two affinity checkpoints on
  disk. Downstream consumers should still load `metal_lomo_auc.json`
  / `platinai_3cell_resistance.json` for MEASURED AUC values.
- **`vina_status.json`** records the read-only probe of `meeko` import
  + `vina --version` binary probe (both `false` on this host); the
  real-Vina path remains blocked per Wf-30 P4 close-out.

## 4. Test coverage (Wf-87 P2 close-out)

- 17 tests in `molmetal/tests/test_promote_metrics.py` — all PASS in 0.37s.
- Cover: smoke run, ≥5 NEW files gate, no-collisions-first-run,
  collision preservation on second run, per-file content checks
  (status, n_metrics, headline values), honest-framing fields,
  no-PROJECTED-leak, promotion log shape, `--dry-run` skip,
  `--force` overwrite.

## 5. Cross-references

- `molmetal/scripts/promote_metrics.py` (NEW, 956 LOC)
- `molmetal/tests/test_promote_metrics.py` (NEW, 395 LOC, 17 tests)
- `molmetal/reports/wf87_p2_promotion/promotion_log.json` (NEW, 14 KB)
- `molmetal/reports/wf87_p1_disk_audit/per_metric_status.json` (input — P1 disk audit)
- `TODO/INDEX.md` §17 (22 conceptual metrics × 5 categories)
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 (0-tolerance honest framing)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate)