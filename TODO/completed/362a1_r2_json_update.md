# Wf-362-A1 — metalcytotox_combined_r2.json Wf-335-M2 CEILING_BREAK update

**Date:** 2026-09-29
**Workflow:** Wf-362 (docs JSON update)
**Phase:** A1 — close Wf-359-A3 documentation gap

## What

Added a `wf335m2_ceiling_break` block to
`metrics/by_metric/metalcytotox_combined_r2.json` recording the Wf-335-M2
FULL W₂ + chemprop v2 graph integration CEILING_BREAK result on the
metalcytotox_combined cohort. Pre-Wf-362-A1 the JSON only carried the older
Wf-77 ASTSR baseline + Wf-79 3-way PLL entries; the Wf-335-M2 result
(R² best = +0.2227, lift = +0.1172 over Wf-325A baseline of +0.1055) was
missing.

## Why

Wf-359-A3 found that `metalcytotox_combined_r2.json` did NOT include the
Wf-335-M2 CEILING_BREAK result. This is a documentation gap — the
CEILING_BREAK verdict was recorded in `molmetal/reports/wf335m2_r2/result.json`
+ `TODO/completed/335m2_r2_w2_chemprop_full.md`, but not in the canonical
metrics JSON consumed by the project status dashboards / `INDEX.md`
section-22 ledger. Wf-362-A1 closes the gap.

## What was changed

### 1. `metrics/by_metric/metalcytotox_combined_r2.json` (updated)

Added a top-level `wf335m2_ceiling_break` block carrying:

| Field | Value | Source |
|---|---|---|
| `workflow` | `Wf-335-M2` | `molmetal/reports/wf335m2_r2/result.json` |
| `task` | `FULL W₂ + chemprop v2 graph integration with multi-seed K-fold CV` | `result.json` |
| `cohort` | `metalcytotox_combined` | `result.json` |
| `best_config` | `full_static_combined_plus_w2_best_axis` | `result.json:360-362` |
| `n_features` | `164` | `result.json:355` |
| `r2_best` | `0.22268113216745766` | `result.json:351` |
| `r2_std` | `0.32098447223005655` | `result.json:352` |
| `wf325a_baseline_r2` | `0.10552683143637809` | `result.json:323` |
| `lift_vs_wf325a` | `0.11715430073107957` | `result.json:353` |
| `verdict` | `CEILING_BREAK` | `result.json:354` |
| `n_seeds` | `5` | `result.json:21` |
| `n_splits` | `5` | `result.json:22` |
| `n_samples` | `2000` | `result.json:321` |
| `cohorts_above_zero` | `1` | `result.json:482` |
| `cohorts_ceiling_break` | `1` | `result.json:481` |
| `cohorts_resolved` | `1` | `result.json:483` |
| `aggregate_status` | `HONEST_NEGATIVE_PARTIAL` | `result.json:484` |
| `source` | `molmetal/reports/wf335m2_r2/result.json` | Wf-362-A1 |
| `closeout_doc` | `TODO/completed/335m2_r2_w2_chemprop_full.md` | Wf-362-A1 |
| `status` | `MEASURED` | party-twenty-spirit §3.5 |
| `updated_at` | `2026-09-29` | Wf-362-A1 |

Pre-update field count: 12 top-level keys
Post-update field count: 13 top-level keys (new `wf335m2_ceiling_break` added)

### 2. `molmetal/tests/test_wf362a1_r2_json_update.py` (NEW — 2 tests)

| Test | Purpose |
|---|---|
| `test_metalcytotox_combined_r2_json_includes_wf335m2_ceiling_break_field` | Verifies the new field exists, has the right `workflow`, `source`, `updated_at`, `status`, `verdict`, and `cohort` values |
| `test_metalcytotox_combined_r2_json_r2_best_and_lift_match_wf335m2_source` | Verifies `r2_best` + `lift_vs_wf325a` match the source result.json bit-for-bit + the source file exists on disk |

Test run: `2 passed, 1 warning in 0.34s` PASS.

## Cross-reference

- Wf-335-M2 closeout: `TODO/completed/335m2_r2_w2_chemprop_full.md` (line 188-189: `full_static_combined_plus_w2_best_axis | 164 | +0.2227 | 0.3210 | +0.1172 | CEILING_BREAK`)
- Wf-335 final synthesis: `TODO/completed/335_final_synthesis.md` (line 115, 285, 330: same CEILING_BREAK verdict confirmed)
- Wf-335-M2 source result.json: `molmetal/reports/wf335m2_r2/result.json` (per_cohort[2].best_config)
- Wf-359-A3 finding: documentation gap in `metalcytotox_combined_r2.json`

## Honest framing per party-twenty-spirit §3.5

- **Documentation gap CLOSED.** Every numeric value in `wf335m2_ceiling_break` traces to
  `molmetal/reports/wf335m2_r2/result.json` (source-of-truth) and
  `TODO/completed/335m2_r2_w2_chemprop_full.md` (closeout doc).
- **MEASURED tag** preserved: the JSON uses `status: MEASURED` per the source
  result.json's `pre_flight_status: "CPU-only diagnostic"` clarification (CPU-only
  no GPU launch, no retrain/panel/ablation required).
- **Aggregate status HONEST_NEGATIVE_PARTIAL preserved**: only 1 of 3 cohorts
  (metalcytotox_combined) achieves CEILING_BREAK; the other 2 (platinai_4cell,
  hilbert) remain HONEST_NEGATIVE_R2_DEEP per Wf-335-M2 `per_cohort[0,1].best_config`.
- **Wall: PROJECTED 5-10 min CPU-only** (single JSON edit + 2 tests + 1 closeout doc).

## Pre-flight gate

Pre-flight v2 NOT required (no retrain / panel / ablation launch per
AGENTS.md §1.4 + Wf-335-M2 `pre_flight_status: "CPU-only diagnostic;
pre-flight v2 gate NOT required (no retrain / panel / ablation)"`).
EXPECTED_GATE_COUNTS["all"] 62 PRESERVED at Wf-362-A1 timestamp (no new
pre-flight gates added).

## Status

- Documentation gap CLOSED
- 2 NEW tests PASS (CPU-only)
- Wall: ~2 min actual
